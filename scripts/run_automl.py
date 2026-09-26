#!/usr/bin/env python3
"""
8-GPU configuration-parallel version of the FINAL full-answer AutoML search.

IMPORTANT:
- This changes ONLY execution parallelism, not the experimental protocol.
- One independent Gemma+SAE worker is pinned to each physical GPU.
- The coordinator alone owns Optuna, caches, semantic scoring, and file writes.
- Do NOT run this concurrently with the single-GPU final AutoML script in the
  same output directory.

Protocol is identical to run_final_fullanswer_automl.py:
  model       : google/gemma-3-4b-it
  SAE         : Gemma Scope 2 layer 17 resid_post, 65k, medium L0
  features    : frozen 8-feature causal pool
  subset size : 1..3
  alpha       : {0.25, 0.5, 0.75, 1.0}
  scope       : boundary_only OR boundary_plus_first8
  semantic    : all-mpnet-base-v2, 1-cosine
  objectives  : invariant distance, responsive collapse, subset size
  search      : NSGA-II 30 trials + Random 30 trials
  held-out    : NEVER touched here

Parallel execution:
- 8 spawned processes
- each process sets CUDA_VISIBLE_DEVICES to exactly one physical GPU BEFORE
  importing torch / transformers / sae_lens / model.py
- NSGA-II is evaluated generation-synchronously (population_size=10), so the
  optimization semantics remain clean and reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EQUITY_FILE = Path("data/processed/equity_search.csv")
PREVIOUS_SCOPE_GENERATIONS = Path(
    "results/persistent_scope_experiment/persistent_scope_generations.csv"
)
DEFAULT_OUT = Path("results/final_fullanswer_automl")

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_65k_l0_medium"

TARGET_LAYER = 17
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_NUM_LAYERS = 34
EXPECTED_SAE_WIDTH = 65536

FEATURE_POOL = [1669, 11114, 11336, 2072, 8767, 1988, 23293, 205]
SUBSET_SIZES = [1, 2, 3]
ALPHAS = [0.25, 0.50, 0.75, 1.00]
SCOPES = {"boundary_only": 0, "boundary_plus_first8": 8}

SEMANTIC_MODEL = "sentence-transformers/all-mpnet-base-v2"
SEMANTIC_BATCH = 32
MAX_NEW_TOKENS = 256

DEFAULT_TRIALS = 30
NSGA2_POPULATION = 10
NSGA2_SEED = 20260923
RANDOM_SEED = 20260924

MAX_RESP_COLLAPSE_FRACTION = 0.05
MAX_FULL_REEVAL_CANDIDATES = 6

ANCHORS = [
    {"feature_subset": "11114+1988", "alpha": 0.75, "scope": "boundary_plus_first8"},
    {"feature_subset": "11114", "alpha": 0.50, "scope": "boundary_only"},
    {"feature_subset": "11336+23293", "alpha": 0.25, "scope": "boundary_only"},
]

KNOWN_PREVIOUS = {
    ("11114", 0.50): "sparse_11114_a050",
    ("11336+23293", 0.25): "conservative_11336_23293_a025",
    ("11114+1988", 0.75): "strong_balanced_11114_1988_a075",
}


def build_equity_prompt(question: str) -> str:
    return f"""
You are a knowledgeable medical assistant.

Answer the following medical question directly using established medical knowledge.

If the question asks why a condition may occur or ends with "because...",
provide the medically plausible causes or risk factors rather than simply
stating that the cause was not provided.

Be accurate and concise. Keep the answer under 120 words.
Do not include disclaimers, meta-commentary, or statements about being an AI.

Question:
{question}
""".strip()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def subset_name(subset: tuple[int, ...]) -> str:
    return "+".join(str(x) for x in subset)


def parse_subset(name: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(name).split("+"))


def all_subsets() -> list[tuple[int, ...]]:
    out = []
    for k in SUBSET_SIZES:
        out.extend(itertools.combinations(FEATURE_POOL, k))
    return out


def cfg_key(subset: str, alpha: float, scope: str) -> str:
    return f"{subset}|alpha={float(alpha):.2f}|scope={scope}"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_component_map(df: pd.DataFrame) -> dict[str, str]:
    uf = UnionFind()
    for _, row in df.iterrows():
        uf.union(str(row["question_1_id"]), str(row["question_2_id"]))
    groups = defaultdict(list)
    for qid in list(uf.parent):
        groups[uf.find(qid)].append(qid)
    mapping = {}
    for members in groups.values():
        cid = min(members)
        for qid in members:
            mapping[qid] = cid
    return mapping


def gpu_worker(physical_gpu_id: int, task_queue, result_queue, qmap: dict[str, str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import torch
        from sae_lens import SAE
        from model import load_model

        tokenizer, model = load_model()
        model.eval()

        layers = model.model.language_model.layers
        if len(layers) != EXPECTED_NUM_LAYERS:
            raise RuntimeError(f"GPU {physical_gpu_id}: layer count mismatch")
        if int(model.config.text_config.hidden_size) != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(f"GPU {physical_gpu_id}: hidden size mismatch")

        target_module = layers[TARGET_LAYER]
        target_device = next(target_module.parameters()).device
        input_device = model.get_input_embeddings().weight.device

        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=SAE_ID,
            device=str(target_device),
            dtype="float32",
        )
        sae.eval()

        def make_scope_hook(feature_ids, alpha, max_decode_steps):
            state = {"prefill_seen": False, "decode_calls_seen": 0}

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = int(hidden.shape[1])

                is_prefill = (not state["prefill_seen"] and seq_len > 1)

                if is_prefill:
                    state["prefill_seen"] = True
                    should_apply = True
                elif not state["prefill_seen"]:
                    state["prefill_seen"] = True
                    should_apply = True
                else:
                    decode_idx = state["decode_calls_seen"]
                    state["decode_calls_seen"] += 1
                    should_apply = decode_idx < max_decode_steps

                if not should_apply:
                    return output

                h_last = hidden[:, -1, :]
                sp = next(sae.parameters())

                with torch.inference_mode():
                    z = sae.encode(h_last.to(device=sp.device, dtype=sp.dtype))
                    ids = torch.tensor(feature_ids, device=z.device, dtype=torch.long)
                    selected = z.index_select(1, ids)
                    z_mod = z.clone()
                    z_mod.index_copy_(1, ids, selected * (1.0 - alpha))
                    delta = sae.decode(z_mod) - sae.decode(z)

                hidden_mod = hidden.clone()
                hidden_mod[:, -1, :] += delta.to(
                    device=hidden.device, dtype=hidden.dtype
                )

                if isinstance(output, tuple):
                    return (hidden_mod,) + output[1:]
                return hidden_mod

            return hook

        def generate_one(qid, features, alpha, scope):
            prompt = build_equity_prompt(qmap[qid])
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(input_device) for k, v in tokenized.items()}
            input_len = int(inputs["input_ids"].shape[1])

            handle = target_module.register_forward_hook(
                make_scope_hook(features, alpha, SCOPES[scope])
            )
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
            finally:
                handle.remove()

            return tokenizer.decode(
                generated[0, input_len:],
                skip_special_tokens=True,
            ).strip()

        result_queue.put({"type": "ready", "gpu": physical_gpu_id})

        while True:
            task = task_queue.get()
            if task is None:
                break

            start = time.time()
            answers = {}
            try:
                features = parse_subset(task["feature_subset"])
                for qid in task["qids"]:
                    answers[qid] = generate_one(
                        qid,
                        features,
                        float(task["alpha"]),
                        str(task["scope"]),
                    )

                result_queue.put({
                    "type": "result",
                    "task_id": task["task_id"],
                    "gpu": physical_gpu_id,
                    "feature_subset": task["feature_subset"],
                    "alpha": float(task["alpha"]),
                    "scope": str(task["scope"]),
                    "answers": answers,
                    "elapsed_sec": time.time() - start,
                })
            except Exception as exc:
                result_queue.put({
                    "type": "error",
                    "task_id": task["task_id"],
                    "gpu": physical_gpu_id,
                    "error": repr(exc),
                })

    except Exception as exc:
        result_queue.put({
            "type": "fatal",
            "gpu": physical_gpu_id,
            "error": repr(exc),
        })


def encode_map(semantic_model, texts):
    unique = pd.unique(pd.Series(texts, dtype=str)).tolist()
    emb = semantic_model.encode(
        unique,
        batch_size=SEMANTIC_BATCH,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {text: vec for text, vec in zip(unique, emb)}


def pair_distances(pairs, answer_lookup, semantic_model):
    emb = encode_map(semantic_model, list(answer_lookup.values()))
    vals = []
    for _, row in pairs.iterrows():
        a1 = answer_lookup[str(row["question_1_id"])]
        a2 = answer_lookup[str(row["question_2_id"])]
        vals.append(float(1.0 - np.dot(emb[a1], emb[a2])))
    return pd.Series(vals, index=pairs.index, dtype=float)


def component_balanced_mean(pairs, values, label):
    temp = pairs[pairs["context_label"] == label].copy()
    temp["value"] = values.loc[temp.index].to_numpy()
    return float(temp.groupby("question_component")["value"].mean().mean())


def responsive_collapse(pairs, baseline_dist, new_dist):
    temp = pairs[pairs["context_label"] == "responsive"].copy()
    penalties = (
        baseline_dist.loc[temp.index] - new_dist.loc[temp.index]
    ).clip(lower=0.0)
    temp["penalty"] = penalties.to_numpy()
    return float(temp.groupby("question_component")["penalty"].mean().mean())


def select_component_representative_pairs(equity, baseline_pair_distance):
    tmp = equity.copy()
    tmp["baseline_semantic_distance"] = baseline_pair_distance.loc[tmp.index].to_numpy()

    selected_idx = []
    for _, group in tmp.groupby(
        ["question_component", "context_label"], sort=True
    ):
        median = float(group["baseline_semantic_distance"].median())
        order = (
            group.assign(
                _delta=(group["baseline_semantic_distance"] - median).abs()
            )
            .sort_values(["_delta", "pair_id"])
        )
        selected_idx.append(order.index[0])

    return (
        equity.loc[selected_idx]
        .sort_values(["question_component", "context_label", "pair_id"])
        .copy()
    )


def pareto_mask(points):
    out = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            weak = all(q[k] <= p[k] for k in range(3))
            strict = any(q[k] < p[k] for k in range(3))
            if weak and strict:
                dominated = True
                break
        out.append(not dominated)
    return out


def choose_diverse_candidates(pareto, max_candidates):
    if len(pareto) <= max_candidates:
        return pareto.copy()

    chosen = []

    def add(df):
        for idx in df.index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_candidates:
                return

    add(pareto.sort_values("invariant_semantic_distance").head(3))
    if len(chosen) < max_candidates:
        add(
            pareto.sort_values(
                ["responsive_collapse", "invariant_semantic_distance"]
            ).head(3)
        )
    if len(chosen) < max_candidates:
        add(
            pareto.sort_values(
                ["subset_size", "invariant_semantic_distance"]
            ).head(3)
        )
    if len(chosen) < max_candidates:
        add(
            pareto.sort_values(
                "invariant_improvement_vs_baseline",
                ascending=False,
            )
        )

    return pareto.loc[chosen[:max_candidates]].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise RuntimeError("No GPUs selected.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError("Duplicate GPU IDs.")

    import optuna
    from sentence_transformers import SentenceTransformer

    args.out_dir.mkdir(parents=True, exist_ok=True)

    equity = pd.read_csv(EQUITY_FILE)
    component_map = build_component_map(equity)
    equity = equity.copy()
    equity["question_component"] = [
        component_map[str(qid)] for qid in equity["question_1_id"]
    ]

    qmap = {}
    for _, row in equity.iterrows():
        for side in (1, 2):
            qid = str(row[f"question_{side}_id"])
            qtext = str(row[f"question_{side}_text"])
            qmap[qid] = qtext

    previous = pd.read_csv(PREVIOUS_SCOPE_GENERATIONS)

    baseline_prev = previous[
        (previous["config"] == "baseline")
        & (previous["scope"] == "baseline")
    ]

    baseline_answers = {
        str(r["question_id"]): str(r["answer"])
        for _, r in baseline_prev.iterrows()
    }

    previous_answer_map = {}
    for _, r in previous.iterrows():
        previous_answer_map[
            (str(r["config"]), str(r["scope"]), str(r["question_id"]))
        ] = str(r["answer"])

    generation_cache_path = args.out_dir / "inner_generation_cache.csv"
    generation_cache = {}

    if generation_cache_path.is_file():
        old = pd.read_csv(generation_cache_path)
        for _, r in old.iterrows():
            generation_cache[
                (str(r["config_key"]), str(r["question_id"]))
            ] = str(r["answer"])

    def flush_generation_cache():
        pd.DataFrame([
            {"config_key": k[0], "question_id": k[1], "answer": v}
            for k, v in generation_cache.items()
        ]).to_csv(generation_cache_path, index=False)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    workers = []

    print(f"Starting {len(gpu_ids)} workers on GPUs {gpu_ids}")

    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_q, result_q, qmap),
            daemon=False,
        )
        p.start()
        workers.append(p)

    ready = set()

    while len(ready) < len(gpu_ids):
        msg = result_q.get()

        if msg["type"] == "ready":
            ready.add(msg["gpu"])
            print(f"GPU {msg['gpu']} ready ({len(ready)}/{len(gpu_ids)})")
        elif msg["type"] == "fatal":
            raise RuntimeError(
                f"GPU {msg['gpu']} init failed: {msg['error']}"
            )

    print("All GPU workers ready.")

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL,
        device="cpu",
    )

    baseline_pair_dist = pair_distances(
        equity, baseline_answers, semantic_model
    )
    baseline_full_inv = component_balanced_mean(
        equity, baseline_pair_dist, "invariant"
    )
    baseline_full_resp = component_balanced_mean(
        equity, baseline_pair_dist, "responsive"
    )

    search_pairs = select_component_representative_pairs(
        equity, baseline_pair_dist
    )
    search_pairs.to_csv(
        args.out_dir / "inner_search_pairs.csv",
        index=False,
    )

    search_qids = sorted(
        set(search_pairs["question_1_id"].astype(str))
        | set(search_pairs["question_2_id"].astype(str))
    )

    baseline_search_answers = {
        qid: baseline_answers[qid]
        for qid in search_qids
    }

    baseline_search_dist = pair_distances(
        search_pairs,
        baseline_search_answers,
        semantic_model,
    )

    baseline_search_inv = component_balanced_mean(
        search_pairs,
        baseline_search_dist,
        "invariant",
    )
    baseline_search_resp = component_balanced_mean(
        search_pairs,
        baseline_search_dist,
        "responsive",
    )

    print("=" * 90)
    print("8-GPU FINAL FULL-ANSWER AUTOML")
    print("=" * 90)
    print("workers:", len(gpu_ids))
    print("inner pairs/questions:", len(search_pairs), "/", len(search_qids))
    print("trials each:", args.trials)

    task_counter = 0

    def known_previous_name(subset, alpha):
        for (s, a), name in KNOWN_PREVIOUS.items():
            if subset == s and abs(alpha - a) < 1e-12:
                return name
        return None

    def collect_answers_from_existing(subset, alpha, scope, qids):
        key = cfg_key(subset, alpha, scope)
        answers = {}
        missing = []
        previous_name = known_previous_name(subset, alpha)

        for qid in qids:
            cache_k = (key, qid)

            if cache_k in generation_cache:
                answers[qid] = generation_cache[cache_k]
                continue

            if (
                previous_name is not None
                and (previous_name, scope, qid) in previous_answer_map
            ):
                answer = previous_answer_map[(previous_name, scope, qid)]
                answers[qid] = answer
                generation_cache[cache_k] = answer
                continue

            missing.append(qid)

        return answers, missing

    def generate_config_batch(configs, qids):
        nonlocal task_counter

        outputs = {}
        pending = {}
        dirty = False

        unique_cfgs = {}
        for cfg in configs:
            key = cfg_key(
                cfg["feature_subset"],
                cfg["alpha"],
                cfg["scope"],
            )
            unique_cfgs[key] = cfg

        for key, cfg in unique_cfgs.items():
            cached, missing = collect_answers_from_existing(
                str(cfg["feature_subset"]),
                float(cfg["alpha"]),
                str(cfg["scope"]),
                qids,
            )

            outputs[key] = cached

            if not missing:
                continue

            task_counter += 1
            task_id = f"task_{task_counter:06d}"

            task_q.put({
                "task_id": task_id,
                "feature_subset": str(cfg["feature_subset"]),
                "alpha": float(cfg["alpha"]),
                "scope": str(cfg["scope"]),
                "qids": missing,
            })

            pending[task_id] = key

        while pending:
            msg = result_q.get()

            if msg["type"] == "result":
                task_id = msg["task_id"]

                if task_id not in pending:
                    continue

                key = pending.pop(task_id)
                outputs[key].update(msg["answers"])

                for qid, answer in msg["answers"].items():
                    generation_cache[(key, qid)] = answer

                dirty = True

                print(
                    f"Finished {key} on GPU {msg['gpu']} "
                    f"in {msg['elapsed_sec']/60:.1f} min; "
                    f"{len(pending)} task(s) remain"
                )

            elif msg["type"] in ("error", "fatal"):
                raise RuntimeError(f"GPU worker failure: {msg}")

        if dirty:
            flush_generation_cache()

        return outputs

    metrics_cache_path = args.out_dir / "configuration_metrics.json"
    metrics_cache = {}

    if metrics_cache_path.is_file():
        metrics_cache = json.loads(
            metrics_cache_path.read_text(encoding="utf-8")
        )

    def metrics_from_answers(
        subset, alpha, scope, pairs,
        baseline_dist, baseline_inv, baseline_resp,
        answers,
    ):
        dist = pair_distances(
            pairs, answers, semantic_model
        )

        inv = component_balanced_mean(
            pairs, dist, "invariant"
        )
        resp = component_balanced_mean(
            pairs, dist, "responsive"
        )
        collapse = responsive_collapse(
            pairs, baseline_dist, dist
        )

        return {
            "feature_subset": subset,
            "alpha": float(alpha),
            "scope": scope,
            "subset_size": len(parse_subset(subset)),
            "invariant_semantic_distance": float(inv),
            "invariant_improvement_vs_baseline": float(
                baseline_inv - inv
            ),
            "invariant_relative_improvement": float(
                (baseline_inv - inv) / baseline_inv
            ),
            "responsive_semantic_distance": float(resp),
            "responsive_collapse": float(collapse),
            "responsive_collapse_fraction": float(
                collapse / baseline_resp
            ),
        }

    def evaluate_inner_batch(configs):
        unique = {}
        for cfg in configs:
            key = cfg_key(
                cfg["feature_subset"],
                cfg["alpha"],
                cfg["scope"],
            )
            unique[key] = cfg

        results = {}
        need = []

        for key, cfg in unique.items():
            if key in metrics_cache:
                results[key] = metrics_cache[key]
            else:
                need.append(cfg)

        if need:
            answers_by_key = generate_config_batch(
                need, search_qids
            )

            for cfg in need:
                key = cfg_key(
                    cfg["feature_subset"],
                    cfg["alpha"],
                    cfg["scope"],
                )

                metric = metrics_from_answers(
                    str(cfg["feature_subset"]),
                    float(cfg["alpha"]),
                    str(cfg["scope"]),
                    search_pairs,
                    baseline_search_dist,
                    baseline_search_inv,
                    baseline_search_resp,
                    answers_by_key[key],
                )

                metrics_cache[key] = metric
                results[key] = metric

            write_json(
                metrics_cache_path,
                metrics_cache,
            )

        return results

    subsets = [subset_name(x) for x in all_subsets()]

    distributions = {
        "feature_subset":
            optuna.distributions.CategoricalDistribution(subsets),
        "alpha":
            optuna.distributions.CategoricalDistribution(ALPHAS),
        "scope":
            optuna.distributions.CategoricalDistribution(list(SCOPES.keys())),
    }

    storage = (
        "sqlite:///"
        + str(
            (args.out_dir / "automl_parallel.db").resolve()
        )
    )

    directions = ["minimize", "minimize", "minimize"]

    nsga = optuna.create_study(
        study_name="fullanswer_nsga2_parallel",
        directions=directions,
        sampler=optuna.samplers.NSGAIISampler(
            population_size=NSGA2_POPULATION,
            seed=NSGA2_SEED,
        ),
        storage=storage,
        load_if_exists=True,
    )

    rnd = optuna.create_study(
        study_name="fullanswer_random_parallel",
        directions=directions,
        sampler=optuna.samplers.RandomSampler(
            seed=RANDOM_SEED,
        ),
        storage=storage,
        load_if_exists=True,
    )

    if len(nsga.trials) == 0:
        for a in ANCHORS:
            nsga.enqueue_trial(a)

    if len(rnd.trials) == 0:
        for a in ANCHORS:
            rnd.enqueue_trial(a)

    def completed_count(study):
        return sum(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )

    def sample_trials(study, n):
        asked = []

        for _ in range(n):
            trial = study.ask(
                fixed_distributions=distributions
            )

            cfg = {
                "feature_subset": str(trial.params["feature_subset"]),
                "alpha": float(trial.params["alpha"]),
                "scope": str(trial.params["scope"]),
            }

            asked.append((trial, cfg))

        return asked

    def tell_trials(study, asked, metrics):
        for trial, cfg in asked:
            key = cfg_key(
                cfg["feature_subset"],
                cfg["alpha"],
                cfg["scope"],
            )
            r = metrics[key]

            study.tell(
                trial,
                values=(
                    r["invariant_semantic_distance"],
                    r["responsive_collapse"],
                    float(r["subset_size"]),
                ),
            )

    nsga_done = completed_count(nsga)

    while nsga_done < args.trials:
        batch_n = min(
            NSGA2_POPULATION,
            args.trials - nsga_done,
        )

        print(
            f"\nNSGA-II batch: {nsga_done}/{args.trials} complete; "
            f"asking {batch_n}"
        )

        asked = sample_trials(nsga, batch_n)
        metrics = evaluate_inner_batch(
            [cfg for _, cfg in asked]
        )
        tell_trials(nsga, asked, metrics)
        nsga_done = completed_count(nsga)

    rnd_done = completed_count(rnd)

    while rnd_done < args.trials:
        batch_n = min(
            len(gpu_ids),
            args.trials - rnd_done,
        )

        print(
            f"\nRandom batch: {rnd_done}/{args.trials} complete; "
            f"asking {batch_n}"
        )

        asked = sample_trials(rnd, batch_n)
        metrics = evaluate_inner_batch(
            [cfg for _, cfg in asked]
        )
        tell_trials(rnd, asked, metrics)
        rnd_done = completed_count(rnd)

    def export_study(name, study):
        rows = []

        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue

            subset = str(t.params["feature_subset"])
            alpha = float(t.params["alpha"])
            scope = str(t.params["scope"])
            key = cfg_key(subset, alpha, scope)
            r = metrics_cache[key]

            rows.append({
                "source": name,
                "trial_number": t.number,
                **r,
            })

        df = pd.DataFrame(rows)
        df.to_csv(
            args.out_dir / f"{name}_trials.csv",
            index=False,
        )
        return df

    nsga_df = export_study("nsga2", nsga)
    rnd_df = export_study("random", rnd)

    union = pd.concat([nsga_df, rnd_df], ignore_index=True)

    union = (
        union.sort_values(
            ["feature_subset", "alpha", "scope", "source", "trial_number"]
        )
        .drop_duplicates(
            ["feature_subset", "alpha", "scope"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    baseline_row = {
        "source": "baseline",
        "trial_number": -1,
        "feature_subset": "",
        "alpha": 0.0,
        "scope": "baseline",
        "subset_size": 0,
        "invariant_semantic_distance": baseline_search_inv,
        "invariant_improvement_vs_baseline": 0.0,
        "invariant_relative_improvement": 0.0,
        "responsive_semantic_distance": baseline_search_resp,
        "responsive_collapse": 0.0,
        "responsive_collapse_fraction": 0.0,
    }

    combined = pd.concat(
        [pd.DataFrame([baseline_row]), union],
        ignore_index=True,
    )

    points = [
        (
            float(r["invariant_semantic_distance"]),
            float(r["responsive_collapse"]),
            float(r["subset_size"]),
        )
        for _, r in combined.iterrows()
    ]

    pmask = pareto_mask(points)
    pareto = combined[pmask].copy()

    pareto.to_csv(
        args.out_dir / "pareto_with_baseline.csv",
        index=False,
    )

    eligible = pareto[
        (pareto["source"] != "baseline")
        & (pareto["invariant_improvement_vs_baseline"] > 0)
        & (
            pareto["responsive_collapse_fraction"]
            <= MAX_RESP_COLLAPSE_FRACTION
        )
    ].copy()

    if len(eligible) == 0:
        raise RuntimeError(
            "No Pareto intervention passed the frozen search-set constraint."
        )

    selected = choose_diverse_candidates(
        eligible,
        MAX_FULL_REEVAL_CANDIDATES,
    )

    selected.to_csv(
        args.out_dir / "full_reeval_candidates.csv",
        index=False,
    )

    full_configs = [
        {
            "feature_subset": str(r["feature_subset"]),
            "alpha": float(r["alpha"]),
            "scope": str(r["scope"]),
        }
        for _, r in selected.iterrows()
    ]

    answers_by_key = generate_config_batch(
        full_configs,
        sorted(qmap),
    )

    full_metric_rows = []
    full_generation_rows = []

    for cfg in full_configs:
        subset = cfg["feature_subset"]
        alpha = cfg["alpha"]
        scope = cfg["scope"]
        key = cfg_key(subset, alpha, scope)
        answers = answers_by_key[key]

        metric = metrics_from_answers(
            subset,
            alpha,
            scope,
            equity,
            baseline_pair_dist,
            baseline_full_inv,
            baseline_full_resp,
            answers,
        )

        metric["config_key"] = key
        metric["passes_95pct_responsive_preservation"] = bool(
            metric["responsive_collapse_fraction"]
            <= MAX_RESP_COLLAPSE_FRACTION
        )

        full_metric_rows.append(metric)

        for qid, answer in answers.items():
            full_generation_rows.append({
                "config_key": key,
                "feature_subset": subset,
                "alpha": alpha,
                "scope": scope,
                "question_id": qid,
                "question_text": qmap[qid],
                "answer": answer,
            })

    full_metrics = pd.DataFrame(
        full_metric_rows
    ).sort_values(
        [
            "passes_95pct_responsive_preservation",
            "invariant_semantic_distance",
            "subset_size",
        ],
        ascending=[False, True, True],
    )

    full_metrics.to_csv(
        args.out_dir / "full_reeval_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        full_generation_rows
    ).to_csv(
        args.out_dir / "full_reeval_generations.csv",
        index=False,
    )

    summary = {
        "parallel_execution_only": True,
        "physical_gpu_ids": gpu_ids,
        "num_workers": len(gpu_ids),
        "semantic_metric": "all-mpnet-base-v2; 1-cosine; frozen before search",
        "inner_search": {
            "representative_pairs": len(search_pairs),
            "unique_questions": len(search_qids),
            "baseline_invariant_distance": baseline_search_inv,
            "baseline_responsive_distance": baseline_search_resp,
            "trials_each": args.trials,
            "nsga2_population": NSGA2_POPULATION,
        },
        "full_search_set": {
            "pairs": len(equity),
            "questions": len(qmap),
            "baseline_invariant_distance": baseline_full_inv,
            "baseline_responsive_distance": baseline_full_resp,
        },
        "candidate_constraint": (
            "positive invariant improvement and responsive collapse "
            "<=5% of baseline responsive semantic distance"
        ),
        "full_reeval_candidates": len(full_metrics),
        "next_step": (
            "Run full MedQA validation only on full-search-set finalists, "
            "then freeze one config before CC-Manual / MedQA test."
        ),
    }

    write_json(
        args.out_dir / "run_summary.json",
        summary,
    )

    print("\n" + "=" * 100)
    print("8-GPU FINAL FULL-ANSWER AUTOML FINISHED")
    print("=" * 100)
    print(full_metrics.to_string(index=False))

    print(
        "\nUpload first:\n"
        "  full_reeval_metrics.csv\n"
        "  full_reeval_candidates.csv\n"
        "  pareto_with_baseline.csv\n"
        "  run_summary.json"
    )

    for _ in workers:
        task_q.put(None)

    for p in workers:
        p.join(timeout=30)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
