#!/usr/bin/env python3
"""
Post-hoc SAE activation-shift audit across CC-LLM and CC-Manual.

Purpose
-------
Diagnose WHY the sparse SAE interventions optimized on CC-LLM did not transfer
to CC-Manual.

This is an exploratory mechanism audit AFTER the external held-out result.
It does NOT:
- change the model, SAE, prompt, feature pool, alpha, or scope;
- re-select a final configuration;
- re-run AutoML;
- turn CC-Manual back into a held-out test.

What is measured
----------------
The original feature screening used:
  Gemma-3-4B-it
  Gemma Scope 2, layer 17 resid_post, width 65k, medium L0
  last prompt token / assistant-generation boundary.

The final intervention acted at:
  boundary + first 8 autoregressive decode steps.

Therefore this audit measures the same eight frozen causal features at exactly
those 9 intervention opportunities under BASELINE greedy decoding:
  step 0 = prompt boundary
  step 1..8 = first eight decode calls.

For each source (CC-LLM / CC-Manual), label (invariant / responsive), and
feature, it summarizes:
- activation coverage;
- activation magnitude;
- boundary |z1-z2|;
- integrated activation mass across the 9-step scope;
- integrated pair |mass1-mass2|.

It also reports aggregate metrics for the frozen final configuration:
  features = {11114, 8767, 1988}

No fairness conclusion should be drawn from this audit alone.  It is descriptive
evidence about representation / activation distribution shift.

Outputs
-------
results/sae_activation_shift_audit/
  question_step_activations.csv
  pair_feature_metrics.csv
  pair_final_config_metrics.csv
  feature_group_summary.csv
  final_config_group_summary.csv
  feature_label_contrast.csv
  source_shift_summary.csv
  run_summary.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEARCH_FILE = Path("data/processed/equity_search.csv")
MANUAL_FILE = Path("data/processed/equity_test.csv")
DEFAULT_OUT = Path("results/sae_activation_shift_audit")

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_65k_l0_medium"

TARGET_LAYER = 17
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_NUM_LAYERS = 34
EXPECTED_SAE_WIDTH = 65536

# Frozen 8-feature causal pool used by the AutoML search.
FEATURE_POOL = [
    1669,
    11114,
    11336,
    2072,
    8767,
    1988,
    23293,
    205,
]

# Frozen final configuration that was selected BEFORE CC-Manual was inspected.
FINAL_FEATURES = [11114, 8767, 1988]

# Exactly the intervention opportunity window:
# prompt boundary + first eight decode calls = 9 layer forward calls.
N_SCOPE_STEPS = 9
GEN_NEW_TOKENS = 9


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_prompt(question: str) -> str:
    # Frozen v5 prompt used throughout the project.
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


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str):
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def add_component_ids(df: pd.DataFrame) -> pd.DataFrame:
    uf = UnionFind()

    for _, row in df.iterrows():
        uf.union(
            str(row["question_1_id"]),
            str(row["question_2_id"]),
        )

    groups = defaultdict(list)

    for qid in list(uf.parent):
        groups[uf.find(qid)].append(qid)

    mapping = {}

    for members in groups.values():
        cid = min(members)
        for qid in members:
            mapping[qid] = cid

    out = df.copy()
    out["question_component"] = [
        mapping[str(qid)]
        for qid in out["question_1_id"]
    ]
    return out


def build_question_rows(df: pd.DataFrame) -> list[dict]:
    qmap: dict[str, str] = {}

    for _, row in df.iterrows():
        for side in (1, 2):
            qid = str(row[f"question_{side}_id"])
            qtext = str(row[f"question_{side}_text"])

            if qid in qmap and qmap[qid] != qtext:
                raise RuntimeError(
                    f"Conflicting text for question id {qid}"
                )

            qmap[qid] = qtext

    return [
        {
            "question_id": qid,
            "question_text": qmap[qid],
        }
        for qid in sorted(qmap)
    ]


def source_worker(
    physical_gpu: int,
    source: str,
    question_rows: list[dict],
    result_queue,
    offline: bool,
):
    # Must be set before importing torch / transformers / sae_lens / model.py.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        from sae_lens import SAE
        from model import load_model

        tokenizer, model = load_model()
        model.eval()

        layers = model.model.language_model.layers

        if len(layers) != EXPECTED_NUM_LAYERS:
            raise RuntimeError(
                f"Expected {EXPECTED_NUM_LAYERS} layers, got {len(layers)}"
            )

        hidden_size = int(model.config.text_config.hidden_size)

        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"Expected hidden size {EXPECTED_HIDDEN_SIZE}, got {hidden_size}"
            )

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

        if int(sae.cfg.d_in) != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError("SAE d_in mismatch.")

        if int(sae.cfg.d_sae) != EXPECTED_SAE_WIDTH:
            raise RuntimeError("SAE width mismatch.")

        sae_param = next(sae.parameters())

        feature_ids_gpu = torch.tensor(
            FEATURE_POOL,
            device=sae_param.device,
            dtype=torch.long,
        )

        output_rows = []
        start = time.time()

        for i, qrow in enumerate(question_rows):
            qid = str(qrow["question_id"])
            question = str(qrow["question_text"])

            messages = [
                {
                    "role": "user",
                    "content": build_prompt(question),
                }
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(input_device)
                if torch.is_tensor(v)
                else v
                for k, v in inputs.items()
            }

            input_len = int(inputs["input_ids"].shape[1])

            captured_h = []

            def hook(_module, _inputs, output):
                if len(captured_h) >= N_SCOPE_STEPS:
                    return

                hidden = (
                    output[0]
                    if isinstance(output, tuple)
                    else output
                )

                if not torch.is_tensor(hidden):
                    raise RuntimeError(
                        f"Unexpected layer output type: {type(hidden)}"
                    )

                # Exactly the representation used in the original screening:
                # the last token of each relevant forward call.
                captured_h.append(
                    hidden[:, -1, :]
                    .detach()
                    .float()
                    .cpu()
                )

            handle = target_module.register_forward_hook(hook)

            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=GEN_NEW_TOKENS,
                        do_sample=False,
                        use_cache=True,
                    )
            finally:
                handle.remove()

            generated_len = int(
                generated.shape[1] - input_len
            )

            if not captured_h:
                raise RuntimeError(
                    f"No layer activations captured for {qid}"
                )

            h = torch.cat(captured_h, dim=0)

            if h.ndim != 2 or h.shape[1] != EXPECTED_HIDDEN_SIZE:
                raise RuntimeError(
                    f"Unexpected captured shape for {qid}: {tuple(h.shape)}"
                )

            with torch.inference_mode():
                z = sae.encode(
                    h.to(
                        device=sae_param.device,
                        dtype=sae_param.dtype,
                    )
                )

                z_sel = z.index_select(
                    1,
                    feature_ids_gpu,
                ).float().cpu()

            # If EOS occurs before all 9 opportunities, missing future
            # intervention calls truly do not occur; represent them as zeros.
            padded = torch.zeros(
                (N_SCOPE_STEPS, len(FEATURE_POOL)),
                dtype=torch.float32,
            )

            n_captured = min(
                int(z_sel.shape[0]),
                N_SCOPE_STEPS,
            )

            padded[:n_captured] = z_sel[:n_captured]

            for step in range(N_SCOPE_STEPS):
                for j, fid in enumerate(FEATURE_POOL):
                    val = float(padded[step, j].item())

                    output_rows.append(
                        {
                            "source": source,
                            "question_id": qid,
                            "step": step,
                            "step_name": (
                                "boundary"
                                if step == 0
                                else f"decode_{step}"
                            ),
                            "step_available": bool(step < n_captured),
                            "feature_id": int(fid),
                            "activation": val,
                            "is_active": bool(val > 0.0),
                            "captured_steps": int(n_captured),
                            "generated_tokens": int(generated_len),
                        }
                    )

            if (
                (i + 1) % 20 == 0
                or i + 1 == len(question_rows)
            ):
                print(
                    f"[GPU {physical_gpu}] {source}: "
                    f"{i + 1}/{len(question_rows)} questions",
                    flush=True,
                )

        result_queue.put(
            {
                "type": "result",
                "gpu": physical_gpu,
                "source": source,
                "elapsed_sec": time.time() - start,
                "rows": output_rows,
            }
        )

    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "gpu": physical_gpu,
                "source": source,
                "error": repr(exc),
            }
        )


def component_balanced_group_summary(
    pair_metrics: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for keys, group in pair_metrics.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {
            col: val
            for col, val in zip(group_cols, keys)
        }

        base["n_pairs"] = int(len(group))
        base["n_components"] = int(
            group["question_component"].nunique()
        )

        for metric in metric_cols:
            base[f"{metric}_pair_mean"] = float(
                group[metric].mean()
            )

            component_means = (
                group.groupby(
                    "question_component"
                )[metric]
                .mean()
            )

            base[
                f"{metric}_component_balanced_mean"
            ] = float(
                component_means.mean()
            )

        rows.append(base)

    return pd.DataFrame(rows)


def make_pair_metrics(
    pair_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    source: str,
):
    trace_source = trace_df[
        trace_df["source"] == source
    ].copy()

    # qid -> [9, 8]
    q_arrays: dict[str, np.ndarray] = {}

    for qid, group in trace_source.groupby(
        "question_id",
        sort=False,
    ):
        pivot = (
            group.pivot(
                index="step",
                columns="feature_id",
                values="activation",
            )
            .reindex(
                index=range(N_SCOPE_STEPS),
                columns=FEATURE_POOL,
                fill_value=0.0,
            )
        )

        q_arrays[str(qid)] = pivot.to_numpy(
            dtype=np.float64
        )

    feature_rows = []
    config_rows = []

    final_indices = [
        FEATURE_POOL.index(fid)
        for fid in FINAL_FEATURES
    ]

    for _, row in pair_df.iterrows():
        q1 = str(row["question_1_id"])
        q2 = str(row["question_2_id"])

        a = q_arrays[q1]
        b = q_arrays[q2]

        common = {
            "source": source,
            "pair_id": str(row["pair_id"]),
            "context_label": str(row["context_label"]),
            "question_component": str(row["question_component"]),
            "question_1_id": q1,
            "question_2_id": q2,
        }

        for j, fid in enumerate(FEATURE_POOL):
            x = a[:, j]
            y = b[:, j]
            both = np.concatenate([x, y])

            positive = both[both > 0.0]

            feature_rows.append(
                {
                    **common,
                    "feature_id": int(fid),
                    "boundary_side_active_rate":
                        float(
                            np.mean(
                                np.array(
                                    [x[0] > 0.0, y[0] > 0.0]
                                )
                            )
                        ),
                    "boundary_any_active":
                        float(
                            (x[0] > 0.0)
                            or (y[0] > 0.0)
                        ),
                    "boundary_mean_activation":
                        float(
                            (x[0] + y[0]) / 2.0
                        ),
                    "boundary_abs_delta":
                        float(
                            abs(x[0] - y[0])
                        ),
                    "scope_side_step_active_rate":
                        float(
                            np.mean(both > 0.0)
                        ),
                    "scope_any_active":
                        float(
                            np.any(both > 0.0)
                        ),
                    "scope_mean_activation_per_opportunity":
                        float(
                            np.mean(both)
                        ),
                    "scope_active_magnitude":
                        float(
                            positive.mean()
                            if len(positive)
                            else 0.0
                        ),
                    "scope_integrated_mass_per_side":
                        float(
                            (
                                np.sum(x)
                                + np.sum(y)
                            )
                            / 2.0
                        ),
                    "scope_integrated_pair_abs_delta":
                        float(
                            abs(
                                np.sum(x)
                                - np.sum(y)
                            )
                        ),
                }
            )

        # Aggregate metrics for the frozen 3-feature final configuration.
        aa = a[:, final_indices]  # [9,3]
        bb = b[:, final_indices]

        side_step_any = np.concatenate(
            [
                np.any(aa > 0.0, axis=1),
                np.any(bb > 0.0, axis=1),
            ]
        )

        config_rows.append(
            {
                **common,
                "boundary_any_feature_side_rate":
                    float(
                        np.mean(
                            [
                                np.any(aa[0] > 0.0),
                                np.any(bb[0] > 0.0),
                            ]
                        )
                    ),
                "boundary_activation_mass_per_side":
                    float(
                        (
                            np.sum(aa[0])
                            + np.sum(bb[0])
                        )
                        / 2.0
                    ),
                "scope_any_feature_step_rate":
                    float(
                        np.mean(side_step_any)
                    ),
                "scope_active_feature_fraction":
                    float(
                        np.mean(
                            np.concatenate(
                                [aa.ravel(), bb.ravel()]
                            )
                            > 0.0
                        )
                    ),
                "scope_activation_mass_per_side":
                    float(
                        (
                            np.sum(aa)
                            + np.sum(bb)
                        )
                        / 2.0
                    ),
                "scope_pair_mass_abs_delta":
                    float(
                        abs(
                            np.sum(aa)
                            - np.sum(bb)
                        )
                    ),
            }
        )

    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(config_rows),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpus",
        default="7,1",
        help=(
            "Exactly two physical GPUs: first for CC-LLM, "
            "second for CC-Manual."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Allow Hugging Face network access. By default the script "
            "uses the already-populated local HF cache in offline mode."
        ),
    )

    args = parser.parse_args()

    gpu_ids = [
        int(x.strip())
        for x in args.gpus.split(",")
        if x.strip()
    ]

    if len(gpu_ids) != 2:
        raise RuntimeError(
            "Exactly two GPUs are required, e.g. --gpus 7,1"
        )

    if len(set(gpu_ids)) != 2:
        raise RuntimeError("GPU IDs must be unique.")

    for path in (SEARCH_FILE, MANUAL_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    search = add_component_ids(
        pd.read_csv(SEARCH_FILE)
    )

    manual = add_component_ids(
        pd.read_csv(MANUAL_FILE)
    )

    search_rows = build_question_rows(search)
    manual_rows = build_question_rows(manual)

    print("=" * 90)
    print("SAE ACTIVATION-SHIFT AUDIT")
    print("=" * 90)
    print(
        "CC-LLM:",
        len(search),
        "pairs /",
        len(search_rows),
        "questions /",
        search["question_component"].nunique(),
        "components",
    )
    print(
        "CC-Manual:",
        len(manual),
        "pairs /",
        len(manual_rows),
        "questions /",
        manual["question_component"].nunique(),
        "components",
    )
    print("features:", FEATURE_POOL)
    print("final features:", FINAL_FEATURES)
    print(
        "scope:",
        "boundary + first 8 decode calls",
    )
    print(
        "HF mode:",
        "online"
        if args.online
        else "offline/local-cache",
    )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    tasks = [
        (
            gpu_ids[0],
            "cc_llm",
            search_rows,
        ),
        (
            gpu_ids[1],
            "cc_manual",
            manual_rows,
        ),
    ]

    workers = []

    for gpu, source, rows in tasks:
        p = ctx.Process(
            target=source_worker,
            args=(
                gpu,
                source,
                rows,
                queue,
                not args.online,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    all_rows = []
    completed = 0

    while completed < 2:
        msg = queue.get()

        if msg["type"] == "error":
            raise RuntimeError(
                f"Worker failed: {msg}"
            )

        all_rows.extend(msg["rows"])
        completed += 1

        print(
            f"Finished {msg['source']} "
            f"on GPU {msg['gpu']} in "
            f"{msg['elapsed_sec']/60:.1f} min",
            flush=True,
        )

    for p in workers:
        p.join(timeout=30)

    trace_df = pd.DataFrame(all_rows)

    trace_df.to_csv(
        args.out_dir
        / "question_step_activations.csv",
        index=False,
    )

    search_feature, search_config = (
        make_pair_metrics(
            search,
            trace_df,
            "cc_llm",
        )
    )

    manual_feature, manual_config = (
        make_pair_metrics(
            manual,
            trace_df,
            "cc_manual",
        )
    )

    pair_feature = pd.concat(
        [search_feature, manual_feature],
        ignore_index=True,
    )

    pair_config = pd.concat(
        [search_config, manual_config],
        ignore_index=True,
    )

    pair_feature.to_csv(
        args.out_dir
        / "pair_feature_metrics.csv",
        index=False,
    )

    pair_config.to_csv(
        args.out_dir
        / "pair_final_config_metrics.csv",
        index=False,
    )

    feature_metric_cols = [
        "boundary_side_active_rate",
        "boundary_any_active",
        "boundary_mean_activation",
        "boundary_abs_delta",
        "scope_side_step_active_rate",
        "scope_any_active",
        "scope_mean_activation_per_opportunity",
        "scope_active_magnitude",
        "scope_integrated_mass_per_side",
        "scope_integrated_pair_abs_delta",
    ]

    config_metric_cols = [
        "boundary_any_feature_side_rate",
        "boundary_activation_mass_per_side",
        "scope_any_feature_step_rate",
        "scope_active_feature_fraction",
        "scope_activation_mass_per_side",
        "scope_pair_mass_abs_delta",
    ]

    feature_summary = (
        component_balanced_group_summary(
            pair_feature,
            [
                "source",
                "context_label",
                "feature_id",
            ],
            feature_metric_cols,
        )
    )

    config_summary = (
        component_balanced_group_summary(
            pair_config,
            [
                "source",
                "context_label",
            ],
            config_metric_cols,
        )
    )

    feature_summary.to_csv(
        args.out_dir
        / "feature_group_summary.csv",
        index=False,
    )

    config_summary.to_csv(
        args.out_dir
        / "final_config_group_summary.csv",
        index=False,
    )

    # Recreate the original "invariant minus responsive" association idea
    # for the most relevant activation-difference metrics.
    contrast_rows = []

    contrast_metrics = [
        "boundary_abs_delta",
        "scope_integrated_pair_abs_delta",
        "scope_integrated_mass_per_side",
        "scope_side_step_active_rate",
    ]

    for source in ["cc_llm", "cc_manual"]:
        for fid in FEATURE_POOL:
            row = {
                "source": source,
                "feature_id": fid,
            }

            for metric in contrast_metrics:
                col = (
                    f"{metric}_component_balanced_mean"
                )

                inv = feature_summary[
                    (feature_summary["source"] == source)
                    & (
                        feature_summary[
                            "context_label"
                        ]
                        == "invariant"
                    )
                    & (
                        feature_summary[
                            "feature_id"
                        ]
                        == fid
                    )
                ][col]

                resp = feature_summary[
                    (feature_summary["source"] == source)
                    & (
                        feature_summary[
                            "context_label"
                        ]
                        == "responsive"
                    )
                    & (
                        feature_summary[
                            "feature_id"
                        ]
                        == fid
                    )
                ][col]

                if len(inv) == 1 and len(resp) == 1:
                    row[
                        f"{metric}_invariant_minus_responsive"
                    ] = float(
                        inv.iloc[0]
                        - resp.iloc[0]
                    )

            contrast_rows.append(row)

    contrast_df = pd.DataFrame(
        contrast_rows
    )

    contrast_df.to_csv(
        args.out_dir
        / "feature_label_contrast.csv",
        index=False,
    )

    # Source shift: CC-Manual minus CC-LLM for each label/feature/metric.
    shift_rows = []

    for label in ["invariant", "responsive"]:
        for fid in FEATURE_POOL:
            row = {
                "context_label": label,
                "feature_id": fid,
            }

            for metric in feature_metric_cols:
                col = (
                    f"{metric}_component_balanced_mean"
                )

                s = feature_summary[
                    (feature_summary["source"] == "cc_llm")
                    & (
                        feature_summary[
                            "context_label"
                        ]
                        == label
                    )
                    & (
                        feature_summary[
                            "feature_id"
                        ]
                        == fid
                    )
                ][col]

                m = feature_summary[
                    (feature_summary["source"] == "cc_manual")
                    & (
                        feature_summary[
                            "context_label"
                        ]
                        == label
                    )
                    & (
                        feature_summary[
                            "feature_id"
                        ]
                        == fid
                    )
                ][col]

                if len(s) == 1 and len(m) == 1:
                    sv = float(s.iloc[0])
                    mv = float(m.iloc[0])

                    row[
                        f"{metric}_cc_llm"
                    ] = sv
                    row[
                        f"{metric}_cc_manual"
                    ] = mv
                    row[
                        f"{metric}_manual_minus_llm"
                    ] = mv - sv
                    row[
                        f"{metric}_manual_over_llm"
                    ] = (
                        mv / sv
                        if abs(sv) > 1e-12
                        else None
                    )

            shift_rows.append(row)

    shift_df = pd.DataFrame(
        shift_rows
    )

    shift_df.to_csv(
        args.out_dir
        / "source_shift_summary.csv",
        index=False,
    )

    # Compact final-config shift summary for console / paper diagnostics.
    compact = []

    for label in ["invariant", "responsive"]:
        for source in ["cc_llm", "cc_manual"]:
            sub = config_summary[
                (config_summary["source"] == source)
                & (
                    config_summary[
                        "context_label"
                    ]
                    == label
                )
            ]

            if len(sub) != 1:
                continue

            compact.append(
                {
                    "source": source,
                    "context_label": label,
                    "boundary_any_feature_side_rate":
                        float(
                            sub[
                                "boundary_any_feature_side_rate_component_balanced_mean"
                            ].iloc[0]
                        ),
                    "scope_any_feature_step_rate":
                        float(
                            sub[
                                "scope_any_feature_step_rate_component_balanced_mean"
                            ].iloc[0]
                        ),
                    "scope_activation_mass_per_side":
                        float(
                            sub[
                                "scope_activation_mass_per_side_component_balanced_mean"
                            ].iloc[0]
                        ),
                    "scope_pair_mass_abs_delta":
                        float(
                            sub[
                                "scope_pair_mass_abs_delta_component_balanced_mean"
                            ].iloc[0]
                        ),
                }
            )

    compact_df = pd.DataFrame(compact)

    summary = {
        "status": (
            "post-hoc exploratory mechanism audit after the failed "
            "CC-Manual external transfer"
        ),
        "scientific_boundary": (
            "This analysis may explain representation / activation shift, "
            "but it must not be used to re-select the intervention or to "
            "re-label CC-Manual as an untouched held-out test."
        ),
        "model": "google/gemma-3-4b-it",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": TARGET_LAYER,
        "feature_pool": FEATURE_POOL,
        "frozen_final_features": FINAL_FEATURES,
        "measured_scope": (
            "prompt boundary + first eight decode calls under baseline greedy decoding"
        ),
        "cc_llm": {
            "pairs": int(len(search)),
            "questions": int(len(search_rows)),
            "components": int(
                search[
                    "question_component"
                ].nunique()
            ),
        },
        "cc_manual": {
            "pairs": int(len(manual)),
            "questions": int(len(manual_rows)),
            "components": int(
                manual[
                    "question_component"
                ].nunique()
            ),
        },
        "final_config_compact_summary":
            compact_df.to_dict(
                orient="records"
            ),
    }

    write_json(
        args.out_dir
        / "run_summary.json",
        summary,
    )

    print("\n" + "=" * 90)
    print("FINAL-CONFIG ACTIVATION SUMMARY")
    print("=" * 90)
    print(
        compact_df.to_string(
            index=False
        )
    )

    print(
        "\nUpload first:\n"
        "  final_config_group_summary.csv\n"
        "  feature_label_contrast.csv\n"
        "  source_shift_summary.csv\n"
        "  run_summary.json"
    )


if __name__ == "__main__":
    try:
        mp.set_start_method(
            "spawn",
            force=True,
        )
        main()
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
