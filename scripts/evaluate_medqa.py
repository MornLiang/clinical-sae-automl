#!/usr/bin/env python3
"""
4-GPU MedQA validation utility gate for the FINAL full-answer AutoML finalists.

This is NOT another search.

Frozen finalists from full 121-pair CC-LLM re-evaluation:
1) 11114+8767+1988, alpha=0.50, boundary_plus_first8
2) 11114+1988,      alpha=0.75, boundary_plus_first8
3) 1669+23293,      alpha=1.00, boundary_only
4) 1988+23293,      alpha=0.50, boundary_only

Protocol:
- MedQA validation split only.
- Exact same prompt/parser as the previous full MedQA validation.
- Reuse the already-computed deterministic baseline from:
    results/pareto_full_validation/medqa_validation_results.csv
- One finalist per GPU worker.
- No CC-Manual.
- No MedQA test.
- No search/tuning.

Utility gate:
    accuracy drop vs baseline >= -0.02
i.e. no more than 2 percentage points absolute degradation.

Outputs:
    results/finalists_medqa_validation/
        medqa_finalist_results.csv
        medqa_finalist_summary.csv
        run_summary.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset


MEDQA_DATASET = "GBaker/MedQA-USMLE-4-options-hf"
MEDQA_SPLIT = "validation"

PREVIOUS_MEDQA_RESULTS = Path(
    "results/pareto_full_validation/medqa_validation_results.csv"
)
DEFAULT_OUT = Path("results/finalists_medqa_validation")

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_65k_l0_medium"

TARGET_LAYER = 17
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_NUM_LAYERS = 34
EXPECTED_SAE_WIDTH = 65536

MAX_NEW_TOKENS = 8
UTILITY_DROP_LIMIT = -0.02

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260923

FINALISTS = [
    {
        "name": "best_full_11114_8767_1988_a050_first8",
        "features": (11114, 8767, 1988),
        "alpha": 0.50,
        "scope": "boundary_plus_first8",
        "max_decode_steps": 8,
    },
    {
        "name": "simple_strong_11114_1988_a075_first8",
        "features": (11114, 1988),
        "alpha": 0.75,
        "scope": "boundary_plus_first8",
        "max_decode_steps": 8,
    },
    {
        "name": "balanced_1669_23293_a100_boundary",
        "features": (1669, 23293),
        "alpha": 1.00,
        "scope": "boundary_only",
        "max_decode_steps": 0,
    },
    {
        "name": "conservative_1988_23293_a050_boundary",
        "features": (1988, 23293),
        "alpha": 0.50,
        "scope": "boundary_only",
        "max_decode_steps": 0,
    },
]


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_medqa_prompt(row: dict[str, Any]) -> str:
    # EXACT prompt from validate_pareto_configs.py
    return f"""
You are a knowledgeable medical doctor.

Answer the following multiple-choice medical question.
Choose the single best answer and return only one letter: A, B, C, or D.

Question:
{row["sent1"]}
{row["sent2"]}

A. {row["ending0"]}
B. {row["ending1"]}
C. {row["ending2"]}
D. {row["ending3"]}

Answer:
""".strip()


def parse_choice(text: str) -> str | None:
    t = str(text).strip().upper()

    m = re.match(r"^\s*([ABCD])\s*[.\)]?\s*$", t)
    if m:
        return m.group(1)

    m = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", t)
    if m:
        return m.group(1)

    return None


def label_to_letter(label: int) -> str:
    return "ABCD"[int(label)]


def paired_bootstrap_accuracy_delta(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    seed: int,
):
    n = len(baseline_correct)

    if n != len(candidate_correct):
        raise ValueError("Paired result length mismatch.")

    observed = float(
        candidate_correct.mean()
        - baseline_correct.mean()
    )

    rng = np.random.default_rng(seed)
    boot = np.empty(
        BOOTSTRAP_REPS,
        dtype=np.float64,
    )

    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot[i] = (
            candidate_correct[idx].mean()
            - baseline_correct[idx].mean()
        )

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return observed, float(lo), float(hi)


def gpu_worker(
    physical_gpu: int,
    cfg: dict,
    rows: list[dict],
    result_queue,
):
    # Must happen BEFORE torch / transformers / sae imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    try:
        import torch
        from sae_lens import SAE
        from model import load_model

        tokenizer, model = load_model()
        model.eval()

        layers = model.model.language_model.layers

        if len(layers) != EXPECTED_NUM_LAYERS:
            raise RuntimeError("Layer count mismatch.")

        if int(
            model.config.text_config.hidden_size
        ) != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError("Hidden size mismatch.")

        target_module = layers[TARGET_LAYER]
        target_device = next(
            target_module.parameters()
        ).device

        input_device = (
            model.get_input_embeddings()
            .weight.device
        )

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

        features = tuple(
            int(x)
            for x in cfg["features"]
        )
        alpha = float(cfg["alpha"])
        max_decode_steps = int(
            cfg["max_decode_steps"]
        )

        def make_hook():
            state = {
                "prefill_seen": False,
                "decode_calls_seen": 0,
            }

            def hook(_module, _inputs, output):
                hidden = (
                    output[0]
                    if isinstance(output, tuple)
                    else output
                )

                seq_len = int(
                    hidden.shape[1]
                )

                is_prefill = (
                    not state["prefill_seen"]
                    and seq_len > 1
                )

                if is_prefill:
                    state["prefill_seen"] = True
                    should_apply = True

                elif not state["prefill_seen"]:
                    state["prefill_seen"] = True
                    should_apply = True

                else:
                    decode_idx = (
                        state["decode_calls_seen"]
                    )
                    state[
                        "decode_calls_seen"
                    ] += 1

                    should_apply = (
                        decode_idx
                        < max_decode_steps
                    )

                if not should_apply:
                    return output

                h_last = hidden[:, -1, :]
                sp = next(
                    sae.parameters()
                )

                with torch.inference_mode():
                    z = sae.encode(
                        h_last.to(
                            device=sp.device,
                            dtype=sp.dtype,
                        )
                    )

                    ids = torch.tensor(
                        features,
                        device=z.device,
                        dtype=torch.long,
                    )

                    selected = z.index_select(
                        1,
                        ids,
                    )

                    z_mod = z.clone()

                    z_mod.index_copy_(
                        1,
                        ids,
                        selected
                        * (1.0 - alpha),
                    )

                    delta = (
                        sae.decode(z_mod)
                        - sae.decode(z)
                    )

                hidden_mod = hidden.clone()

                hidden_mod[:, -1, :] += (
                    delta.to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                )

                if isinstance(
                    output,
                    tuple,
                ):
                    return (
                        hidden_mod,
                    ) + output[1:]

                return hidden_mod

            return hook

        outputs = []
        start = time.time()

        for i, row in enumerate(rows):
            prompt = build_medqa_prompt(
                row
            )

            tokenized = (
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            )

            inputs = {
                k: v.to(input_device)
                for k, v in tokenized.items()
            }

            input_len = int(
                inputs[
                    "input_ids"
                ].shape[1]
            )

            handle = (
                target_module
                .register_forward_hook(
                    make_hook()
                )
            )

            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=(
                            MAX_NEW_TOKENS
                        ),
                        do_sample=False,
                        use_cache=True,
                    )
            finally:
                handle.remove()

            raw = tokenizer.decode(
                generated[
                    0,
                    input_len:,
                ],
                skip_special_tokens=True,
            ).strip()

            pred = parse_choice(raw)

            gold = label_to_letter(
                int(row["label"])
            )

            outputs.append(
                {
                    "config":
                        cfg["name"],
                    "feature_subset":
                        "+".join(
                            str(x)
                            for x in features
                        ),
                    "alpha":
                        alpha,
                    "scope":
                        cfg["scope"],
                    "id":
                        str(row["id"]),
                    "gold":
                        gold,
                    "pred":
                        pred
                        if pred is not None
                        else "",
                    "correct":
                        bool(
                            pred == gold
                        ),
                    "valid_prediction":
                        bool(
                            pred is not None
                        ),
                    "raw_generation":
                        raw,
                }
            )

            if (
                (i + 1) % 200 == 0
                or i + 1 == len(rows)
            ):
                print(
                    f"[GPU {physical_gpu}] "
                    f"{cfg['name']}: "
                    f"{i + 1}/{len(rows)}",
                    flush=True,
                )

        result_queue.put(
            {
                "type": "result",
                "gpu": physical_gpu,
                "config":
                    cfg["name"],
                "elapsed_sec":
                    time.time()
                    - start,
                "rows":
                    outputs,
            }
        )

    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "gpu": physical_gpu,
                "config":
                    cfg.get(
                        "name",
                        "unknown",
                    ),
                "error":
                    repr(exc),
            }
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpus",
        default="7,1,2,0",
        help=(
            "Exactly four physical GPUs "
            "for the four frozen finalists."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    gpu_ids = [
        int(x.strip())
        for x in args.gpus.split(",")
        if x.strip()
    ]

    if len(gpu_ids) != len(
        FINALISTS
    ):
        raise RuntimeError(
            f"Need exactly {len(FINALISTS)} GPUs; "
            f"got {gpu_ids}"
        )

    if len(
        set(gpu_ids)
    ) != len(
        gpu_ids
    ):
        raise RuntimeError(
            "Duplicate GPU IDs."
        )

    if not PREVIOUS_MEDQA_RESULTS.is_file():
        raise FileNotFoundError(
            "Missing previous deterministic "
            f"MedQA validation results: "
            f"{PREVIOUS_MEDQA_RESULTS}"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load the same validation split.
    dataset = load_dataset(
        MEDQA_DATASET,
        split=MEDQA_SPLIT,
    )

    medqa = (
        dataset
        .to_pandas()
        .reset_index(drop=True)
    )

    rows = medqa.to_dict(
        orient="records"
    )

    # Reuse the exact baseline from the previous validation.
    previous = pd.read_csv(
        PREVIOUS_MEDQA_RESULTS
    )

    baseline = previous[
        previous["config"]
        == "baseline"
    ].copy()

    if len(baseline) != len(
        medqa
    ):
        raise RuntimeError(
            "Previous baseline length does not "
            "match current MedQA validation split."
        )

    baseline["id"] = (
        baseline["id"]
        .astype(str)
    )

    current_ids = set(
        medqa["id"]
        .astype(str)
        .tolist()
    )

    if set(
        baseline["id"]
    ) != current_ids:
        raise RuntimeError(
            "Previous baseline IDs do not match "
            "current MedQA validation split."
        )

    ctx = mp.get_context(
        "spawn"
    )

    result_queue = (
        ctx.Queue()
    )

    workers = []

    print(
        "Starting finalist MedQA validation:",
        flush=True,
    )

    for gpu, cfg in zip(
        gpu_ids,
        FINALISTS,
    ):
        print(
            f"  GPU {gpu}: {cfg['name']}",
            flush=True,
        )

        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu,
                cfg,
                rows,
                result_queue,
            ),
            daemon=False,
        )

        p.start()
        workers.append(p)

    results = []

    completed = 0

    while completed < len(
        FINALISTS
    ):
        msg = result_queue.get()

        if msg["type"] == "error":
            raise RuntimeError(
                f"Worker failed: {msg}"
            )

        results.extend(
            msg["rows"]
        )

        completed += 1

        print(
            f"Finished {msg['config']} "
            f"on GPU {msg['gpu']} in "
            f"{msg['elapsed_sec']/60:.1f} min "
            f"({completed}/{len(FINALISTS)})",
            flush=True,
        )

    for p in workers:
        p.join(
            timeout=30
        )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        args.out_dir
        / "medqa_finalist_results.csv",
        index=False,
    )

    # Paired baseline comparison by ID.
    base_by_id = (
        baseline
        .set_index("id")
        .sort_index()
    )

    baseline_correct = (
        base_by_id[
            "correct"
        ]
        .astype(bool)
        .to_numpy(
            dtype=np.float64
        )
    )

    baseline_accuracy = float(
        baseline_correct.mean()
    )

    summary_rows = []

    for idx, cfg in enumerate(
        FINALISTS
    ):
        sub = result_df[
            result_df[
                "config"
            ]
            == cfg["name"]
        ].copy()

        sub["id"] = (
            sub["id"]
            .astype(str)
        )

        sub = (
            sub.set_index("id")
            .loc[
                base_by_id.index
            ]
        )

        correct = (
            sub["correct"]
            .astype(bool)
            .to_numpy(
                dtype=np.float64
            )
        )

        delta, lo, hi = (
            paired_bootstrap_accuracy_delta(
                baseline_correct,
                correct,
                seed=(
                    BOOTSTRAP_SEED
                    + idx
                ),
            )
        )

        summary_rows.append(
            {
                "config":
                    cfg["name"],
                "feature_subset":
                    "+".join(
                        str(x)
                        for x
                        in cfg[
                            "features"
                        ]
                    ),
                "alpha":
                    cfg["alpha"],
                "scope":
                    cfg["scope"],
                "n":
                    len(sub),
                "baseline_accuracy":
                    baseline_accuracy,
                "accuracy":
                    float(
                        correct.mean()
                    ),
                "correct_count":
                    int(
                        correct.sum()
                    ),
                "valid_rate":
                    float(
                        sub[
                            "valid_prediction"
                        ]
                        .astype(bool)
                        .mean()
                    ),
                "accuracy_delta_vs_baseline":
                    delta,
                "bootstrap_95ci_low":
                    lo,
                "bootstrap_95ci_high":
                    hi,
                "passes_utility_gate":
                    bool(
                        delta
                        >= UTILITY_DROP_LIMIT
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        args.out_dir
        / "medqa_finalist_summary.csv",
        index=False,
    )

    write_json(
        args.out_dir
        / "run_summary.json",
        {
            "medqa_dataset":
                MEDQA_DATASET,
            "split":
                MEDQA_SPLIT,
            "n":
                len(medqa),
            "baseline_accuracy":
                baseline_accuracy,
            "utility_gate":
                (
                    "accuracy delta vs baseline "
                    ">= -0.02"
                ),
            "finalists":
                [
                    {
                        "name":
                            x["name"],
                        "features":
                            list(
                                x[
                                    "features"
                                ]
                            ),
                        "alpha":
                            x["alpha"],
                        "scope":
                            x["scope"],
                    }
                    for x in FINALISTS
                ],
            "important_note":
                (
                    "Validation only. "
                    "No CC-Manual or MedQA test "
                    "was touched."
                ),
        },
    )

    print("\nMEDQA FINALIST SUMMARY")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nUpload:\n"
        "  medqa_finalist_summary.csv\n"
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
