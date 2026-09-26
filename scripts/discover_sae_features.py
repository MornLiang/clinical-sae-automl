#!/usr/bin/env python3
"""
Extract layer-17 Gemma Scope 2 candidate features from the EquityMedQA search set.

Frozen setup:
  model       : google/gemma-3-4b-it
  SAE release : gemma-scope-2-4b-it-res
  SAE id      : layer_17_width_65k_l0_medium
  hook        : Gemma text decoder layer 17 output (resid_post)
  representation used for screening:
      the LAST PROMPT TOKEN (the assistant-generation boundary) only

Why the last prompt token?
- It is the same semantic position for every question: immediately before generation.
- It avoids aligning demographic words with different tokenizations/positions.
- It is exactly the position we can later intervene on in a localized causal test.

Candidate screening:
For each counterfactual pair, compute
    raw_delta_j = |z1_j - z2_j|
and a residual-space contribution proxy
    contribution_delta_j = raw_delta_j * ||W_dec[j]||_2

For each feature, aggregate separately over:
    invariant pairs
    responsive pairs

Primary screening score:
    invariant_mean_contribution - responsive_mean_contribution

This DOES NOT label a feature as "bias".
It only prioritizes features whose counterfactual activation changes are more
pronounced on expert-labelled invariant pairs than on responsive pairs.

No generation and no intervention happen in this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sae_lens import SAE
from tqdm import tqdm

from model import load_model


# =============================================================================
# Frozen configuration
# =============================================================================

INPUT_FILE = Path("data/processed/equity_search.csv")
OUT_DIR = Path("results/sae_layer17_65k_candidates")
CACHE_DIR = OUT_DIR / "question_latents"

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_65k_l0_medium"

TARGET_LAYER = 17
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_SAE_WIDTH = 65536
EXPECTED_NUM_LAYERS = 34

MIN_INVARIANT_SUPPORT = 3
TOP_K_SAVE = 200
TOP_K_EXAMPLE_FEATURES = 50
EXAMPLES_PER_FEATURE = 6
RECON_CHECK_QUESTIONS = 8


# =============================================================================
# Frozen prompt: same v5 instruction family used for the baseline generation
# =============================================================================

def build_frozen_v5_prompt(question: str) -> str:
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


# =============================================================================
# Helpers
# =============================================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def get_text_layers(model):
    if not hasattr(model, "model"):
        raise RuntimeError("Expected model.model; local Gemma wrapper differs.")

    multimodal_base = model.model

    if not hasattr(multimodal_base, "language_model"):
        raise RuntimeError(
            "Expected model.model.language_model; do not guess a hook path."
        )

    text_model = multimodal_base.language_model

    if not hasattr(text_model, "layers"):
        raise RuntimeError(
            "Expected model.model.language_model.layers; do not guess a hook path."
        )

    return text_model.layers


def build_question_map(df: pd.DataFrame) -> dict[str, str]:
    qmap: dict[str, str] = {}

    for _, row in df.iterrows():
        for side in (1, 2):
            qid = str(row[f"question_{side}_id"])
            text = str(row[f"question_{side}_text"])

            if qid in qmap and qmap[qid] != text:
                raise RuntimeError(
                    f"Question ID {qid} maps to multiple different texts."
                )

            qmap[qid] = text

    return qmap


# =============================================================================
# Main
# =============================================================================

def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {
        "pair_id",
        "question_1_id",
        "question_2_id",
        "question_1_text",
        "question_2_text",
        "context_label",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    labels = set(df["context_label"].astype(str))
    if not labels.issubset({"invariant", "responsive"}):
        raise RuntimeError(f"Unexpected context labels: {sorted(labels)}")

    n_inv = int((df["context_label"] == "invariant").sum())
    n_resp = int((df["context_label"] == "responsive").sum())

    if n_inv == 0 or n_resp == 0:
        raise RuntimeError("Both invariant and responsive pairs are required.")

    qmap = build_question_map(df)

    print("=" * 80)
    print("SEARCH DATA")
    print("=" * 80)
    print("pairs:", len(df))
    print("invariant:", n_inv)
    print("responsive:", n_resp)
    print("unique questions:", len(qmap))

    # -------------------------------------------------------------------------
    # Load model and verify the same structure already validated
    # -------------------------------------------------------------------------

    tokenizer, model = load_model()
    model.eval()

    layers = get_text_layers(model)

    if len(layers) != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LAYERS} layers, got {len(layers)}."
        )

    hidden_size = int(model.config.text_config.hidden_size)
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"Expected hidden size {EXPECTED_HIDDEN_SIZE}, got {hidden_size}."
        )

    target_module = layers[TARGET_LAYER]
    target_device = next(target_module.parameters()).device

    if target_device.type != "cuda":
        raise RuntimeError(
            f"Layer {TARGET_LAYER} is on {target_device}; GPU placement required."
        )

    print("\n" + "=" * 80)
    print("MODEL / SAE")
    print("=" * 80)
    print("target layer:", TARGET_LAYER)
    print("target device:", target_device)
    print("SAE release:", SAE_RELEASE)
    print("SAE id:", SAE_ID)

    # Official registered SAELens alias: no direct-repo warning and it selects
    # the Gemma-3 checkpoint converter automatically.
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=str(target_device),
        dtype="float32",
    )
    sae.eval()

    if int(sae.cfg.d_in) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"SAE d_in={sae.cfg.d_in}, expected {EXPECTED_HIDDEN_SIZE}."
        )

    if int(sae.cfg.d_sae) != EXPECTED_SAE_WIDTH:
        raise RuntimeError(
            f"SAE d_sae={sae.cfg.d_sae}, expected {EXPECTED_SAE_WIDTH}."
        )

    if tuple(sae.W_enc.shape) != (
        EXPECTED_HIDDEN_SIZE,
        EXPECTED_SAE_WIDTH,
    ):
        raise RuntimeError(
            f"Unexpected W_enc shape {tuple(sae.W_enc.shape)}."
        )

    if tuple(sae.W_dec.shape) != (
        EXPECTED_SAE_WIDTH,
        EXPECTED_HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected W_dec shape {tuple(sae.W_dec.shape)}."
        )

    decoder_norm = torch.linalg.vector_norm(
        sae.W_dec.detach().float(),
        dim=1,
    ).cpu()

    print(
        "decoder norm min/median/max:",
        f"{decoder_norm.min().item():.6f} / "
        f"{decoder_norm.median().item():.6f} / "
        f"{decoder_norm.max().item():.6f}",
    )

    sae_param = next(sae.parameters())
    input_device = model.get_input_embeddings().weight.device

    # -------------------------------------------------------------------------
    # Extract/cached last-prompt-token SAE activations for every unique question
    # -------------------------------------------------------------------------

    reconstruction_stats = []
    l0_values = []

    def extract_one(qid: str, question: str) -> torch.Tensor:
        cache_path = CACHE_DIR / f"{qid}.pt"
        text_hash = sha256_text(question)

        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu")
            if payload.get("text_sha256") != text_hash:
                raise RuntimeError(
                    f"Cached text hash mismatch for {qid}; do not reuse stale cache."
                )

            z = payload["z"].float()
            if tuple(z.shape) != (EXPECTED_SAE_WIDTH,):
                raise RuntimeError(
                    f"Bad cached latent shape for {qid}: {tuple(z.shape)}"
                )

            l0_values.append(int(payload["l0"]))
            return z

        prompt = build_frozen_v5_prompt(question)

        messages = [
            {
                "role": "user",
                "content": prompt,
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
            k: v.to(input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            if not torch.is_tensor(hidden):
                raise RuntimeError(
                    f"Unexpected layer output type: {type(hidden)}"
                )

            # Fixed screening representation:
            # last prompt token / assistant generation boundary.
            captured["h_last"] = hidden[:, -1, :].detach()

        handle = target_module.register_forward_hook(hook)

        try:
            with torch.inference_mode():
                _ = model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()

        if "h_last" not in captured:
            raise RuntimeError(f"Layer hook did not fire for {qid}.")

        h = captured["h_last"]

        if tuple(h.shape) != (1, EXPECTED_HIDDEN_SIZE):
            raise RuntimeError(
                f"Unexpected last-token activation shape for {qid}: {tuple(h.shape)}"
            )

        if not torch.isfinite(h).all():
            raise RuntimeError(f"NaN/Inf activation for {qid}.")

        h_sae = h.to(
            device=sae_param.device,
            dtype=sae_param.dtype,
        )

        with torch.inference_mode():
            z_gpu = sae.encode(h_sae)

        if tuple(z_gpu.shape) != (1, EXPECTED_SAE_WIDTH):
            raise RuntimeError(
                f"Unexpected SAE latent shape for {qid}: {tuple(z_gpu.shape)}"
            )

        if not torch.isfinite(z_gpu).all():
            raise RuntimeError(f"NaN/Inf SAE latent for {qid}.")

        l0 = int((z_gpu != 0).sum().item())
        l0_values.append(l0)

        # A small multi-question reconstruction check is folded into this run,
        # so no separate extra job is needed.
        if len(reconstruction_stats) < RECON_CHECK_QUESTIONS:
            with torch.inference_mode():
                h_hat = sae.decode(z_gpu)

            cos = F.cosine_similarity(
                h_sae.float(),
                h_hat.float(),
                dim=-1,
            ).item()

            rel_l2 = (
                torch.linalg.vector_norm(
                    h_sae.float() - h_hat.float(),
                    dim=-1,
                )
                / torch.linalg.vector_norm(
                    h_sae.float(),
                    dim=-1,
                ).clamp_min(1e-12)
            ).item()

            reconstruction_stats.append(
                {
                    "question_id": qid,
                    "l0": l0,
                    "cosine": float(cos),
                    "relative_l2": float(rel_l2),
                }
            )

        z_cpu = z_gpu.squeeze(0).detach().cpu().float()

        payload = {
            "question_id": qid,
            "text_sha256": text_hash,
            "z": z_cpu,
            "l0": l0,
        }
        torch.save(payload, cache_path)

        return z_cpu

    latents: dict[str, torch.Tensor] = {}

    for qid, question in tqdm(
        sorted(qmap.items()),
        desc="Unique questions",
    ):
        latents[qid] = extract_one(qid, question)

    # -------------------------------------------------------------------------
    # Aggregate pairwise feature differences
    # -------------------------------------------------------------------------

    width = EXPECTED_SAE_WIDTH
    dtype = torch.float64

    inv_sum = torch.zeros(width, dtype=dtype)
    inv_sumsq = torch.zeros(width, dtype=dtype)
    inv_support = torch.zeros(width, dtype=torch.int64)
    inv_raw_sum = torch.zeros(width, dtype=dtype)

    resp_sum = torch.zeros(width, dtype=dtype)
    resp_sumsq = torch.zeros(width, dtype=dtype)
    resp_support = torch.zeros(width, dtype=torch.int64)
    resp_raw_sum = torch.zeros(width, dtype=dtype)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Pair differences",
    ):
        q1 = str(row["question_1_id"])
        q2 = str(row["question_2_id"])

        z1 = latents[q1]
        z2 = latents[q2]

        raw_delta = torch.abs(z1 - z2)
        contribution = raw_delta * decoder_norm

        c64 = contribution.to(dtype)
        r64 = raw_delta.to(dtype)
        support = raw_delta > 0

        if row["context_label"] == "invariant":
            inv_sum += c64
            inv_sumsq += c64 * c64
            inv_support += support.to(torch.int64)
            inv_raw_sum += r64
        else:
            resp_sum += c64
            resp_sumsq += c64 * c64
            resp_support += support.to(torch.int64)
            resp_raw_sum += r64

    inv_mean = inv_sum / n_inv
    resp_mean = resp_sum / n_resp

    inv_raw_mean = inv_raw_sum / n_inv
    resp_raw_mean = resp_raw_sum / n_resp

    contrast = inv_mean - resp_mean

    inv_var = (
        (inv_sumsq / n_inv) - inv_mean.square()
    ).clamp_min(0.0)

    resp_var = (
        (resp_sumsq / n_resp) - resp_mean.square()
    ).clamp_min(0.0)

    pooled_sd = torch.sqrt(
        (inv_var + resp_var) / 2.0
    )

    standardized_contrast = contrast / pooled_sd.clamp_min(1e-12)

    ratio = (
        (inv_mean + 1e-12)
        / (resp_mean + 1e-12)
    )

    eligible = (
        (inv_support >= MIN_INVARIANT_SUPPORT)
        & (contrast > 0)
        & torch.isfinite(contrast)
    )

    eligible_ids = torch.where(eligible)[0]

    if len(eligible_ids) == 0:
        raise RuntimeError(
            "No eligible candidate features. "
            "Do not silently relax the screening rule."
        )

    ranked_ids = eligible_ids[
        torch.argsort(
            contrast[eligible_ids],
            descending=True,
        )
    ]

    ranked_ids = ranked_ids[:TOP_K_SAVE]

    summary_rows = []

    for rank, fid_t in enumerate(ranked_ids, start=1):
        fid = int(fid_t.item())

        summary_rows.append(
            {
                "rank": rank,
                "feature_id": fid,
                "decoder_norm": float(decoder_norm[fid].item()),
                "invariant_mean_contribution_delta": float(inv_mean[fid].item()),
                "responsive_mean_contribution_delta": float(resp_mean[fid].item()),
                "contrast_score": float(contrast[fid].item()),
                "contrast_ratio": float(ratio[fid].item()),
                "standardized_contrast": float(
                    standardized_contrast[fid].item()
                ),
                "invariant_support_pairs": int(inv_support[fid].item()),
                "responsive_support_pairs": int(resp_support[fid].item()),
                "invariant_mean_raw_latent_delta": float(
                    inv_raw_mean[fid].item()
                ),
                "responsive_mean_raw_latent_delta": float(
                    resp_raw_mean[fid].item()
                ),
            }
        )

    candidates_df = pd.DataFrame(summary_rows)

    candidates_path = OUT_DIR / "candidate_features.csv"
    candidates_df.to_csv(
        candidates_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save supporting pair examples for the highest-ranked candidate features
    # -------------------------------------------------------------------------

    example_feature_ids = [
        int(x)
        for x in ranked_ids[:TOP_K_EXAMPLE_FEATURES].tolist()
    ]

    example_rows = []

    for fid in example_feature_ids:
        per_pair = []

        for _, row in df.iterrows():
            q1 = str(row["question_1_id"])
            q2 = str(row["question_2_id"])

            raw_delta = float(
                torch.abs(
                    latents[q1][fid]
                    - latents[q2][fid]
                ).item()
            )

            contribution_delta = (
                raw_delta
                * float(decoder_norm[fid].item())
            )

            per_pair.append(
                {
                    "feature_id": fid,
                    "pair_id": str(row["pair_id"]),
                    "context_label": str(row["context_label"]),
                    "question_1_id": q1,
                    "question_2_id": q2,
                    "question_1_text": str(row["question_1_text"]),
                    "question_2_text": str(row["question_2_text"]),
                    "raw_latent_delta": raw_delta,
                    "contribution_delta": contribution_delta,
                }
            )

        per_pair.sort(
            key=lambda r: r["contribution_delta"],
            reverse=True,
        )

        example_rows.extend(
            per_pair[:EXAMPLES_PER_FEATURE]
        )

    examples_path = OUT_DIR / "candidate_support_examples.csv"
    pd.DataFrame(example_rows).to_csv(
        examples_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    l0_tensor = torch.tensor(
        l0_values,
        dtype=torch.float32,
    )

    run_summary = {
        "model": "google/gemma-3-4b-it",
        "target_layer": TARGET_LAYER,
        "hook_semantics": "resid_post / decoder layer output",
        "screening_position": "last prompt token / assistant generation boundary",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "sae_width": EXPECTED_SAE_WIDTH,
        "pairs": len(df),
        "invariant_pairs": n_inv,
        "responsive_pairs": n_resp,
        "unique_questions": len(qmap),
        "minimum_invariant_support": MIN_INVARIANT_SUPPORT,
        "eligible_feature_count": int(eligible.sum().item()),
        "saved_candidate_count": len(candidates_df),
        "question_l0": {
            "mean": float(l0_tensor.mean().item()),
            "median": float(l0_tensor.median().item()),
            "min": float(l0_tensor.min().item()),
            "max": float(l0_tensor.max().item()),
        },
        "decoder_norm": {
            "min": float(decoder_norm.min().item()),
            "median": float(decoder_norm.median().item()),
            "max": float(decoder_norm.max().item()),
        },
        "reconstruction_checks": reconstruction_stats,
        "important_note": (
            "Candidates are screening candidates only, not identified bias features. "
            "Causal intervention must be tested next."
        ),
    }

    summary_path = OUT_DIR / "run_summary.json"
    write_json(summary_path, run_summary)

    print("\n" + "=" * 80)
    print("CANDIDATE EXTRACTION FINISHED")
    print("=" * 80)
    print("eligible features:", run_summary["eligible_feature_count"])
    print(
        "L0 mean / median / min / max:",
        f'{run_summary["question_l0"]["mean"]:.2f} / '
        f'{run_summary["question_l0"]["median"]:.2f} / '
        f'{run_summary["question_l0"]["min"]:.0f} / '
        f'{run_summary["question_l0"]["max"]:.0f}',
    )

    print("\nTop 20 candidates:")
    print(
        candidates_df.head(20)[
            [
                "rank",
                "feature_id",
                "contrast_score",
                "invariant_mean_contribution_delta",
                "responsive_mean_contribution_delta",
                "invariant_support_pairs",
                "responsive_support_pairs",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(candidates_path)
    print(examples_path)
    print(summary_path)
    print(CACHE_DIR)
    print(
        "\nNo intervention was performed. "
        "The next step is a small causal single-feature suppression test."
    )


if __name__ == "__main__":
    main()
