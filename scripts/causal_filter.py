#!/usr/bin/env python3
"""
Cross-component causal screen for ALL component-aware broad SAE candidates.

No answer generation and no AutoML happen here.

Inputs:
  data/processed/equity_search.csv
  results/sae_layer17_65k_candidates/question_latents/*.pt
  results/sae_layer17_65k_component_candidates/broad_candidates.csv

Frozen model/SAE:
  google/gemma-3-4b-it
  Gemma Scope 2 resid_post, layer 17, width 65k, medium L0

Protocol:
1. For each broad candidate feature, find its TWO invariant question-components
   with the largest mean cached |z1-z2|.
2. Within each selected component, choose the invariant pair with the largest
   cached |z1-z2| for that feature.
3. For BOTH questions in each selected pair, measure next-token distribution
   changes under alpha = 0, 0.5, 1.0 using:
       z'_j = (1-alpha) z_j
       h'   = h + decode(z') - decode(z)
   at the last prompt token only.
4. alpha=0 must be an exact no-op at the logit level.
5. Summarize each selected component by the larger alpha=1 KL effect of its two
   questions (the feature may be active on only one side).
6. A feature is "cross-component causal eligible" iff BOTH selected invariant
   components have component max KL(alpha=1) > 1e-4.

Why 1e-4?
The prior causal feasibility run measured inactive-side KL around 1e-8 with
exactly zero logit differences. 1e-4 is used only as a numerical separation
threshold, NOT as a fairness or clinical threshold.

Final layer-17 AutoML pool rule (predeclared here):
Take the first 8 features by the EXISTING component-aware conservative
association ranking among features that pass cross-component causal eligibility.
If fewer than 8 pass, keep all that pass; do not relax the threshold.

This script does NOT identify "bias features" and does NOT evaluate fairness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sae_lens import SAE
from tqdm import tqdm

from model import load_model


SEARCH_FILE = Path("data/processed/equity_search.csv")
LATENT_DIR = Path("results/sae_layer17_65k_candidates/question_latents")
BROAD_FILE = Path(
    "results/sae_layer17_65k_component_candidates/broad_candidates.csv"
)
OUT_DIR = Path("results/sae_layer17_65k_causal_screen")

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_17_width_65k_l0_medium"

TARGET_LAYER = 17
EXPECTED_HIDDEN_SIZE = 2560
EXPECTED_SAE_WIDTH = 65536
EXPECTED_NUM_LAYERS = 34

N_COMPONENTS_PER_FEATURE = 2
ALPHAS = [0.0, 0.5, 1.0]
MEANINGFUL_KL_THRESHOLD = 1e-4
FINAL_POOL_SIZE = 8


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


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_components(df: pd.DataFrame) -> dict[str, str]:
    uf = UnionFind()

    for _, row in df.iterrows():
        uf.union(
            str(row["question_1_id"]),
            str(row["question_2_id"]),
        )

    groups: dict[str, list[str]] = defaultdict(list)

    for qid in list(uf.parent):
        groups[uf.find(qid)].append(qid)

    mapping: dict[str, str] = {}

    for members in groups.values():
        cid = min(members)
        for qid in members:
            mapping[qid] = cid

    return mapping


def load_latent(qid: str) -> torch.Tensor:
    path = LATENT_DIR / f"{qid}.pt"

    if not path.is_file():
        raise FileNotFoundError(f"Missing cached latent: {path}")

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    z = payload["z"].float()

    if tuple(z.shape) != (EXPECTED_SAE_WIDTH,):
        raise RuntimeError(
            f"Unexpected latent shape for {qid}: {tuple(z.shape)}"
        )

    if not torch.isfinite(z).all():
        raise RuntimeError(f"NaN/Inf in cached latent {qid}")

    return z


def get_text_layers(model):
    if not hasattr(model, "model"):
        raise RuntimeError("Expected model.model.")
    if not hasattr(model.model, "language_model"):
        raise RuntimeError("Expected model.model.language_model.")

    text_model = model.model.language_model

    if not hasattr(text_model, "layers"):
        raise RuntimeError("Expected model.model.language_model.layers.")

    return text_model.layers


def tokenize_prompt(tokenizer, model, question: str):
    messages = [
        {
            "role": "user",
            "content": build_frozen_v5_prompt(question),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_device = model.get_input_embeddings().weight.device

    return {
        k: v.to(input_device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }


def make_hook(
    sae: SAE,
    feature_id: int,
    alpha: float,
    record: dict,
):
    state = {"applied": False}

    def hook(_module, _inputs, output):
        if state["applied"]:
            return output

        hidden = output[0] if isinstance(output, tuple) else output

        if not torch.is_tensor(hidden):
            raise RuntimeError(
                f"Unexpected layer output type: {type(hidden)}"
            )

        if (
            hidden.ndim != 3
            or hidden.shape[-1] != EXPECTED_HIDDEN_SIZE
        ):
            raise RuntimeError(
                f"Unexpected layer output shape: {tuple(hidden.shape)}"
            )

        h_last = hidden[:, -1, :]

        sae_param = next(sae.parameters())

        h_sae = h_last.to(
            device=sae_param.device,
            dtype=sae_param.dtype,
        )

        with torch.inference_mode():
            z = sae.encode(h_sae)
            z_mod = z.clone()

            activation = z[:, feature_id].clone()

            z_mod[:, feature_id] = (
                z_mod[:, feature_id]
                * (1.0 - alpha)
            )

            recon = sae.decode(z)
            recon_mod = sae.decode(z_mod)
            delta = recon_mod - recon

        hidden_mod = hidden.clone()

        hidden_mod[:, -1, :] = (
            hidden_mod[:, -1, :]
            + delta.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
        )

        record.update(
            {
                "feature_activation": float(
                    activation.squeeze().item()
                ),
                "delta_l2": float(
                    torch.linalg.vector_norm(
                        delta.float(),
                        dim=-1,
                    ).squeeze().item()
                ),
            }
        )

        state["applied"] = True

        if isinstance(output, tuple):
            return (hidden_mod,) + output[1:]
        return hidden_mod

    return hook


def next_token_logits(
    tokenizer,
    model,
    target_module,
    sae,
    question: str,
    feature_id: int | None = None,
    alpha: float | None = None,
):
    inputs = tokenize_prompt(
        tokenizer,
        model,
        question,
    )

    record = {}
    handle = None

    if feature_id is not None:
        if alpha is None:
            raise ValueError("alpha required.")

        handle = target_module.register_forward_hook(
            make_hook(
                sae,
                feature_id,
                alpha,
                record,
            )
        )

    try:
        with torch.inference_mode():
            output = model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    logits = (
        output.logits[:, -1, :]
        .detach()
        .float()
        .cpu()
    )

    return logits, record


def logit_metrics(
    base_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> dict:
    base_logp = F.log_softmax(
        base_logits,
        dim=-1,
    )
    new_logp = F.log_softmax(
        new_logits,
        dim=-1,
    )
    base_p = base_logp.exp()

    kl = F.kl_div(
        new_logp,
        base_p,
        reduction="batchmean",
    ).item()

    diff = new_logits - base_logits

    return {
        "kl_base_to_intervention": float(kl),
        "max_abs_logit_diff": float(
            diff.abs().max().item()
        ),
        "mean_abs_logit_diff": float(
            diff.abs().mean().item()
        ),
        "logit_l2": float(
            torch.linalg.vector_norm(
                diff,
                dim=-1,
            ).mean().item()
        ),
        "top1_changed": bool(
            base_logits.argmax(dim=-1).item()
            != new_logits.argmax(dim=-1).item()
        ),
    }


def main():
    if not SEARCH_FILE.is_file():
        raise FileNotFoundError(f"Missing {SEARCH_FILE}")
    if not LATENT_DIR.is_dir():
        raise FileNotFoundError(f"Missing {LATENT_DIR}")
    if not BROAD_FILE.is_file():
        raise FileNotFoundError(f"Missing {BROAD_FILE}")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(SEARCH_FILE)
    broad = pd.read_csv(BROAD_FILE)

    if len(broad) == 0:
        raise RuntimeError("Broad candidate table is empty.")

    component_map = build_components(df)

    df = df.copy()
    df["question_component"] = [
        component_map[str(qid)]
        for qid in df["question_1_id"]
    ]

    invariant_df = df[
        df["context_label"] == "invariant"
    ].copy()

    # Load all required question latents once.
    qids = sorted(
        {
            *df["question_1_id"].astype(str).tolist(),
            *df["question_2_id"].astype(str).tolist(),
        }
    )

    latent_map = {
        qid: load_latent(qid)
        for qid in qids
    }

    # -------------------------------------------------------------------------
    # Pre-select two invariant components + one top pair per component
    # for every broad candidate using ONLY cached latents.
    # -------------------------------------------------------------------------

    selected_cases = []

    for _, frow in broad.iterrows():
        fid = int(frow["feature_id"])

        component_candidates = []

        for cid, group in invariant_df.groupby(
            "question_component"
        ):
            pair_rows = []

            for _, row in group.iterrows():
                q1 = str(row["question_1_id"])
                q2 = str(row["question_2_id"])

                delta = abs(
                    float(
                        latent_map[q1][fid].item()
                        - latent_map[q2][fid].item()
                    )
                )

                pair_rows.append(
                    (
                        delta,
                        row,
                    )
                )

            pair_rows.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            top_delta, top_row = pair_rows[0]

            component_mean_delta = sum(
                x[0]
                for x in pair_rows
            ) / len(pair_rows)

            if component_mean_delta > 0:
                component_candidates.append(
                    {
                        "component": cid,
                        "component_mean_delta": component_mean_delta,
                        "top_pair_delta": top_delta,
                        "row": top_row,
                    }
                )

        component_candidates.sort(
            key=lambda x: x["component_mean_delta"],
            reverse=True,
        )

        chosen = component_candidates[
            :N_COMPONENTS_PER_FEATURE
        ]

        if len(chosen) < N_COMPONENTS_PER_FEATURE:
            raise RuntimeError(
                f"Feature {fid} has fewer than "
                f"{N_COMPONENTS_PER_FEATURE} invariant components "
                f"despite being in the broad pool."
            )

        for component_rank, item in enumerate(
            chosen,
            start=1,
        ):
            row = item["row"]

            selected_cases.append(
                {
                    "feature_id": fid,
                    "association_rank": int(
                        frow["rank"]
                    ),
                    "association_score": float(
                        frow["conservative_score"]
                    ),
                    "question_component": item[
                        "component"
                    ],
                    "component_rank_for_feature": component_rank,
                    "component_mean_cached_delta": item[
                        "component_mean_delta"
                    ],
                    "pair_id": str(
                        row["pair_id"]
                    ),
                    "question_1_id": str(
                        row["question_1_id"]
                    ),
                    "question_2_id": str(
                        row["question_2_id"]
                    ),
                    "question_1_text": str(
                        row["question_1_text"]
                    ),
                    "question_2_text": str(
                        row["question_2_text"]
                    ),
                    "top_pair_cached_delta": item[
                        "top_pair_delta"
                    ],
                }
            )

    selected_cases_df = pd.DataFrame(
        selected_cases
    )

    selected_cases_df.to_csv(
        OUT_DIR / "selected_component_cases.csv",
        index=False,
    )

    print("=" * 80)
    print("CAUSAL SCREEN PLAN")
    print("=" * 80)
    print("broad features:", len(broad))
    print(
        "components per feature:",
        N_COMPONENTS_PER_FEATURE,
    )
    print(
        "selected feature-component cases:",
        len(selected_cases_df),
    )
    print(
        "question-feature evaluations:",
        len(selected_cases_df) * 2,
    )
    print(
        "meaningful KL threshold:",
        MEANINGFUL_KL_THRESHOLD,
    )

    # -------------------------------------------------------------------------
    # Load model + SAE once.
    # -------------------------------------------------------------------------

    tokenizer, model = load_model()
    model.eval()

    layers = get_text_layers(model)

    if len(layers) != EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_LAYERS} layers; "
            f"got {len(layers)}."
        )

    if int(
        model.config.text_config.hidden_size
    ) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("Hidden size mismatch.")

    target_module = layers[TARGET_LAYER]
    target_device = next(
        target_module.parameters()
    ).device

    if target_device.type != "cuda":
        raise RuntimeError(
            f"Target layer on {target_device}, CUDA required."
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

    # -------------------------------------------------------------------------
    # Causal logit screen.
    # -------------------------------------------------------------------------

    detail_rows = []

    iterator = tqdm(
        selected_cases,
        desc="Feature-component causal cases",
    )

    for case in iterator:
        fid = int(case["feature_id"])

        for side in (1, 2):
            qid = case[
                f"question_{side}_id"
            ]
            question = case[
                f"question_{side}_text"
            ]

            base_logits, _ = next_token_logits(
                tokenizer,
                model,
                target_module,
                sae,
                question,
            )

            alpha_results = {}

            for alpha in ALPHAS:
                int_logits, record = next_token_logits(
                    tokenizer,
                    model,
                    target_module,
                    sae,
                    question,
                    feature_id=fid,
                    alpha=alpha,
                )

                metrics = logit_metrics(
                    base_logits,
                    int_logits,
                )

                metrics.update(record)

                alpha_results[
                    str(alpha)
                ] = metrics

                if (
                    alpha == 0.0
                    and metrics[
                        "max_abs_logit_diff"
                    ] != 0.0
                ):
                    raise RuntimeError(
                        f"alpha=0 is not exact no-op: "
                        f"feature={fid}, qid={qid}, "
                        f"max_diff="
                        f'{metrics["max_abs_logit_diff"]}'
                    )

            a05 = alpha_results["0.5"]
            a1 = alpha_results["1.0"]

            detail_rows.append(
                {
                    **case,
                    "question_side": side,
                    "question_id": qid,
                    "question_text": question,
                    "feature_activation": a1.get(
                        "feature_activation"
                    ),
                    "delta_l2_alpha05": a05.get(
                        "delta_l2"
                    ),
                    "delta_l2_alpha1": a1.get(
                        "delta_l2"
                    ),
                    "kl_alpha05": a05[
                        "kl_base_to_intervention"
                    ],
                    "kl_alpha1": a1[
                        "kl_base_to_intervention"
                    ],
                    "max_logit_diff_alpha05": a05[
                        "max_abs_logit_diff"
                    ],
                    "max_logit_diff_alpha1": a1[
                        "max_abs_logit_diff"
                    ],
                    "top1_changed_alpha05": a05[
                        "top1_changed"
                    ],
                    "top1_changed_alpha1": a1[
                        "top1_changed"
                    ],
                    "dose_kl_non_decreasing": bool(
                        a1[
                            "kl_base_to_intervention"
                        ]
                        >= a05[
                            "kl_base_to_intervention"
                        ]
                    ),
                    "meaningful_alpha1_effect": bool(
                        a1[
                            "kl_base_to_intervention"
                        ]
                        > MEANINGFUL_KL_THRESHOLD
                    ),
                }
            )

    details_df = pd.DataFrame(
        detail_rows
    )

    details_path = (
        OUT_DIR
        / "question_level_causal_results.csv"
    )

    details_df.to_csv(
        details_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Component-level causal summaries.
    # -------------------------------------------------------------------------

    component_rows = []

    group_cols = [
        "feature_id",
        "association_rank",
        "association_score",
        "question_component",
        "component_rank_for_feature",
        "pair_id",
    ]

    for keys, group in details_df.groupby(
        group_cols,
        sort=False,
    ):
        (
            fid,
            association_rank,
            association_score,
            cid,
            component_rank,
            pair_id,
        ) = keys

        component_rows.append(
            {
                "feature_id": int(fid),
                "association_rank": int(
                    association_rank
                ),
                "association_score": float(
                    association_score
                ),
                "question_component": cid,
                "component_rank_for_feature": int(
                    component_rank
                ),
                "pair_id": pair_id,
                "component_max_kl_alpha05": float(
                    group["kl_alpha05"].max()
                ),
                "component_max_kl_alpha1": float(
                    group["kl_alpha1"].max()
                ),
                "component_max_logit_diff_alpha1": float(
                    group[
                        "max_logit_diff_alpha1"
                    ].max()
                ),
                "component_has_meaningful_alpha1_effect": bool(
                    group["kl_alpha1"].max()
                    > MEANINGFUL_KL_THRESHOLD
                ),
                "component_any_top1_change_alpha1": bool(
                    group[
                        "top1_changed_alpha1"
                    ].any()
                ),
                "component_dose_non_decreasing": bool(
                    group[
                        "component_rank_for_feature"
                    ].notna().all()
                    and (
                        group["kl_alpha1"]
                        >= group["kl_alpha05"]
                    ).all()
                ),
            }
        )

    component_df = pd.DataFrame(
        component_rows
    )

    component_path = (
        OUT_DIR
        / "component_level_causal_results.csv"
    )

    component_df.to_csv(
        component_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Feature-level cross-component causal ranking.
    # -------------------------------------------------------------------------

    feature_rows = []

    for fid, group in component_df.groupby(
        "feature_id",
        sort=False,
    ):
        group = group.sort_values(
            "component_rank_for_feature"
        )

        if len(group) != N_COMPONENTS_PER_FEATURE:
            raise RuntimeError(
                f"Feature {fid} component count mismatch."
            )

        min_kl = float(
            group[
                "component_max_kl_alpha1"
            ].min()
        )

        mean_kl = float(
            group[
                "component_max_kl_alpha1"
            ].mean()
        )

        all_meaningful = bool(
            group[
                "component_has_meaningful_alpha1_effect"
            ].all()
        )

        all_dose = bool(
            group[
                "component_dose_non_decreasing"
            ].all()
        )

        frow = broad[
            broad["feature_id"] == fid
        ].iloc[0]

        feature_rows.append(
            {
                "feature_id": int(fid),
                "association_rank": int(
                    frow["rank"]
                ),
                "association_score": float(
                    frow["conservative_score"]
                ),
                "cross_component_min_kl_alpha1": min_kl,
                "cross_component_mean_kl_alpha1": mean_kl,
                "meaningful_components": int(
                    group[
                        "component_has_meaningful_alpha1_effect"
                    ].sum()
                ),
                "causal_eligible": all_meaningful,
                "dose_non_decreasing_both_components": all_dose,
                "any_top1_change_alpha1": bool(
                    group[
                        "component_any_top1_change_alpha1"
                    ].any()
                ),
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    # IMPORTANT:
    # final pool follows the EXISTING component-aware association rank,
    # not a newly invented combined score.
    eligible_df = feature_df[
        feature_df["causal_eligible"]
    ].sort_values(
        "association_rank"
    )

    pool_df = eligible_df.head(
        FINAL_POOL_SIZE
    ).copy()

    selected_ids = set(
        pool_df["feature_id"].astype(int)
    )

    feature_df["selected_for_layer17_automl_pool"] = (
        feature_df["feature_id"]
        .astype(int)
        .isin(selected_ids)
    )

    feature_df = feature_df.sort_values(
        [
            "causal_eligible",
            "association_rank",
        ],
        ascending=[
            False,
            True,
        ],
    )

    ranking_path = (
        OUT_DIR
        / "feature_causal_ranking.csv"
    )
    pool_path = (
        OUT_DIR
        / "layer17_automl_feature_pool.csv"
    )

    feature_df.to_csv(
        ranking_path,
        index=False,
    )

    pool_df.to_csv(
        pool_path,
        index=False,
    )

    summary = {
        "broad_features_tested": len(broad),
        "components_per_feature": (
            N_COMPONENTS_PER_FEATURE
        ),
        "question_feature_cases": len(
            details_df
        ),
        "meaningful_kl_threshold": (
            MEANINGFUL_KL_THRESHOLD
        ),
        "causal_eligible_features": int(
            feature_df[
                "causal_eligible"
            ].sum()
        ),
        "final_pool_rule": (
            "First 8 by pre-existing component-aware "
            "association rank among features with "
            "KL(alpha=1)>1e-4 in both selected invariant "
            "components. Threshold is not relaxed."
        ),
        "final_layer17_pool": (
            pool_df["feature_id"]
            .astype(int)
            .tolist()
        ),
        "important_note": (
            "Causal eligibility only establishes that "
            "suppression affects model logits across two "
            "invariant components. It does not establish "
            "fairness improvement or feature semantics."
        ),
    }

    summary_path = (
        OUT_DIR
        / "run_summary.json"
    )
    write_json(
        summary_path,
        summary,
    )

    print("\n" + "=" * 80)
    print("CROSS-COMPONENT CAUSAL SCREEN FINISHED")
    print("=" * 80)
    print(
        "broad features tested:",
        summary[
            "broad_features_tested"
        ],
    )
    print(
        "causal eligible:",
        summary[
            "causal_eligible_features"
        ],
    )
    print(
        "final layer17 pool:",
        summary[
            "final_layer17_pool"
        ],
    )

    print("\nTop feature causal results:")
    print(
        feature_df.head(20)[
            [
                "feature_id",
                "association_rank",
                "association_score",
                "cross_component_min_kl_alpha1",
                "cross_component_mean_kl_alpha1",
                "meaningful_components",
                "causal_eligible",
                "dose_non_decreasing_both_components",
                "any_top1_change_alpha1",
                "selected_for_layer17_automl_pool",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        OUT_DIR
        / "selected_component_cases.csv"
    )
    print(details_path)
    print(component_path)
    print(ranking_path)
    print(pool_path)
    print(summary_path)

    print(
        "\nNo generation, semantic judge, or AutoML was used."
    )


if __name__ == "__main__":
    main()
