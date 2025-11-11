# src/pipelines/human_eval.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# IMPORTANT: use src.* here to avoid package confusion
from src.pipelines.judge import judge

AXES = ["helpfulness", "faithfulness", "harmlessness", "style", "sensitivity"]


@dataclass
class AgreementResult:
    fleiss_kappa: Dict[str, float]
    judge_vs_human_rho: Dict[str, float]
    n_items: int
    n_ratings: int
    systems: List[str]


def _coerce_0_5(x) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    v = min(5.0, max(0.0, v))
    return int(round(v))


def load_human_csv(path: str | bytes) -> pd.DataFrame:
    """Load the human eval CSV and normalize scores to integer bins 0..5."""
    df = pd.read_csv(path)
    required = {"item_id", "system", "prompt", "answer", "rater_id", *AXES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in AXES:
        df[col] = df[col].apply(_coerce_0_5).astype(int)
    df["item_id"] = df["item_id"].astype(str)
    df["system"] = df["system"].astype(str)
    return df


def _fleiss_kappa_from_counts(n_ij: np.ndarray) -> float:
    """
    n_ij: (N_items, k_categories) counts per rating category per item.
    Fleiss, 1971.
    """
    N, k = n_ij.shape
    if N == 0:
        return float("nan")
    n = n_ij.sum(axis=1)  # raters per item
    # guard: need at least 2 raters to define κ
    if np.any(n < 2):
        return float("nan")
    P_i = ((n_ij * (n_ij - 1)).sum(axis=1) / (n * (n - 1)))
    P_bar = P_i.mean()
    p_j = n_ij.sum(axis=0) / n.sum()
    P_e = (p_j ** 2).sum()
    denom = 1 - P_e
    if math.isclose(denom, 0.0):
        return float("nan")
    return float((P_bar - P_e) / denom)


def _counts_matrix(df_axis: pd.DataFrame) -> np.ndarray:
    """Build an (N_items, 6) matrix of counts for ratings 0..5."""
    mats = []
    for (_, _), g in df_axis.groupby(["item_id", "system"]):
        counts = np.zeros(6, dtype=int)
        for v in g.values:
            counts[int(v)] += 1
        mats.append(counts)
    return np.stack(mats, axis=0) if mats else np.zeros((0, 6), dtype=int)


def compute_fleiss_kappa(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ax in AXES:
        mat = _counts_matrix(df[["item_id", "system", ax]].rename(columns={ax: "score"})["score"].to_frame().join(
            df[["item_id", "system"]]
        ).drop_duplicates())
        # The above ensures one row per (item_id, system, rater) effectively via join/dedup
        # If you prefer explicit grouping:
        groups = df.groupby(["item_id", "system"])[ax].apply(lambda g: np.bincount(g, minlength=6))
        mat = np.vstack(groups.values) if len(groups) else np.zeros((0, 6), dtype=int)
        out[ax] = _fleiss_kappa_from_counts(mat)
    return out


def _judge_scores(df_items: pd.DataFrame, rubric_text: str) -> pd.DataFrame:
    # one judge score per (item_id, system); take first prompt/answer within group
    firsts = df_items.groupby(["item_id", "system"], as_index=False).first()[["item_id", "system", "prompt", "answer"]]
    rows = []
    for _, r in firsts.iterrows():
        s = judge(r["prompt"], r["answer"], rubric_text)
        rows.append({"item_id": r["item_id"], "system": r["system"], **{ax: s[ax] for ax in AXES}})
    return pd.DataFrame(rows)


def compute_judge_vs_human_rho(df: pd.DataFrame, rubric_text: str) -> Dict[str, float]:
    human_mean = df.groupby(["item_id", "system"], as_index=False)[AXES].mean()
    jdf = _judge_scores(df, rubric_text)
    merged = human_mean.merge(jdf, on=["item_id", "system"], suffixes=("_human", "_judge"))
    rho: Dict[str, float] = {}
    for ax in AXES:
        a = merged[f"{ax}_human"].values
        b = merged[f"{ax}_judge"].values
        rho[ax] = float(spearmanr(a, b)[0]) if len(merged) >= 2 else float("nan")
    return rho


def summarize_agreement(df: pd.DataFrame, rubric_text: str) -> AgreementResult:
    fleiss = compute_fleiss_kappa(df)
    rho = compute_judge_vs_human_rho(df, rubric_text)
    return AgreementResult(
        fleiss_kappa=fleiss,
        judge_vs_human_rho=rho,
        n_items=df[["item_id", "system"]].drop_duplicates().shape[0],
        n_ratings=df.shape[0],
        systems=sorted(df["system"].unique().tolist()),
    )
