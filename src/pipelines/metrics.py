# src/pipelines/metrics.py
from __future__ import annotations
from typing import Dict, List, Any, Iterable, Optional
import numpy as np
import pandas as pd
import re

# ----------------------------
# Classic QA-style text metrics
# ----------------------------
def exact_match(pred: str, gold: str) -> int:
    return int((pred or "").strip().lower() == (gold or "").strip().lower())

def f1_overlap(pred: str, gold: str) -> float:
    pred_tokens = (pred or "").lower().split()
    gold_tokens = (gold or "").lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)

def evaluate_squad(preds: List[str], golds: List[str]) -> Dict[str, float]:
    em = np.mean([exact_match(p, g) for p, g in zip(preds, golds)])
    f1s = np.mean([f1_overlap(p, g) for p, g in zip(preds, golds)])
    return {"Exact Match (EM)": round(float(em), 3), "F1 Overlap": round(float(f1s), 3)}

# ----------------------------
# Uncertainty (fixed interface)
# ----------------------------
def estimate_uncertainty(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "low": 0.0, "high": 0.0, "n": 0}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0  # sample std
    z = 1.96  # ≈95% CI (normal approx)
    half = z * std / float(np.sqrt(n)) if n > 1 else 0.0
    low = max(0.0, mean - half)
    high = min(1.0, mean + half)
    return {"mean": mean, "std": std, "low": float(low), "high": float(high), "n": int(n)}

# ---------------------------------------
# Detoxify-based toxicity (optional, safe)
# ---------------------------------------
try:
    from detoxify import Detoxify  # type: ignore
except Exception:
    Detoxify = None  # gracefully handle missing/failed install

_DETOX_MODEL = None  # lazy cache

def _get_detox():
    """Return Detoxify model or None (no internet/SSL/offline)."""
    global _DETOX_MODEL
    if _DETOX_MODEL is not None:
        return _DETOX_MODEL
    if Detoxify is None:
        return None
    try:
        _DETOX_MODEL = Detoxify("unbiased")
    except Exception:
        _DETOX_MODEL = None
    return _DETOX_MODEL

# Lightweight fallback (keyword-based) so we never crash
_BAD_PATTERNS = [
    r"\bidiot\b", r"\bstupid\b", r"\bdumb\b", r"\bshut up\b",
    r"\bhate\b", r"\bkill\b", r"\btrash\b", r"\bworthless\b",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS), re.IGNORECASE)

def _fallback_toxicity(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    hits = len(_BAD_RE.findall(text))
    norm = max(1.0, len(text) / 50.0)
    return max(0.0, min(1.0, hits / norm))

def toxicity_breakdown(text: str) -> Dict[str, float]:
    """
    Returns per-category toxicity scores as floats. If Detoxify unavailable, returns {}.
    """
    model = _get_detox()
    if model is None or not isinstance(text, str) or not text.strip():
        return {}
    try:
        scores = model.predict(text)  # dict[str, float]
        return {k: float(v) for k, v in scores.items()}
    except Exception:
        return {}

def toxicity_score(text: str) -> Dict[str, float]:
    """
    Returns {'composite': float in [0,1]}.
    Uses Detoxify if available; otherwise a simple keyword fallback.
    """
    bd = toxicity_breakdown(text)
    if bd:
        comp = float(bd.get("toxicity", sum(bd.values()) / max(1, len(bd))))
        return {"composite": comp}
    # Fallback heuristic
    return {"composite": _fallback_toxicity(text)}

def toxicity_scores(texts: List[str]) -> pd.DataFrame:
    """
    Batch scoring -> DataFrame with at least a 'composite' column.
    """
    rows: List[Dict[str, float]] = []
    for t in texts:
        try:
            d = toxicity_score(t if isinstance(t, str) else "")
            rows.append({"composite": float(d.get("composite", 0.0))})
        except Exception:
            rows.append({"composite": 0.0})
    return pd.DataFrame(rows)

def summarize_scores(data: Any, col: str = "composite") -> Dict[str, float]:
    """
    Accepts:
      - float/int
      - dict (e.g., {'composite': 0.17} or any single numeric)
      - pandas.DataFrame

    Returns: {'mean','std','max','n'}
    """
    # number
    if isinstance(data, (int, float)):
        val = float(data)
        return {"mean": val, "std": 0.0, "max": val, "n": 1}

    # dict
    if isinstance(data, dict):
        if col in data and isinstance(data[col], (int, float)):
            val = float(data[col])
            return {"mean": val, "std": 0.0, "max": val, "n": 1}
        nums = [float(v) for v in data.values() if isinstance(v, (int, float))]
        if len(nums) == 1:
            val = nums[0]
            return {"mean": val, "std": 0.0, "max": val, "n": 1}
        data = pd.DataFrame([data])

    # DataFrame
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            series = pd.to_numeric(data[col], errors="coerce").dropna()
        else:
            num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            if len(num_cols) == 1:
                series = pd.to_numeric(data[num_cols[0]], errors="coerce").dropna()
            else:
                return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        vals = series.to_numpy()
        if len(vals) == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }

    # fallback
    return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
