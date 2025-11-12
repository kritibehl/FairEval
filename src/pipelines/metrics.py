# src/pipelines/metrics.py
from __future__ import annotations
from typing import Dict, List, Any
from functools import lru_cache
import numpy as np
import pandas as pd
from dataclasses import dataclass
import re


# --- Classic QA metrics (you already used these) ---
def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())

def f1_overlap(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
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

def estimate_uncertainty(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {"Mean": 0.0, "Std": 0.0, "95 % CI": 0.0}
    mean = arr.mean()
    std = arr.std(ddof=0)
    ci95 = 1.96 * std / np.sqrt(arr.size)
    return {"Mean": round(float(mean), 3), "Std": round(float(std), 3), "95 % CI": round(float(ci95), 3)}

# --- Toxicity metrics (Detoxify) ---
# We expose two names your app imports: toxicity_score (single string) and summarize_scores (aggregate)
# Plus a vectorized helper toxicity_scores(texts) that returns a DataFrame.

_detox_model = None

def _get_detox():
    global _detox_model
    if _detox_model is None:
        from detoxify import Detoxify
        # "unbiased" model has multiple labels (toxicity, insult, threat, etc.)
        _detox_model = Detoxify("unbiased")
    return _detox_model

def toxicity_score(text: str) -> Dict[str, float]:
    """
    Score a single string. Returns per-label probabilities and a composite = max label.
    """
    if not isinstance(text, str) or not text.strip():
        return {"composite": 0.0}
    model = _get_detox()
    # Detoxify can handle list or str; we normalize to list and take first row
    preds = model.predict([text])
    df = pd.DataFrame(preds)
    df["composite"] = df.max(axis=1)
    return {k: float(df.iloc[0][k]) for k in df.columns}

def toxicity_scores(texts: list[str]) -> pd.DataFrame:
    """
    Batch score a list of texts. Returns a DataFrame with a 'composite' column
    and per-category columns from Detoxify.
    """
    model = _get_detox()
    # Detoxify can take a list; returns dict of lists
    raw = model.predict([t if isinstance(t, str) else "" for t in texts])
    df = pd.DataFrame(raw)
    # If no explicit composite is provided, use 'toxicity' as composite.
    if "composite" not in df.columns:
        if "toxicity" in df.columns:
            df["composite"] = df["toxicity"]
        else:
            df["composite"] = 0.0
    return df


def summarize_scores(data, col: str = "composite"):
    """
    Accepts:
      - float/int (single score)
      - dict (e.g., {"composite": 0.17} or any single numeric)
      - pandas.DataFrame (with a numeric column; defaults to 'composite')

    Returns a summary dict: mean/std/max/n
    """
    # --- number ---
    if isinstance(data, (int, float)):
        val = float(data)
        return {"mean": val, "std": 0.0, "max": val, "n": 1}

    # --- dict ---
    if isinstance(data, dict):
        if col in data and isinstance(data[col], (int, float)):
            val = float(data[col])
            return {"mean": val, "std": 0.0, "max": val, "n": 1}
        # if dict has a single numeric value, use it
        nums = [float(v) for v in data.values() if isinstance(v, (int, float))]
        if len(nums) == 1:
            val = nums[0]
            return {"mean": val, "std": 0.0, "max": val, "n": 1}
        # else, turn dict into one-row frame and continue
        data = pd.DataFrame([data])

    # --- DataFrame ---
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        use_col = None
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            use_col = col
        else:
            num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            if len(num_cols) == 1:
                use_col = num_cols[0]
            else:
                return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        vals = pd.to_numeric(data[use_col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }

    # --- fallback ---
    return {"mean": 0.0, "std": 0.0, "max": 0.0, "n": 0}
# src/pipelines/metrics.py

_detox_model = None   # cache after first load

def _get_detox():
    global _detox_model
    if _detox_model is not None:
        return _detox_model
    try:
        from detoxify import Detoxify
        _detox_model = Detoxify("unbiased")  # downloads weights on first use
    except Exception as e:
        # Could not load model (SSL, offline, etc.). Keep None to trigger fallback.
        _detox_model = None
        print(f"[FairEval] Detoxify unavailable, using fallback scorer: {e}")
    return _detox_model

# --- lightweight fallback toxicity (keyword-based) ---
_BAD_PATTERNS = [
    r"\bidiot\b", r"\bstupid\b", r"\bdumb\b", r"\bshut up\b",
    r"\bhate\b", r"\bkill\b", r"\btrash\b", r"\bworthless\b",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS), re.IGNORECASE)

def _fallback_toxicity(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    # score = (# toxic hits) / (len(text)/50 + 1) capped to [0,1]
    hits = len(_BAD_RE.findall(text))
    norm = max(1.0, len(text) / 50.0)
    return max(0.0, min(1.0, hits / norm))

def toxicity_score(text: str) -> Dict[str, float]:
    """
    Returns a dict with 'composite' (0..1). Uses Detoxify if available,
    otherwise a simple heuristic so the app never crashes.
    """
    model = _get_detox()
    if model is None:
        return {"composite": _fallback_toxicity(text)}
    scores = model.predict(text or "")
    # Detoxify returns dict of labels; we use 'toxicity' when present or mean of available
    if "toxicity" in scores:
        comp = float(scores["toxicity"])
    else:
        comp = float(sum(scores.values()) / max(1, len(scores)))
    return {"composite": comp}

def toxicity_scores(texts):
    # convenience: vectorized scoring
    import pandas as pd
    return pd.DataFrame([toxicity_score(t) for t in texts])

def summarize_scores(df):
    # expects a column 'composite'
    import numpy as np
    vals = df["composite"].astype(float).to_numpy()
    return {
        "mean": float(np.mean(vals)) if len(vals) else 0.0,
        "std": float(np.std(vals)) if len(vals) else 0.0,
        "max": float(np.max(vals)) if len(vals) else 0.0,
        "n": int(len(vals)),
    }
    
def toxicity_breakdown(text: str) -> Dict[str, float]:
    """
    Return per-category Detoxify scores for a single text.
    Keys usually include: 'toxicity', 'severe_toxicity', 'obscene',
    'identity_attack', 'insult', 'threat', 'sexual_explicit', etc.
    """
    if not text or not isinstance(text, str):
        return {}
    model = _get_detox()  # reuse your cached Detoxify instance
    # Detoxify returns a dict of lists (because it can batch). Handle single example.
    out = model.predict([text])
    # convert any list values like [0.0123] -> 0.0123
    return {k: (v[0] if isinstance(v, (list, tuple)) and v else float(v)) for k, v in out.items()}

def _safe_get_detox():
    """
    Returns a Detoxify model or None if loading fails (offline/SSL/etc).
    We never raise to the caller.
    """
    try:
        from detoxify import Detoxify
        # Try unbiased; Detoxify will cache weights under ~/.cache/torch
        return Detoxify("unbiased")
    except Exception as e:
        # Log-friendly: return None, callers will soft-fallback
        print(f"[FairEval] Detoxify unavailable: {e}")
        return None

def toxicity_breakdown(text: str) -> Dict[str, float]:
    """
    Returns per-category toxicity scores. If model unavailable, returns {}.
    """
    model = _safe_get_detox()
    if not model or not text:
        return {}
    try:
        scores = model.predict(text)
        # ensure plain floats
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"[FairEval] Detoxify predict failed: {e}")
        return {}

def toxicity_score(text: str) -> float:
    """
    Returns a single composite toxicity score in [0,1].
    If model unavailable or fails, returns 0.0 (neutral) so app never crashes.
    """
    bd = toxicity_breakdown(text)
    if not bd:
        return 0.0
    # common primary key; otherwise take mean of all categories
    if "toxicity" in bd:
        return float(bd["toxicity"])
    return float(sum(bd.values()) / max(1, len(bd)))