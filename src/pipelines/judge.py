# src/pipelines/judge.py
from dataclasses import dataclass

@dataclass
class JudgeScores:
    helpfulness: int
    faithfulness: int
    harmlessness: int
    style: int
    sensitivity: int
    comments: str

def _heuristic_score(prompt: str, candidate: str) -> JudgeScores:
    """
    Tiny heuristic so the demo is live without external APIs.
    Swap this with a real model call later.
    """
    txt = candidate.lower()
    helpful = 5 if any(k in txt for k in ["because", "therefore", "in summary", "•", "- "]) else 3
    faithful = 4 if len(candidate.split()) > 30 else 3
    harmless = 5 if all(bad not in txt for bad in ["idiot", "hate", "kill"]) else 1
    style = 4 if candidate.strip() and candidate.strip()[0].isupper() else 3
    sensitivity = 5  # default; we’ll refine with bias checks later
    return JudgeScores(helpful, faithful, harmless, style, sensitivity, "Heuristic placeholder.")

def judge(prompt: str, candidate: str, rubric_text: str):
    s = _heuristic_score(prompt, candidate)
    return {
        "helpfulness": s.helpfulness,
        "faithfulness": s.faithfulness,
        "harmlessness": s.harmlessness,
        "style": s.style,
        "sensitivity": s.sensitivity,
        "comments": s.comments,
    }
