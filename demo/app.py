# app.py
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# --- Make sure "src" is importable no matter where you run from ---
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Project imports ---
from src.pipelines.judge import judge
from src.pipelines.human_eval import load_human_csv, summarize_agreement, AXES
from src.pipelines.metrics import (
    toxicity_score,
    summarize_scores,
    toxicity_breakdown,  # per-category scores
)

# ---------- Helpers ----------
def to_df(score):
    """Normalize float/dict → DataFrame with a 'composite' column."""
    if isinstance(score, (int, float)):
        return pd.DataFrame({"composite": [float(score)]})
    if isinstance(score, dict):
        if "composite" in score and isinstance(score["composite"], (int, float)):
            return pd.DataFrame({"composite": [float(score["composite"])]})
        for v in score.values():
            if isinstance(v, (int, float)):
                return pd.DataFrame({"composite": [float(v)]})
        return pd.DataFrame({"composite": [0.0]})
    # assume already a DataFrame
    return score


def show_toxicity_breakdown(model_name: str, tox_data: dict):
    """Render a simple bar chart for per-category toxicity."""
    if not isinstance(tox_data, dict) or not tox_data:
        return

    df = pd.DataFrame(list(tox_data.items()), columns=["category", "score"])

    # Keep the most interpretable labels if present
    keep = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit",
    ]
    if any(k in df["category"].values for k in keep):
        df = df[df["category"].isin(keep)]

    df = df.sort_values("score", ascending=False)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Score"),
            y=alt.Y("category:N", title="Toxicity Category", sort="-x"),
            tooltip=["category", alt.Tooltip("score:Q", format=".4f")],
        )
        .properties(title=f"{model_name} — Toxicity Breakdown", height=220)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------- UI ----------
st.set_page_config(page_title="FairEval Demo", layout="wide")
st.title("FairEval — Human-Aligned Evaluation for Generative Models")

left, right = st.columns(2)

with left:
    st.subheader("Prompt")
    prompt = st.text_area(
        "Enter a prompt:",
        "Summarize the article about climate change policy in 3 bullet points.",
    )
    st.subheader("Model A Output")
    out_a = st.text_area("Paste Model A output:", "Model A answer goes here...")
    st.subheader("Model B Output")
    out_b = st.text_area("Paste Model B output:", "Model B answer goes here...")

with right:
    st.subheader("LLM-as-Judge (Rubric) + Toxicity")
    rubric_path = ROOT / "config" / "prompts" / "judge_rubric.md"
    rubric_text = rubric_path.read_text(encoding="utf-8")

    if st.button("Score Both"):
        # Rubric scoring (LLM-as-judge)
        score_a = judge(prompt, out_a, rubric_text)
        score_b = judge(prompt, out_b, rubric_text)

        # Composite toxicity (single score per output)
        tox_a = toxicity_score(out_a)
        tox_b = toxicity_score(out_b)

        st.write("**Model A**")
        st.json(summarize_scores(to_df(tox_a)))
        st.write("**Model B**")
        st.json(summarize_scores(to_df(tox_b)))

        # Per-category toxicity breakdown (visual)
        st.write("### 🔬 Detailed Toxicity Breakdown")
        tox_a_breakdown = toxicity_breakdown(out_a) if out_a else {}
        tox_b_breakdown = toxicity_breakdown(out_b) if out_b else {}
        c1, c2 = st.columns(2)
        with c1:
            show_toxicity_breakdown("Model A", tox_a_breakdown)
        with c2:
            show_toxicity_breakdown("Model B", tox_b_breakdown)
    else:
        st.info("Enter outputs on the left, then click **Score Both**.")

# ---------- Human eval upload ----------
st.markdown("---")
st.subheader("Human Evaluation — Upload CSV (κ & ρ)")

uploaded = st.file_uploader("Upload human_eval.csv", type=["csv"])
if uploaded is not None:
    try:
        df = load_human_csv(uploaded)
        res = summarize_agreement(df, rubric_text)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Fleiss’ κ (inter-rater reliability)**")
            st.dataframe(pd.DataFrame([res.fleiss_kappa], index=["κ"]).T)
        with col2:
            st.write("**Judge ↔ Human (Spearman ρ)**")
            st.dataframe(pd.DataFrame([res.judge_vs_human_rho], index=["ρ"]).T)
        st.caption(
            f"Items {res.n_items} · Ratings {res.n_ratings} · Systems {', '.join(res.systems)}"
        )
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
else:
    st.caption("Upload data/processed/human_eval.csv to see κ and ρ metrics.")

# ---------- Model-level metrics ----------
st.markdown("---")
st.subheader("Model-Level Metrics  📊")

col1, col2 = st.columns(2)

with col1:
    st.write("### SQuAD-style EM/F1 Evaluation")
    golds = st.text_area("Enter gold references (one per line)", "")
    preds = st.text_area("Enter model predictions (one per line)", "")
    if st.button("Compute EM/F1"):
        from src.pipelines.metrics import evaluate_squad

        if golds.strip() and preds.strip():
            g = golds.strip().split("\n")
            p = preds.strip().split("\n")
            if len(g) != len(p):
                st.warning("⚠ Number of preds and golds must match")
            else:
                st.json(evaluate_squad(p, g))

with col2:
    st.write("### Uncertainty (Self-Consistency)")
    scores_text = st.text_area("Enter scores (e.g. 0.8, 0.76, 0.83)", "")
    if st.button("Compute Uncertainty"):
        from src.pipelines.metrics import estimate_uncertainty

        if scores_text.strip():
            scores = [float(x) for x in scores_text.split(",")]
            st.json(estimate_uncertainty(scores))

# ---------- Fairness & Toxicity Dashboard ----------
st.markdown("---")
st.subheader("Fairness & Toxicity Dashboard 🧭")

st.write(
    "Upload a CSV with columns: `group`, `text` (model output). "
    "We'll score toxicity per row and summarize by group."
)
csv_file = st.file_uploader("Upload CSV", type=["csv"])
if csv_file is not None:
    df_in = pd.read_csv(csv_file)
    required_cols = {"group", "text"}
    missing = required_cols - set(df_in.columns)
    if missing:
        st.error(f"Missing columns: {sorted(missing)}")
    else:
        from src.pipelines.metrics import toxicity_scores

        tox = toxicity_scores(df_in["text"].fillna("").tolist())
        df = pd.concat([df_in.reset_index(drop=True), tox.reset_index(drop=True)], axis=1)

        # Per-group summary
        grp = (
            df.groupby("group")["composite"]
            .agg(["mean", "median", "max", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        st.write("**Per-Group Toxicity Summary**")
        st.dataframe(grp, use_container_width=True)

        # Mean toxicity by group
        st.write("**Mean Toxicity by Group**")
        chart = (
            alt.Chart(grp)
            .mark_bar()
            .encode(x=alt.X("group:N", sort="-y"), y="mean:Q", tooltip=list(grp.columns))
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        # Distribution (density per group)
        st.write("**Toxicity Distribution by Group**")
        kde = (
            alt.Chart(df)
            .transform_density(
                "composite",
                as_=["composite", "density"],
                groupby=["group"],
                steps=128,
            )
            .mark_area(opacity=0.4)
            .encode(x="composite:Q", y="density:Q", color="group:N")
            .properties(height=280)
        )
        st.altair_chart(kde, use_container_width=True)

        # Global stats
        st.write("**Global Toxicity Stats**")
        st.json(summarize_scores(df))
