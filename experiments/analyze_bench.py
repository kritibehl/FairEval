#!/usr/bin/env python3
"""
Analyze FairEval bench results.

- Input: experiments/results/faireval_results.csv (default; or pass a path)
- Ensures there's a 'composite' column (builds it if missing)
- Prints a per-model summary to stdout
- Writes:
    experiments/results/summary_by_model.csv
    experiments/results/README_results.md
"""

import sys
from pathlib import Path
import pandas as pd

def ensure_composite(df: pd.DataFrame) -> pd.DataFrame:
    if "composite" in df.columns:
        return df

    # If no 'composite', try to construct from any numeric columns other than model/prompt/output
    ignore = {"model", "prompt", "output"}
    numeric_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        # last resort—create a 0.0 composite
        df["composite"] = 0.0
        return df

    # Heuristic: if exactly one numeric column, rename it to composite
    if len(numeric_cols) == 1:
        df = df.rename(columns={numeric_cols[0]: "composite"})
        return df

    # Otherwise, average numeric columns row-wise as composite
    df["composite"] = df[numeric_cols].mean(axis=1)
    return df

def to_markdown_table(df: pd.DataFrame) -> str:
    # Pretty markdown table without index
    return df.to_markdown(index=False, floatfmt=".6f")

def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/faireval_results.csv")
    if not results_path.exists():
        print(f"[analyze_bench] Results file not found: {results_path}")
        sys.exit(1)

    out_dir = results_path.parent
    df = pd.read_csv(results_path)
    if df.empty:
        print("[analyze_bench] CSV is empty.")
        sys.exit(1)

    df = ensure_composite(df)

    if "model" not in df.columns:
        print("[analyze_bench] Missing 'model' column in results.")
        sys.exit(1)

    # Per-model summary
    summary = (
        df.groupby("model")["composite"]
        .agg(["mean", "median", "max", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    # Print quick view
    print("\n=== Tail of results ===")
    print(df.tail(5))
    print("\n=== Columns ===")
    print(list(df.columns))
    print("\n=== Per-model toxicity summary (higher = worse) ===")
    print(summary)

    # Save artifacts
    summary_csv = out_dir / "summary_by_model.csv"
    summary_md  = out_dir / "README_results.md"
    summary.to_csv(summary_csv, index=False)

    md = [
        "# FairEval – Benchmark Summary",
        "",
        f"**Source:** `{results_path}`",
        "",
        "## Per-model Toxicity Summary (higher = worse)",
        "",
        to_markdown_table(summary),
        "",
        "> *Composite = toxicity score; lower is better.*",
    ]
    summary_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\n[analyze_bench] Wrote:\n- {summary_csv}\n- {summary_md}")

if __name__ == "__main__":
    main()
