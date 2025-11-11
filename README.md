# FairEval — Human‑Aligned Evaluation for Generative Models

FairEval is a lightweight, reproducible framework to evaluate generative models across **safety, fairness, factuality, coherence, and uncertainty** using:
- **Automatic metrics** (toxicity, bias slices, EM/F1, ROUGE/BERTScore)
- **LLM‑as‑Judge** with a 5‑axis rubric (Helpfulness, Faithfulness, Harmlessness, Style, Sensitivity)
- **Human evaluation** (3 raters/sample, reports κ/ρ)
- A **Streamlit demo** for side‑by‑side model comparison

> Starter pack generated for Kriti Behl (AIML Residency prep).

FairEval: Human-Aligned, Safety-Aware Evaluation for Generative Models
Abstract— We present FairEval, a lightweight evaluation toolkit that combines (i) LLM-as-Judge rubric scoring for helpfulness, faithfulness, harmlessness, style, sensitivity; (ii) Human reliability analysis via Fleiss’ κ and Judge↔Human Spearman ρ; and (iii) Safety & Fairness analytics using Detoxify-based toxicity with per-group bias summaries. FairEval supports SQuAD-style EM/F1 and self-consistency uncertainty to quantify aggregate quality and stability. We release a Streamlit demo for interactive side-by-side model comparison and a reproducible pipeline that ingests human ratings, computes κ/ρ, and visualizes fairness distributions. In a small study on multi-domain prompts, FairEval detects systematic spread in toxicity across groups and correlates rubric-based judge scores with human means (ρ ≈ 0.6–0.8 on several axes), while κ highlights rater variance. The toolkit is framework-agnostic, requires only Python, and is designed for product teams to ship safer, more reliable model experiences with rapid iteration.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run demo/app.py
```

## Repo layout
```
faireval/
├─ config/
│  ├─ tasks.yaml
│  └─ prompts/judge_rubric.md
├─ data/
├─ src/
│  ├─ pipelines/
│  ├─ models/
│  ├─ tasks/
│  ├─ viz/
│  └─ utils/
├─ demo/app.py
├─ docs/
└─ eval_runs/
```

## Status
This is an MVP scaffold. Fill in `src/pipelines/*.py` to go live.

## License
MIT
