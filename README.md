# FairEval — Human-Aligned Evaluation for Generative Models

![Tests](https://github.com/kritibehl/FairEval/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/kritibehl/FairEval/branch/main/graph/badge.svg)](https://codecov.io/gh/kritibehl/FairEval)

**FairEval** is a lightweight, reproducible framework for evaluating generative models across **safety, fairness, factuality, coherence, and uncertainty** using:
- **Automatic metrics** — toxicity, bias slices, EM/F1, ROUGE, BERTScore  
- **LLM-as-Judge** — 5-axis rubric (Helpfulness, Faithfulness, Harmlessness, Style, Sensitivity)  
- **Human evaluation** — 3 raters/sample, reporting κ and ρ  
- **Streamlit demo** — side-by-side model comparison  

---

## Overview
FairEval unifies **automatic**, **LLM-based**, and **human** evaluations to measure both model quality and ethical alignment.  
It provides an end-to-end pipeline for:
- Metric computation and aggregation  
- Human-AI agreement analysis (Fleiss’ κ, Spearman ρ)  
- Fairness & toxicity analytics using Detoxify  
- Uncertainty estimation for stability checks  
- A Streamlit dashboard for model-to-model comparisons  

---

## Abstract
**FairEval: Human-Aligned, Safety-Aware Evaluation for Generative Models**  
We present FairEval, a reproducible evaluation toolkit that combines:
1. **LLM-as-Judge rubric scoring** across helpfulness, faithfulness, harmlessness, style, and sensitivity  
2. **Human reliability analysis** via Fleiss’ κ and Judge↔Human Spearman ρ  
3. **Safety & Fairness analytics** with Detoxify-based toxicity and bias slice reporting  

FairEval supports SQuAD-style EM/F1, BERTScore, and self-consistency uncertainty estimation to quantify model reliability.  
A Streamlit demo enables interactive, side-by-side model evaluation, while a reproducible pipeline ingests human ratings, computes reliability metrics, and visualizes fairness distributions.  
The framework is Python-only, lightweight, and designed for product and research teams seeking to ship safer, more trustworthy model experiences.

---

Medium Article:
“FairEval — A Human-Aligned Evaluation Framework for Generative Models”
https://medium.com/@kriti0608/faireval-a-human-aligned-evaluation-framework-for-generative-models-d822bfd5c99d

Citation  
If you use FairEval, please cite:

Behl, K. (2025). FairEval: Human-Aligned Evaluation Framework for Generative Models (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17625268

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run demo/app.py
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

an-Aligned Evaluation Framework for Generative Models (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17625268
