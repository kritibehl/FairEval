# FairEval — Human-Aligned Evaluation Framework for Generative Models

![Tests](https://github.com/kritibehl/FairEval/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/kritibehl/FairEval/branch/main/graph/badge.svg)](https://codecov.io/gh/kritibehl/FairEval)

FairEval is a reproducible evaluation framework for large language models designed to measure safety, fairness, factuality, hallucination behavior, clarity, and indirect intent understanding.  
It integrates automatic metrics, LLM-as-judge scoring, and human evaluation into a unified system.

---

## Overview

FairEval provides:

- Automatic metrics (toxicity, EM/F1, ROUGE, BERTScore, hallucination checks)
- LLM-as-Judge evaluation using a structured rubric (Helpfulness, Faithfulness, Harmlessness, Style, Sensitivity)
- Human evaluation pipeline (multi-rater annotations, Fleiss’ κ, Spearman ρ)
- Fairness and uncertainty analysis
- Streamlit demo for side-by-side model comparisons
- Config-driven, reproducible evaluation suites

---

## Abstract

**FairEval: A Human-Aligned, Safety-Aware Evaluation Framework for Generative Models**

FairEval combines three evaluation paradigms:

1. LLM-as-Judge rubric scoring  
2. Human reliability analysis  
3. Automatic safety and fairness checks  

It additionally supports EM/F1, ROUGE, BERTScore, self-consistency uncertainty, and toxicity analysis.  
A Streamlit demo enables interactive model-to-model comparisons, while pipelines compute reliability and fairness metrics.

The framework is Python-based, lightweight, and built for applied ML and research teams.

---

## Medium Article

FairEval — A Human-Aligned Evaluation Framework for Generative Models  
https://medium.com/@kriti0608/faireval-a-human-aligned-evaluation-framework-for-generative-models-d822bfd5c99d

---

## Citation

If you use FairEval, please cite:

Behl, K. (2025). FairEval: Human-Aligned Evaluation Framework for Generative Models (v1.0.0). Zenodo.  
https://doi.org/10.5281/zenodo.17625268

@software{Behl_FairEval_2025,
author = {Kriti Behl},
title = {FairEval: Human-Aligned Evaluation Framework for Generative Models},
year = {2025},
doi = {10.5281/zenodo.17625268},
url = {https://doi.org/10.5281/zenodo.17625268}

}


---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Run the Streamlit demo
streamlit run demo/app.py

Run an evaluation suite
python -m faireval.cli run \
  --config faireval/config/tasks.yaml

