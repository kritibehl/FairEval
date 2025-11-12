from src.pipelines.metrics import summarize_scores, estimate_uncertainty

def test_summarize_scores_smoke():
    import pandas as pd
    df = pd.DataFrame({"composite":[0.1,0.2,0.3]})
    res = summarize_scores(df)
    assert "mean" in res

def test_uncertainty_range():
    vals = [0.8,0.75,0.9]
    res = estimate_uncertainty(vals)
    assert 0 <= res["std"] <= 1
