# Paired test vs grid baseline (2026)

Date 2026-11-05T06:09:10+00:00 · commit `49007408099498ab01c0ef35e49c92ef6ac1f5d7` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/paired_test.py`

Per-race difference in Spearman (model − grid baseline) over 14 races of 2026, from `evals/results/final_test_2026.json`. 95% CI: percentile bootstrap over races (10,000 resamples, seed 0). p-value: Wilcoxon signed-rank, two-sided.

| Model | Mean difference | 95% CI | Races model better / baseline better | Wilcoxon p |
|---|---|---|---|---|
| Final model: Selected by evals/experiments.py: experiment H18, highest mean Spearman on validation 2025. | +0.016 | [-0.018, +0.055] | 7 / 7 | 0.626 |
| XGBoost, leak-free notebook features (reference, not tuned) | +0.009 | [-0.040, +0.068] | 7 / 7 | 0.808 |
