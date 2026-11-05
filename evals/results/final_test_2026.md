# Final test: 2026 (evaluated once)

Date 2026-11-05T06:03:11+00:00 · commit `5bac3c457fe85333da11c0872f7e215264978e10` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/final_test.py`

Train 2018–2025 (3458 rows); test 2026 rounds 1–14 (308 rows, 14 races). Per-race metrics, mean ± std over races. MAE is on predicted finishing rank.

| Model | Spearman | Top-10 overlap | MAE | NDCG@10 | Spearman vs grid baseline | Races beating baseline |
|---|---|---|---|---|---|---|
| Grid-position baseline (finish = grid) | 0.650 ± 0.182 | 0.750 ± 0.116 | 3.403 ± 1.126 | 0.855 ± 0.085 | — | — |
| Final model: Selected by evals/experiments.py: experiment H18, highest mean Spearman on validation 2025. | 0.667 ± 0.183 | 0.764 ± 0.108 | 3.461 ± 1.191 | 0.855 ± 0.099 | +0.017 | 7/14 |
| XGBoost, leak-free notebook features (reference, not tuned) | 0.659 ± 0.185 | 0.771 ± 0.107 | 3.539 ± 1.277 | 0.865 ± 0.083 | +0.009 | 7/14 |

Selection used 2025 only. 2026 was seen once before, in evals/backtest_audit.py, with the old leaky/leak-removed notebook features, before any of these choices were made.
