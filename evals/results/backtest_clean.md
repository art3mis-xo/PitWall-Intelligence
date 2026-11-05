# Leak-free backtest (validation 2025)

Date 2026-11-05T06:00:15+00:00 · commit `8d1f0263b03b07cb3b149b31b5cc9b90eb6c6934` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/backtest_clean.py`

Train 2018–2024 (2979 rows), evaluate 2025 (479 rows, 24 races). Per-race metrics, mean ± std over races. MAE is on predicted finishing rank.

| Model | Spearman | Top-10 overlap | MAE | NDCG@10 | Races beating baseline (Spearman) |
|---|---|---|---|---|---|
| Grid-position baseline (finish = grid) | 0.651 ± 0.207 | 0.779 ± 0.128 | 3.344 ± 1.124 | 0.905 ± 0.076 | — |
| XGBoost clean, 18 features | 0.615 ± 0.240 | 0.746 ± 0.138 | 3.720 ± 1.198 | 0.887 ± 0.084 | 8/24 |
| XGBoost clean, 20 features (+ race weather) | 0.625 ± 0.226 | 0.750 ± 0.138 | 3.640 ± 1.186 | 0.887 ± 0.092 | 9/24 |
