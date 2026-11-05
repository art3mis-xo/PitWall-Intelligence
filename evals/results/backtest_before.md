# Backtest before the leakage fix

Date 2026-11-05T05:55:00+00:00 · commit `0094f29c4d29fc2ddc286f3c967f96103db74cf5` · xgboost 3.3.0 XGBRegressor · 3766 rows, 187 races (2018-2026) · `python evals/backtest_audit.py`

Train on all seasons before the holdout; per-race metrics averaged over the holdout's races.

| Holdout | Model | Races | Spearman (mean ± std) | Top-10 overlap | MAE |
|---|---|---|---|---|---|
| 2025 | Grid-position baseline (finish = grid) | 24 | 0.651 ± 0.207 | 0.779 | 3.344 |
| 2025 | avg_team_finish_position alone (no model) | 24 | 0.813 ± 0.079 | 0.833 | 2.507 |
| 2025 | XGBoost, all 20 features (notebook) | 24 | 0.874 ± 0.098 | 0.875 | 1.970 |
| 2025 | XGBoost, leaky removed (13 features, race weather kept) | 24 | 0.601 ± 0.233 | 0.733 | 3.453 |
| 2025 | XGBoost, leaky + race weather removed (11 features) | 24 | 0.611 ± 0.211 | 0.729 | 3.430 |
| 2026 | Grid-position baseline (finish = grid) | 14 | 0.650 ± 0.182 | 0.750 | 3.403 |
| 2026 | avg_team_finish_position alone (no model) | 14 | 0.806 ± 0.107 | 0.821 | 2.545 |
| 2026 | XGBoost, all 20 features (notebook) | 14 | 0.805 ± 0.139 | 0.829 | 2.485 |
| 2026 | XGBoost, leaky removed (13 features, race weather kept) | 14 | 0.650 ± 0.187 | 0.743 | 3.679 |
| 2026 | XGBoost, leaky + race weather removed (11 features) | 14 | 0.646 ± 0.206 | 0.743 | 3.660 |
