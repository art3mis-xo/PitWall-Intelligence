# Experiments log (validation 2025 only)

Date 2026-11-05T06:02:46+00:00 · commit `4e897b77a3d8b4403d8eaf11cdeb90790029450d` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/experiments.py`

Rules fixed before running: primary metric is mean per-race Spearman on 2025 (tie-break NDCG@10). Stage 1 compares objectives; stage 2 adds pre-race feature groups one at a time and keeps a group only if Spearman improves; stage 3 is a random hyperparameter search (20 draws, seed 0). 2026 is not used. MAE is on predicted finishing rank.

Grid-position baseline on 2025: Spearman 0.651 ± 0.207, top-10 0.779 ± 0.128, MAE 3.344 ± 1.124, NDCG@10 0.905 ± 0.076.

| ID | Stage | Experiment | Features | Spearman | Top-10 | MAE | NDCG@10 | Beats baseline (races) | Kept |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 1 | XGB regression on finish position | 18 | 0.615 ± 0.240 | 0.746 ± 0.138 | 3.720 ± 1.198 | 0.887 ± 0.084 | 8/24 |  |
| E2 | 1 | XGB regression on finish - grid (grid residual) | 18 | 0.627 ± 0.230 | 0.763 ± 0.124 | 3.624 ± 1.225 | 0.893 ± 0.079 | 9/24 |  |
| E3 | 1 | LightGBM lambdarank (group = race) | 18 | 0.628 ± 0.232 | 0.758 ± 0.121 | 3.599 ± 1.103 | 0.883 ± 0.096 | 9/24 |  |
| E4 | 2 | E3 + quali: qualifying position, best-lap gap to pole (%), position vs teammate | 21 | 0.632 ± 0.229 | 0.750 ± 0.122 | 3.575 ± 1.198 | 0.885 ± 0.092 | 9/24 | yes |
| E5 | 2 | E4 + pit_lane: pit-lane start flag | 22 | 0.630 ± 0.240 | 0.750 ± 0.102 | 3.554 ± 1.271 | 0.882 ± 0.090 | 10/24 | no |
| E6 | 2 | E4 + driver_form: driver average finish over last 3 / 5 races, points over last 5 | 24 | 0.635 ± 0.233 | 0.758 ± 0.110 | 3.541 ± 1.168 | 0.884 ± 0.092 | 12/24 | yes |
| E7 | 2 | E6 + reliability: driver and team DNF rate over last 10 races | 26 | 0.638 ± 0.226 | 0.754 ± 0.125 | 3.521 ± 1.155 | 0.884 ± 0.088 | 12/24 | yes |
| E8 | 2 | E7 + circuit_history: past position changes at this circuit (all drivers / this driver) | 28 | 0.643 ± 0.224 | 0.767 ± 0.117 | 3.516 ± 1.151 | 0.885 ± 0.096 | 12/24 | yes |
| H1 | 3 | E8 hyperparameters: n_estimators=500, num_leaves=15, learning_rate=0.02, min_child_samples=20, colsample_bytree=1.0, reg_lambda=1.0 | 28 | 0.653 ± 0.217 | 0.746 ± 0.118 | 3.390 ± 1.174 | 0.893 ± 0.076 | 11/24 |  |
| H2 | 3 | E8 hyperparameters: n_estimators=500, num_leaves=15, learning_rate=0.05, min_child_samples=20, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.632 ± 0.233 | 0.754 ± 0.110 | 3.571 ± 1.208 | 0.879 ± 0.098 | 12/24 |  |
| H3 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=15, learning_rate=0.02, min_child_samples=10, colsample_bytree=1.0, reg_lambda=1.0 | 28 | 0.661 ± 0.208 | 0.762 ± 0.117 | 3.328 ± 1.081 | 0.900 ± 0.064 | 12/24 |  |
| H4 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=15, learning_rate=0.02, min_child_samples=40, colsample_bytree=0.7, reg_lambda=5.0 | 28 | 0.658 ± 0.211 | 0.767 ± 0.120 | 3.325 ± 1.112 | 0.900 ± 0.077 | 12/24 |  |
| H5 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=15, learning_rate=0.1, min_child_samples=10, colsample_bytree=0.85, reg_lambda=1.0 | 28 | 0.630 ± 0.224 | 0.746 ± 0.118 | 3.595 ± 1.158 | 0.886 ± 0.086 | 9/24 |  |
| H6 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=31, learning_rate=0.1, min_child_samples=10, colsample_bytree=1.0, reg_lambda=1.0 | 28 | 0.622 ± 0.228 | 0.750 ± 0.114 | 3.628 ± 1.158 | 0.889 ± 0.082 | 11/24 |  |
| H7 | 3 | E8 hyperparameters: n_estimators=500, num_leaves=31, learning_rate=0.05, min_child_samples=10, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.608 ± 0.234 | 0.758 ± 0.128 | 3.737 ± 1.130 | 0.883 ± 0.090 | 7/24 |  |
| H8 | 3 | E8 hyperparameters: n_estimators=100, num_leaves=31, learning_rate=0.05, min_child_samples=40, colsample_bytree=1.0, reg_lambda=5.0 | 28 | 0.656 ± 0.214 | 0.754 ± 0.122 | 3.370 ± 1.091 | 0.890 ± 0.085 | 13/24 |  |
| H9 | 3 | E8 hyperparameters: n_estimators=100, num_leaves=31, learning_rate=0.05, min_child_samples=20, colsample_bytree=0.7, reg_lambda=5.0 | 28 | 0.657 ± 0.213 | 0.754 ± 0.110 | 3.345 ± 1.062 | 0.894 ± 0.084 | 13/24 |  |
| H10 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=31, learning_rate=0.02, min_child_samples=10, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.645 ± 0.218 | 0.750 ± 0.114 | 3.458 ± 1.137 | 0.889 ± 0.077 | 12/24 |  |
| H11 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=7, learning_rate=0.1, min_child_samples=20, colsample_bytree=0.7, reg_lambda=0.0 | 28 | 0.641 ± 0.222 | 0.750 ± 0.102 | 3.491 ± 1.088 | 0.887 ± 0.084 | 11/24 |  |
| H12 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=10, colsample_bytree=0.85, reg_lambda=5.0 | 28 | 0.635 ± 0.227 | 0.754 ± 0.110 | 3.491 ± 1.143 | 0.884 ± 0.088 | 10/24 |  |
| H13 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=31, learning_rate=0.02, min_child_samples=40, colsample_bytree=0.85, reg_lambda=5.0 | 28 | 0.657 ± 0.212 | 0.746 ± 0.118 | 3.353 ± 1.096 | 0.891 ± 0.077 | 13/24 |  |
| H14 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=31, learning_rate=0.1, min_child_samples=40, colsample_bytree=0.85, reg_lambda=1.0 | 28 | 0.637 ± 0.222 | 0.750 ± 0.114 | 3.570 ± 1.109 | 0.885 ± 0.086 | 12/24 |  |
| H15 | 3 | E8 hyperparameters: n_estimators=100, num_leaves=31, learning_rate=0.05, min_child_samples=20, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.643 ± 0.216 | 0.746 ± 0.125 | 3.471 ± 1.114 | 0.888 ± 0.078 | 11/24 |  |
| H16 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=7, learning_rate=0.02, min_child_samples=10, colsample_bytree=0.7, reg_lambda=5.0 | 28 | 0.667 ± 0.200 | 0.763 ± 0.117 | 3.307 ± 1.072 | 0.900 ± 0.074 | 14/24 |  |
| H17 | 3 | E8 hyperparameters: n_estimators=300, num_leaves=15, learning_rate=0.02, min_child_samples=10, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.654 ± 0.216 | 0.763 ± 0.106 | 3.349 ± 1.107 | 0.892 ± 0.084 | 11/24 |  |
| H18 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=7, learning_rate=0.02, min_child_samples=40, colsample_bytree=1.0, reg_lambda=5.0 | 28 | 0.668 ± 0.200 | 0.763 ± 0.121 | 3.282 ± 1.057 | 0.899 ± 0.075 | 14/24 |  |
| H19 | 3 | E8 hyperparameters: n_estimators=500, num_leaves=31, learning_rate=0.1, min_child_samples=20, colsample_bytree=1.0, reg_lambda=0.0 | 28 | 0.630 ± 0.230 | 0.758 ± 0.132 | 3.587 ± 1.211 | 0.879 ± 0.097 | 9/24 |  |
| H20 | 3 | E8 hyperparameters: n_estimators=200, num_leaves=31, learning_rate=0.1, min_child_samples=20, colsample_bytree=1.0, reg_lambda=1.0 | 28 | 0.627 ± 0.216 | 0.758 ± 0.118 | 3.629 ± 1.067 | 0.890 ± 0.080 | 8/24 |  |

**Selected:** H18 (E8 hyperparameters: n_estimators=200, num_leaves=7, learning_rate=0.02, min_child_samples=40, colsample_bytree=1.0, reg_lambda=5.0), Spearman 0.668 ± 0.200 vs baseline 0.651 ± 0.207. Written to `configs/final.json`.
