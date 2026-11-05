# Finding and removing target leakage

**Short version.** The original model looked strong (Spearman 0.874 on 2025) because some of its inputs were computed from the race it was predicting. After rebuilding every input from information available before the race, the best model scores Spearman 0.667 on 14 held-out 2026 races, against 0.650 for simply predicting "finish = grid position". The difference is not statistically meaningful: the leak-free model matches a strong baseline; it does not beat it.

Every number below comes from a file in this folder, produced by a committed script.

## 1. What was wrong

The model predicts each driver's finishing position. Some of its 20 inputs used information that only exists once the race is over:

| Feature | What it actually measured | Why that leaks |
|---|---|---|
| `avg_team_finish_position` | Average finishing position of the driver's team **in this race** | It contains the teammate's result, part of the answer |
| `rolling_avg_finish_position` | Team average over the last 3 races, **including this one** | A third of it is the answer |
| `var_delta_i_norm` | How much drivers gained or lost places **in this race** | Needs everyone's finishing position |
| `teammate_wins`, `teammate_losses`, `teammate_win_pct` | Head-to-head record counted over **all seasons 2018–2026** | When predicting a 2019 race, it already knows 2020–2026 |
| `circuit_wet_race_frequency` | Share of wet races at the circuit over **all seasons** | Includes future races and this race |
| `race_avg_air_temp`, `race_avg_track_temp` | Measured **during** the race | Borderline: a forecast would be known, the measurement is not |

From [`backtest_before.md`](backtest_before.md) (`python evals/backtest_audit.py`), training on all seasons before the holdout:

| Holdout | Model | Spearman |
|---|---|---|
| 2025 | Grid-position baseline | 0.651 |
| 2025 | `avg_team_finish_position` alone, no model | 0.813 |
| 2025 | Original model, all 20 features | **0.874** |
| 2025 | Original model, 7 leaky features removed | 0.601 |
| 2026 | Grid-position baseline | 0.650 |
| 2026 | `avg_team_finish_position` alone, no model | 0.806 |
| 2026 | Original model, all 20 features | **0.805** |
| 2026 | Original model, 7 leaky features removed | 0.650 |

One leaked column with no model at all scored about as well as the full model. Without the leaks, the model did not beat the grid.

## 2. How each feature was fixed

In [`2Feature_engineering.py`](../../2Feature_engineering.py), every history feature now uses only races **before** the one being predicted (sorted by season and round, shifted by one race):

| Old feature | Replacement |
|---|---|
| `avg_team_finish_position` | `team_avg_finish_last5`: team average finish over the previous 5 races |
| `rolling_avg_finish_position` | `team_avg_finish_last3`: previous 3 races |
| `var_delta_i_norm` | `circuit_pos_change_var_hist`: average of that measure over earlier races at the circuit |
| teammate head-to-head | Counted cumulatively up to the previous race |
| `circuit_wet_race_frequency` | Earlier races at the circuit only |
| race temperatures | Dropped by default (`"race_weather": true` in a config adds them back) |

Qualifying results and the starting grid are allowed: they are known before the race starts. A data fix was also made: FastF1 reports a pit-lane start as grid position 0, which looked like starting ahead of pole; those drivers now start last, with a `pit_lane_start` flag.

**Guard against regressions.** [`tests/test_features_no_leak.py`](../../tests/test_features_no_leak.py) scrambles the results of one race (reverses the finishing order, changes points, status, rain and temperatures) and asserts that no feature of that race or any earlier race changes. When the one-race shift is removed, the test fails.

## 3. Validation protocol

- **Train** 2018–2024, **validate** on 2025. Every choice (features, model type, hyperparameters) was made on 2025 only.
- **Test** on 2026 (rounds 1–14), once, after the choice was frozen in [`configs/final.json`](../../configs/final.json). [`evals/final_test.py`](../final_test.py) refuses to run if that file has uncommitted changes.
- The grid-position baseline is reported in every table.
- Metrics are computed per race and averaged over races: Spearman rank correlation, top-10 overlap, mean absolute error of the predicted finishing rank, and NDCG@10.

One caveat: 2026 was evaluated once before this protocol, during the audit above, using the old feature set. None of the later choices used 2026.

## 4. What was tried (validation 2025)

From [`experiments.md`](experiments.md) (`python evals/experiments.py`). The rule was fixed before running: pick the highest mean Spearman on 2025.

| Step | Change | Spearman (2025) |
|---|---|---|
| — | Grid-position baseline | 0.651 |
| E1 | Leak-free features, XGBoost regression | 0.615 |
| E2 | XGBoost on finish minus grid | 0.627 |
| E3 | LightGBM ranking model (lambdarank, one group per race) | 0.628 |
| E4–E8 | Added qualifying position and gaps, driver form, DNF rates, circuit history (pit-lane flag tried and dropped) | 0.643 |
| H18 | Best of 20 random hyperparameter settings | 0.668 |

H18 is the best of 28 configurations scored on the same 24 races, so its 2025 number is optimistic by construction. The 2026 result is the honest one.

## 5. Final result (2026, evaluated once)

From [`final_test_2026.md`](final_test_2026.md) (`python evals/final_test.py`): trained on 2018–2025, tested on 14 races of 2026.

| Model | Spearman | Top-10 overlap | MAE (places) | NDCG@10 | Races beating baseline |
|---|---|---|---|---|---|
| Grid-position baseline | 0.650 ± 0.182 | 0.750 | 3.403 | 0.855 | — |
| Final model (LightGBM lambdarank, 28 pre-race features) | 0.667 ± 0.183 | 0.764 | 3.461 | 0.855 | 7 of 14 |

From [`paired_test_2026.md`](paired_test_2026.md) (`python evals/paired_test.py`), comparing the two race by race: the mean Spearman difference is +0.016, with a 95% bootstrap confidence interval of −0.018 to +0.055 and a Wilcoxon signed-rank p-value of 0.626. The interval includes zero, so the model **matches** the grid-position baseline; it does not measurably beat it.

## 6. What this means

The starting grid already reflects most of what can be known before a race: qualifying pace, car performance and penalties. The features that made the original model look much better than that were reading the answer. The honest conclusion is that this model, with these pre-race inputs, is as good as a strong baseline, and the numbers above show exactly how that was measured.

*Note on MAE:* [`backtest_before.md`](backtest_before.md) reports MAE on the model's raw predicted position; later files report MAE on the predicted finishing rank, so MAE values are not comparable across those files.
