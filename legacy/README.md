# Legacy

Earlier experiments kept for reference. Nothing in the app, pipeline, tests or evals imports or reads these files.

| File | What it was |
|---|---|
| `Build_dataset_lapwise.py` | Builds a lap-level dataset (2018–2022); the model never used it. |
| `Feature_Engineering.ipynb` | Notebook version of `2Feature_engineering.py`, superseded by the script. |
| `Model_training.ipynb` | First training notebook (fixed split, RF/LightGBM/grid search), superseded by `3Model_training.ipynb`. |
| `gbm_f1_ranker.txt` | LightGBM lambdarank model on a different feature set; the code that trained it is not in the repo. |
| `used_cols.csv` | Feature list for `gbm_f1_ranker.txt`. |
| `feature_importance.png` | Gain importances for the `gbm_f1_ranker.txt` feature set. |
| `predictions_S2026_R6.csv` | 2026 Round 6 prediction export from the ranker (`pred_score` column). |
| `eval_models.py` | Eval stubs for a tyre-degradation model, winner classifier and pit optimizer, none of which were built. |
| `PLAN.md` | Original week-by-week build plan. |
| `miscellaneous/` | Notebooks and a script from first exploring FastF1. |
