# Paraphrase robustness (live agent)

Date 2026-11-14T12:29:00+00:00 · commit `559555e92a553e2591fc41fc5f4e7a6c0d133e65` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/robustness_eval.py`

8 questions, each asked in 4 wordings (seed 11); answers computed from the data, scored with agent_eval.py's rules.

| Metric | Result |
|---|---|
| Accuracy, all wordings | 32/32 |
| Consistency (item right in every wording) | 8/8 |
| Accuracy by wording (1-4) | 8/8 · 8/8 · 8/8 · 8/8 |
| Errors | 0 |

## Inconsistent items

