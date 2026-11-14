# Agent latency benchmark

Date 2026-11-14T09:35:52+00:00 · commit `069966e67d590abeea96476d90e263f37dd44750` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/latency_bench.py`

6 questions (stats, prediction, explanation) x 3 runs per setting, settings interleaved. Latency is end to end (all model and tool calls). Answer quality per setting: see agent_eval_history.md.

| Setting | Runs | p50 (s) | p90 (s) | LLM calls / question | Input tokens | Output tokens | Answered |
|---|---|---|---|---|---|---|---|
| default reasoning | 18 | 6.87 | 13.7 | 2.33 | 1900 | 287 | 18/18 |
| reasoning_effort=low | 18 | 3.08 | 10.7 | 2 | 1533 | 171 | 18/18 |
