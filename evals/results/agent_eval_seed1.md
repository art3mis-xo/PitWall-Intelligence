# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-14T12:24:51+00:00 · commit `559555e92a553e2591fc41fc5f4e7a6c0d133e65` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 1`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 1). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 24/24 (100%) |
| Right arguments (season, round, driver) | 22/22 (100%) |
| Correct answer | 22/24 (92%) |
| Every number grounded in tool output (model's answer) | 24/24 (100%) |
| Replies with no unverified number shown unflagged (after output guardrail) | 24/24 (100%) |
| Latency, median / p90 | 7.36 s / 11.18 s |
| Tokens per question, median | 1577 |
| Tool calls per question, mean / max | 0.96 / 2 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) | 6/6 (100%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **explain** — Why is Pierre Gasly predicted to finish P10 in round 2 of 2026? → incorrect. Reply: “Pierre Gasly is projected to finish **P10** in round 2 of 2026 mainly because the model weighs his recent performance and the current team’s form more heavily t”
- **explain** — Why is Nico Hulkenberg predicted to finish P13 in round 4 of 2026? → incorrect. Reply: “Nico Hulkenberg is projected to finish **P13** in round 4 of 2026, starting from **P10**.   The model’s SHAP‑based explanation shows several factors pulling his”

**Reading the failures:** explain "incorrect" rows are strict-label misses (the reply paraphrases the top factor's label); they are kept as failures rather than loosening the grader after seeing results.
