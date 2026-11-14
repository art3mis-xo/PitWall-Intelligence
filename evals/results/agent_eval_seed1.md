# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-14T09:02:58+00:00 · commit `74d06c10222e289dec8d26ab24e6df6969b6922a` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 1`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 1). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 24/24 (100%) |
| Right arguments (season, round, driver) | 22/22 (100%) |
| Correct answer | 22/24 (92%) |
| Every number grounded in tool output | 23/24 (96%) |
| Latency, median / p90 | 4.61 s / 11.18 s |
| Tokens per question, median | 1637 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) | 5/6 (83%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **explain** — Why is Pierre Gasly predicted to finish P10 in round 2 of 2026? → incorrect. Reply: “Pierre Gasly is projected to finish 10th in the 2026 Round 2 because the model weighs several recent performance indicators that collectively push his expected ”
- **explain** — Why is Nico Hulkenberg predicted to finish P13 in round 4 of 2026? → incorrect. Reply: “Nico Hülkenberg is projected to finish 13th in the 2026 Round 4 because several key factors pull his predicted result down:  | Feature | Value | SHAP contributi”
- **explain** — Why is Nico Hulkenberg predicted to finish P11 in round 11 of 2026? → ungrounded numbers [13.0]. Reply: “Nico Hülkenberg is projected to finish **P11** in round 11 of 2026 mainly because his starting position and recent performance metrics are not favourable for a ”

**Reading the failures:** explain "incorrect" rows are strict-label misses (the reply paraphrases the top factor's label, e.g. "Team average finish" for "Team avg finish"); they are kept as failures rather than loosening the grader after seeing results. Any "ungrounded numbers" row is a number in the reply that no tool returned.
