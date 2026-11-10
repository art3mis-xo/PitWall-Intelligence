# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-10T05:15:13+00:00 · commit `39f83d124be46fb43b852162e7dc6d98faaad939` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 1`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 1). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 23/24 (96%) |
| Right arguments (season, round, driver) | 21/21 (100%) |
| Correct answer | 21/24 (88%) |
| Every number grounded in tool output | 24/24 (100%) |
| Latency, median / p90 | 9.3 s / 19.26 s |
| Tokens per question, median | 1660 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 8/9 (89%) | 8/8 (100%) | 8/9 (89%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) | 6/6 (100%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **stats** — What is HAD's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “”
- **explain** — Why is Pierre Gasly predicted to finish P10 in round 2 of 2026? → incorrect. Reply: “Pierre Gasly is projected to finish 10th in the 2026 Round 2 because the model weighs several factors that, together, pull his expected result down:  | Feature ”
- **explain** — Why is Nico Hulkenberg predicted to finish P13 in round 4 of 2026? → incorrect. Reply: “Nico Hülkenberg is projected to finish 13th in the 2026 Round 4 race mainly because several key performance indicators are working against him:  | Feature | Val”

**Reading the failures:** the explain "incorrect" rows are strict-label misses (the reply says "Team average finish" for the label "Team avg finish"); they are kept as failures rather than loosening the grader after seeing results. The empty replies to code-only questions ("What is HAD's ...") are real: the agent called no tool and returned nothing.
