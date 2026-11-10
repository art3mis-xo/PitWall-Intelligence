# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-10T05:10:35+00:00 · commit `39f83d124be46fb43b852162e7dc6d98faaad939` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 0`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 0). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 23/24 (96%) |
| Right arguments (season, round, driver) | 21/21 (100%) |
| Correct answer | 21/24 (88%) |
| Every number grounded in tool output | 24/24 (100%) |
| Latency, median / p90 | 9.82 s / 18.47 s |
| Tokens per question, median | 1662 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 8/9 (89%) | 8/8 (100%) | 8/9 (89%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) | 6/6 (100%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **stats** — What is ALO's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “”
- **explain** — Why is Lewis Hamilton predicted to finish P6 in round 4 of 2026? → incorrect. Reply: “Lewis Hamilton is projected to finish **P6** in the 2026 Round 4 race because the model’s key drivers all point to a mid‑field result:  | Feature | Value | SHAP”
- **explain** — Why is Oscar Piastri predicted to finish P8 in round 9 of 2026? → incorrect. Reply: “Oscar Piastri is projected to finish **P8** in round 9 of 2026 because the model balances a few key factors that, together, keep him near his starting spot.  | ”

**Reading the failures:** the explain "incorrect" rows are strict-label misses (the reply says "Team average finish" for the label "Team avg finish"); they are kept as failures rather than loosening the grader after seeing results. The empty replies to code-only questions ("What is HAD's ...") are real: the agent called no tool and returned nothing.
