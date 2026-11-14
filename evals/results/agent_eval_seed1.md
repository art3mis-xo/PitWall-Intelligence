# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-14T12:47:12+00:00 · commit `b6c6ce0eece01a65adcdd01153de6a755a519c55` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 1`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 1). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 24/24 (100%) |
| Right arguments (season, round, driver) | 22/22 (100%) |
| Correct answer | 22/24 (92%) |
| Every number grounded in tool output (model's answer) | 22/24 (92%) |
| Replies with no unverified number shown unflagged (after output guardrail) | 24/24 (100%) |
| Latency, median / p90 | 5.66 s / 9.14 s |
| Tokens per question, median | 1726 |
| Tool calls per question, mean / max | 0.92 / 1 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) | 4/6 (67%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **explain** — Why is Pierre Gasly predicted to finish P10 in round 2 of 2026? → incorrect. Reply: “Pierre Gasly (GAS) is projected to finish **P10** in round 2 of 2026 mainly because of a combination of recent performance trends and his starting position.  | ”
- **explain** — Why is Nico Hulkenberg predicted to finish P13 in round 4 of 2026? → incorrect, ungrounded numbers [14.0]. Reply: “Nico Hulkenberg (HUL) is projected to finish **P13** in round 4 of 2026 for several quantifiable reasons:  | Feature | Driver’s value | SHAP contribution | Effe”
- **explain** — Why is Nico Hulkenberg predicted to finish P11 in round 11 of 2026? → ungrounded numbers [13.0]. Reply: “Nico Hulkenberg (HUL) is projected to finish **P11** in round 11 of 2026 mainly because of a combination of his starting position, recent performance, and the c”

**Reading the failures:** the ungrounded rows here are approximations of real values (a team average of about 13.4 described as "around 13th"); the grader counts rounding beyond ±0.05 as ungrounded, and the output guardrail shows users a warning in these cases.
