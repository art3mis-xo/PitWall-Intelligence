# Agent evaluation (live LangGraph agent on Groq)

Date 2026-11-14T08:59:50+00:00 · commit `74d06c10222e289dec8d26ab24e6df6969b6922a` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py --seed 0`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 0). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 24/24 (100%) |
| Right arguments (season, round, driver) | 22/22 (100%) |
| Correct answer | 22/24 (92%) |
| Every number grounded in tool output | 24/24 (100%) |
| Latency, median / p90 | 5.57 s / 16.74 s |
| Tokens per question, median | 1628 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) | 9/9 (100%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 5/6 (83%) | 6/6 (100%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |
| out_of_scope | 2 | 2/2 (100%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **explain** — Why is Oscar Piastri predicted to finish P8 in round 9 of 2026? → incorrect. Reply: “Oscar Piastri is projected to finish **P8** in round 9 of 2026 largely because his starting grid position and the recent performance of his team and himself lin”
- **ambiguous** — What is Kimi's teammate head-to-head win percentage? → incorrect. Reply: “Kimi Räikkönen’s teammate head‑to‑head win percentage is **61.5 %** (based on 79 races up to 2021 R22).”

**Reading the failures:** explain "incorrect" rows are strict-label misses (the reply paraphrases the top factor's label, e.g. "Team average finish" for "Team avg finish"); they are kept as failures rather than loosening the grader after seeing results. Any "ungrounded numbers" row is a number in the reply that no tool returned.
