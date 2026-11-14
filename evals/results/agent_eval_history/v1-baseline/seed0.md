# Agent evaluation — baseline (before fixes)

Date 2026-11-10T05:03:29+00:00 · commit `b361b586b00778626aed40e10c55f1e74defbf53` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/agent_eval.py`

Model `openai/gpt-oss-20b`, temperature 0, 24 questions generated from the data (seed 0). Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.

| Metric | Result |
|---|---|
| Right tool chosen | 18/24 (75%) |
| Right arguments (season, round, driver) | 16/17 (94%) |
| Correct answer | 18/24 (75%) |
| Every number grounded in tool output | 21/24 (88%) |
| Latency, median / p90 | 7.21 s / 19.98 s |
| Tokens per question, median | 1700 |
| Errors | 0 |

| Kind | Questions | Tool | Args | Correct | Grounded |
|---|---|---|---|---|---|
| stats | 9 | 4/9 (44%) | 3/4 (75%) | 4/9 (44%) | 8/9 (89%) |
| predict | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| explain | 6 | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 4/6 (67%) |
| ambiguous | 1 | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |
| out_of_scope | 2 | 1/2 (50%) | — | 2/2 (100%) | 2/2 (100%) |

## Failures

- **stats** — What is Isack Hadjar's teammate head-to-head win percentage? → wrong tool, incorrect, ungrounded numbers [1.0, 1.0, 1.0]. Reply: “Isack Hadjar has never competed in Formula 1, so he has no F1 teammate to compare against. Consequently, a head‑to‑head win percentage for him in F1 is not appl”
- **stats** — What is Hamilton's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “Could you let me know which season (and if relevant, which specific race) you’re interested in? That way I can pull the correct teammate data and calculate the ”
- **stats** — What is Arvid Lindblad's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “”
- **stats** — What is Ocon's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “I’m not sure which season you’re referring to. Could you let me know the year (or round) you’d like the teammate win‑percentage for?”
- **stats** — What is PER's teammate head-to-head win percentage? → wrong tool, incorrect. Reply: “”
- **explain** — Why is Max Verstappen predicted to finish P8 in round 3 of 2026? → ungrounded numbers [0.228, 0.131, 0.131, 0.228]. Reply: “Max Verstappen is projected to finish **P8** in Round 3 of 2026, despite starting from **P11** on the grid.   The model’s SHAP‑based explanation shows a mix of ”
- **explain** — Why is Oscar Piastri predicted to finish P8 in round 9 of 2026? → ungrounded numbers [0.213, 0.133, 0.127]. Reply: “Oscar Piastri is projected to finish **P8** in round 9 of 2026, starting from **P8** on the grid. The model’s SHAP‑grounded explanation shows a balance of facto”
- **stats** — How does Carlos do against his teammate head-to-head, as a win percentage? → wrong args. Reply: “Carlos Sainz’s head‑to‑head win percentage against his current teammate is **45.9 %** (based on 186 races up to 2026 R14).”
- **ambiguous** — What is Kimi's teammate head-to-head win percentage? → incorrect. Reply: “Kimi Räikkönen’s teammate head‑to‑head win percentage is **61.5 %** (based on 79 races up to 2021 R22).”
- **out_of_scope** — What was the fastest lap at the 1995 Monaco Grand Prix? → wrong tool. Reply: “”

**Grader note (added before the fixes):** this run used grader v1, which read the non-breaking hyphen (U+2011) as no sign and the "1" in "Formula 1" as a statistic. Its 3 grounding failures (#2, #15, #19) were grader errors, verified against the transcripts; corrected in grader v2. Tool, argument and correctness scores are unaffected.
