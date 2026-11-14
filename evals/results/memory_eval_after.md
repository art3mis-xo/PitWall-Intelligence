# Memory evaluation (after)

Date 2026-11-14T12:44:07+00:00 · commit `b6c6ce0eece01a65adcdd01153de6a755a519c55` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/memory_eval.py --tag after`

8 two-turn conversations (seed 21) whose second question depends on the first, plus one 10-turn conversation for growth. History is passed exactly as the live chat passes it.

| Follow-up turn | Result |
|---|---|
| Right tool (where one is needed) | 8/8 |
| Right arguments | 6/6 |
| Correct | 8/8 |
| Grounded | 7/8 |
| Errors | 0 |

| Turn (10-turn conversation) | Input tokens | Seconds |
|---|---|---|
| 1 | 1711 | 11.2 |
| 2 | 1827 | 5.81 |
| 3 | 2339 | 10.31 |
| 4 | 2275 | 15.69 |
| 5 | 2393 | 3.4 |
| 6 | 2675 | 14.68 |
| 7 | 2781 | 12.62 |
| 8 | 3157 | 13.34 |
| 9 | 1592 | 2.74 |
| 10 | 1729 | 11.68 |

## Failed follow-ups

- driver_followup: “And how many places does he gain or lose per race on average?” → Lance Stroll averages **–1.37 places per race** (i.e., he gains about 1.4 positions from grid to finish on average).
