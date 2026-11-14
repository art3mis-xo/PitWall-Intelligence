# Memory evaluation (before)

Date 2026-11-14T12:39:06+00:00 · commit `7e2d52e1eb5bbf5f74f0813cc56b4a1af409626d` · xgboost 3.3.0, lightgbm 4.6.0 · 3766 rows, 187 races (2018-2026) · `python evals/memory_eval.py --tag before`

8 two-turn conversations (seed 21) whose second question depends on the first, plus one 10-turn conversation for growth. History is passed exactly as the live chat passes it.

| Follow-up turn | Result |
|---|---|
| Right tool (where one is needed) | 8/8 |
| Right arguments | 5/6 |
| Correct | 7/8 |
| Grounded | 7/8 |
| Errors | 0 |

| Turn (10-turn conversation) | Input tokens | Seconds |
|---|---|---|
| 1 | 1555 | 3.45 |
| 2 | 1481 | 1.16 |
| 3 | 2997 | 12.22 |
| 4 | 2185 | 9.42 |
| 5 | 2301 | 13.58 |
| 6 | 2570 | 14.64 |
| 7 | 2693 | 13.73 |
| 8 | 3057 | 20.88 |
| 9 | 1539 | 10.85 |
| 10 | 1682 | 7.85 |

## Failed follow-ups

- driver_followup: “And how many places does he gain or lose per race on average?” → Lance Stroll averages **–1.37 places per race** (i.e., he gains about 1.4 positions from grid to finish on average).
- why_winner: “Why is that driver predicted to win?” → I’m sorry, but I couldn’t find a driver named “KIM” in the dataset. Could you clarify which driver you’re referring to?
