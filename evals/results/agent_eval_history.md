# Agent eval across versions

Each row is one saved run of `evals/agent_eval.py`. Seed 0 is the question set used while fixing the agent; seed 1 was never used for changes. v1 was graded with grader v1 (see its note); later runs with v2. Latency here is one run per version and swings with Groq load (v3 to v4 changed nothing latency-related); see evals/latency_bench.py for repeated measurements.

| Version | Questions | Right tool | Right args | Correct | Grounded | Latency p50 / p90 (s) | Tool calls mean / max |
|---|---|---|---|---|---|---|---|
| v1-baseline (seed0) | 24 | 75% | 94% | 75% | 88% | 7.2 / 20.0 | 1.12 / 4 |
| v3-empty-reply-retry (seed0) | 24 | 100% | 100% | 92% | 100% | 5.6 / 16.7 | 1.17 / 2 |
| v3-empty-reply-retry (seed1) | 24 | 100% | 100% | 92% | 96% | 4.6 / 11.2 | 1.17 / 2 |
| v4-tool-budget (seed0) | 24 | 100% | 100% | 96% | 100% | 1.1 / 8.2 | 1.17 / 2 |
| v4-tool-budget (seed1) | 24 | 100% | 100% | 92% | 96% | 3.2 / 10.5 | 1.17 / 2 |
