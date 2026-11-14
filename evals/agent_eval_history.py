"""
Compare agent-eval runs across agent versions, from the saved results in
evals/results/agent_eval_history/<version>/seed<N>.json (one folder per version,
in the order the changes were made). Recomputes every metric from the stored cases.

    python evals/agent_eval_history.py
Writes evals/results/agent_eval_history.md
"""
import json
import statistics
from pathlib import Path

HISTORY = Path(__file__).resolve().parent / "results" / "agent_eval_history"


def pct(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "—"


def main():
    lines = ["# Agent eval across versions", "",
             "Each row is one saved run of `evals/agent_eval.py`. Seed 0 is the question set used while fixing the "
             "agent; seed 1 was never used for changes. v1 was graded with grader v1 (see its note); later runs with v2.", "",
             "| Version | Questions | Right tool | Right args | Correct | Grounded | Latency p50 / p90 (s) | Tool calls mean / max |",
             "|---|---|---|---|---|---|---|---|"]
    for version in sorted(p for p in HISTORY.iterdir() if p.is_dir()):
        for run in sorted(version.glob("seed*.json")):
            rows = [r for r in json.loads(run.read_text())["cases"] if "error" not in r]
            lat = sorted(r["latency_s"] for r in rows)
            calls = [len(r["tool_calls"]) for r in rows]
            lines.append(f"| {version.name} ({run.stem}) | {len(rows)} | {pct(rows, 'tool_ok')} | {pct(rows, 'args_ok')} | "
                         f"{pct(rows, 'correct')} | {pct(rows, 'grounded')} | "
                         f"{statistics.median(lat):.1f} / {lat[int(0.9 * (len(lat) - 1))]:.1f} | "
                         f"{statistics.mean(calls):.2f} / {max(calls)} |")
    out = HISTORY.parent / "agent_eval_history.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
