"""
Automated evaluation of the PitWall chat agent (LangGraph + Groq), the one the app serves.

No hand-written labels: every expected answer is computed by code from the same
sources the agent's tools read (data/features.csv and the predictor). The eval
therefore measures whether the agent picks the right tool, passes the right
arguments and reports the tools' numbers faithfully. It does not measure whether
the race predictions themselves are right (see evals/final_test.py for that).

Scored per question, all automatically (no LLM judge):
  tool        the expected tool was called (cases that need no tool: no tool called)
  args        that call used the right season / round / driver
  correct     the expected answer appears in the reply
  grounded    every number in the reply appears in a tool output or the question

    python evals/agent_eval.py [--seed N]   # needs GROQ_API_KEY; 24 questions per seed
Writes evals/results/agent_eval.json and .md
"""
import random
import statistics
import sys
import time
import unicodedata

from common import ROOT, header, meta, write

sys.path.insert(0, str(ROOT))
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from pitwall.backend.services import agent  # noqa: E402
from pitwall.backend.services.race_predictor import load_features, predict_race  # noqa: E402

SEASON = 2026
PAUSE_S = 2.0          # stay under Groq's per-minute limits
# Number parsing and grounding are shared with the app's output guardrail
from pitwall.backend.services.guardrails import matches as _matches, numbers, ungrounded  # noqa: E402


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


# ── cases: questions plus answers computed from the data ─────────────────────
def build_cases(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    df = load_features()
    season = df[df.season == SEASON]
    names = season.drop_duplicates("driver_id").set_index("driver_id")["driver_full_name"]
    drivers = sorted(rng.sample(sorted(names.index), 8))
    rounds = sorted(rng.sample(sorted(season["round"].unique()), 6))
    cases = []

    for i, code in enumerate(drivers):   # stats tool, asked by code, full name or surname
        latest = df[df.driver_id == code].sort_values(["season", "round"]).iloc[-1]
        who = [code, names[code], names[code].split()[-1]][i % 3]
        cases.append({"kind": "stats", "question": f"What is {who}'s teammate head-to-head win percentage?",
                      "tool": "features_for_stats_tool", "driver": code,
                      "expect_number": round(100 * float(latest["teammate_win_pct"]), 1)})

    for rnd in rounds:                   # prediction tool
        winner = predict_race(SEASON, int(rnd))[0]
        cases.append({"kind": "predict", "question": f"Who does the model predict to win round {rnd} of the {SEASON} season?",
                      "tool": "race_predictor_tool", "season": SEASON, "round": int(rnd),
                      "expect_text": [winner["driver_full_name"].split()[-1]]})

    for rnd in rounds:                   # explanation tool, a driver from the middle of the order
        r = rng.choice(predict_race(SEASON, int(rnd))[3:15])
        cases.append({"kind": "explain", "question": f"Why is {r['driver_full_name']} predicted to finish "
                      f"P{r['predicted_rank']} in round {rnd} of {SEASON}?",
                      "tool": "explain_prediction_tool", "season": SEASON, "round": int(rnd), "driver": r["driver_id"],
                      "expect_text": [r["top_factors"][0]["label"].split(" (")[0]]})

    sai = df[df.driver_id == "SAI"].sort_values(["season", "round"]).iloc[-1]
    cases += [
        {"kind": "stats", "question": "How does Carlos do against his teammate head-to-head, as a win percentage?",
         "tool": "features_for_stats_tool", "driver": "SAI",   # first name only: still one driver in the data
         "expect_number": round(100 * float(sai["teammate_win_pct"]), 1)},
        {"kind": "ambiguous", "question": "What is Kimi's teammate head-to-head win percentage?",   # two Kimis
         "tool": "features_for_stats_tool", "expect_text": ["Raikkonen", "Antonelli"], "expect_no_value": True},
    ]
    cases += [                           # outside what the tools know: must not invent numbers
        {"kind": "out_of_scope", "question": "What was the fastest lap at the 1995 Monaco Grand Prix?", "tool": None},
        {"kind": "out_of_scope", "question": f"What will the air temperature be on race day in São Paulo in {SEASON}?",
         "tool": None},
    ]
    return cases


# ── run and score ────────────────────────────────────────────────────────────
def run_case(case: dict) -> dict:
    start = time.perf_counter()
    for attempt in range(4):
        try:
            out = agent.app.invoke({"messages": [HumanMessage(case["question"])]}, {"recursion_limit": 10})
            break
        except Exception as e:   # rate limits and transient Groq errors
            if "tokens per day" in str(e):   # a daily quota doesn't come back by waiting minutes
                sys.exit("Groq daily token limit reached; no results written. Retry after the quota resets "
                         "or set GROQ_API_KEY to a separate eval key.")
            if attempt == 3:
                return {**case, "error": str(e)[:200]}
            time.sleep(10 * (attempt + 1))
    latency = time.perf_counter() - start

    msgs = out["messages"]
    calls = [c for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    tool_text = "\n".join(str(m.content) for m in msgs if isinstance(m, ToolMessage))
    last = msgs[-1]
    flagged = last.additional_kwargs.get("guardrail_flagged") if isinstance(last, AIMessage) else None
    reply = str(last.content)                                  # what the user sees
    answer = str(msgs[-2].content) if flagged else reply       # what the model wrote, before the guard
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in msgs if isinstance(m, AIMessage))

    expected = case["tool"]
    used = [c for c in calls if c["name"] == expected]
    tool_ok = (not calls) if expected is None else bool(used)

    args_ok = None
    if used and expected:
        a = used[0]["args"]
        args_ok = all([
            "season" not in case or int(a.get("season", -1)) == case["season"],
            "round" not in case or int(a.get("round_num", -1)) == case["round"],
            "driver" not in case or agent.resolve_driver(str(a.get("driver_id", "")))[0] == case["driver"],
        ])

    folded = fold(answer)
    correct = all(fold(t) in folded for t in case.get("expect_text", []))
    if "expect_number" in case:
        correct = any(_matches(n, case["expect_number"]) for n in numbers(answer))
    if case.get("expect_no_value"):   # ambiguous: naming both drivers is right, quoting a value is not
        correct = correct and not numbers(answer)

    missing = ungrounded(answer, tool_text + "\n" + case["question"])
    unflagged = [n for n in missing if n not in (flagged or [])]
    if case["kind"] == "out_of_scope":   # right = says it can't answer, without inventing numbers
        correct = not missing
    return {**case, "reply": reply, "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
            "tool_output": tool_text,
            "tool_ok": tool_ok, "args_ok": args_ok, "correct": correct,
            "grounded": not missing, "ungrounded_numbers": missing,
            "guard_flagged": bool(flagged), "shown_unflagged": not unflagged,
            "latency_s": round(latency, 2), "tokens": tokens}


def rate(rows, key) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else "—"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="question set; 0 was used while fixing the agent")
    seed = parser.parse_args().seed
    cases = build_cases(seed)
    rows = []
    for i, case in enumerate(cases, 1):
        rows.append(run_case(case))
        r = rows[-1]
        print(f"{i:2}/{len(cases)} {case['kind']:12} tool={r.get('tool_ok')} args={r.get('args_ok')} "
              f"correct={r.get('correct')} grounded={r.get('grounded')} {r.get('latency_s', '')}s")
        time.sleep(PAUSE_S)

    ok = [r for r in rows if "error" not in r]
    lat = sorted(r["latency_s"] for r in ok)
    summary = {"cases": len(rows), "errors": len(rows) - len(ok),
               "tool_selection": rate(ok, "tool_ok"), "arguments": rate(ok, "args_ok"),
               "correct": rate(ok, "correct"), "grounded": rate(ok, "grounded"),
               "no_unflagged_ungrounded": rate(ok, "shown_unflagged"),
               "latency_median_s": round(statistics.median(lat), 2),
               "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2),
               "tokens_median": int(statistics.median(r["tokens"] for r in ok)),
               "tool_calls_mean": round(statistics.mean(len(r["tool_calls"]) for r in ok), 2),
               "tool_calls_max": max(len(r["tool_calls"]) for r in ok)}

    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "features.csv")
    m = meta(f"python evals/agent_eval.py --seed {seed}", df, agent_model=agent.llm.model_name, seed=seed,
             labels="expected answers computed by code from data/features.csv and the predictor; no hand labels")
    lines = header("Agent evaluation (live LangGraph agent on Groq)", m) + [
        f"Model `{m['agent_model']}`, temperature 0, {summary['cases']} questions generated from the data (seed {seed}). "
        "Expected answers are computed by code from the feature store and the predictor; scoring is rule-based, no LLM judge.",
        "",
        "| Metric | Result |", "|---|---|",
        f"| Right tool chosen | {summary['tool_selection']} |",
        f"| Right arguments (season, round, driver) | {summary['arguments']} |",
        f"| Correct answer | {summary['correct']} |",
        f"| Every number grounded in tool output (model's answer) | {summary['grounded']} |",
        f"| Replies with no unverified number shown unflagged (after output guardrail) | {summary['no_unflagged_ungrounded']} |",
        f"| Latency, median / p90 | {summary['latency_median_s']} s / {summary['latency_p90_s']} s |",
        f"| Tokens per question, median | {summary['tokens_median']} |",
        f"| Tool calls per question, mean / max | {summary['tool_calls_mean']} / {summary['tool_calls_max']} |",
        f"| Errors | {summary['errors']} |",
        "", "| Kind | Questions | Tool | Args | Correct | Grounded |", "|---|---|---|---|---|---|",
    ]
    for kind in dict.fromkeys(r["kind"] for r in rows):
        k = [r for r in ok if r["kind"] == kind]
        lines.append(f"| {kind} | {len(k)} | {rate(k, 'tool_ok')} | {rate(k, 'args_ok')} | {rate(k, 'correct')} | {rate(k, 'grounded')} |")
    fails = [r for r in ok if not (r["tool_ok"] and r["correct"] and r["grounded"] and r["args_ok"] is not False)]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            why = [w for w, bad in [("wrong tool", not r["tool_ok"]), ("wrong args", r["args_ok"] is False),
                                    ("incorrect", not r["correct"]),
                                    (f"ungrounded numbers {r['ungrounded_numbers']}", not r["grounded"])] if bad]
            lines.append(f"- **{r['kind']}** — {r['question']} → {', '.join(why)}. Reply: “{r['reply'][:160].replace(chr(10), ' ')}”")
    write("agent_eval" if seed == 0 else f"agent_eval_seed{seed}", {"meta": m, "summary": summary, "cases": rows}, lines)


if __name__ == "__main__":
    main()
