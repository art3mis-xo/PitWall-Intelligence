"""
Multi-turn (memory) evaluation of the chat agent, through the same history handling as the live
chat (agent.build_messages: earlier turns arrive as text from the browser).

1. Follow-ups: 8 two-turn conversations whose second question only makes sense with the first
   ("Why is that driver predicted to win?"). Expected answers computed from the data; scored by
   rules as in agent_eval.py: right tool and arguments where a tool is needed, correct, grounded.
2. Growth: one 10-turn conversation; input tokens and latency per turn.

    python evals/memory_eval.py [--tag NAME]    # needs GROQ_API_KEY (GROQ_EVAL_API_KEY if set)
Writes evals/results/memory_eval[_NAME].json and .md
"""
import argparse
import random
import sys
import time

from common import ROOT, header, meta, write

sys.path.insert(0, str(ROOT))
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from pitwall.backend.services import agent  # noqa: E402
from pitwall.backend.services.guardrails import matches, numbers, ungrounded  # noqa: E402
from pitwall.backend.services.race_predictor import load_features, predict_race  # noqa: E402

SEED, SEASON = 21, 2026
LONG_CONVERSATION = [
    "Who does the model predict to win round 9 of 2026?", "Why is that driver predicted to win?",
    "Who is predicted second?", "What is Lando Norris's teammate head-to-head win percentage?",
    "And how many places does he gain or lose per race on average?",
    "Who does the model predict to win round 13 of 2026?", "Why?",
    "What is Verstappen's teammate head-to-head win percentage?", "Compare that with Norris.",
    "Summarise what we discussed in two sentences.",
]


def conversations() -> list[dict]:
    rng = random.Random(SEED)
    df = load_features()
    season = df[df.season == SEASON]
    names = season.drop_duplicates("driver_id").set_index("driver_id")["driver_full_name"]
    convs = []
    for code in rng.sample(sorted(names.index), 4):          # pronoun follow-up on a driver
        delta = float(df[df.driver_id == code].sort_values(["season", "round"]).iloc[-1]["avg_finish_grid_delta"])
        convs.append({"kind": "driver_followup",
                      "turns": [f"What is {names[code]}'s teammate head-to-head win percentage?",
                                "And how many places does he gain or lose per race on average?"],
                      "tool": "features_for_stats_tool", "driver": code, "expect_abs_number": round(abs(delta), 3)})
    for rnd in rng.sample(sorted(season["round"].unique()), 4):
        order = predict_race(SEASON, int(rnd))
        first = [f"Who does the model predict to win round {rnd} of {SEASON}?"]
        if len(convs) < 6:                                   # "that driver" -> explanation
            convs.append({"kind": "why_winner", "turns": first + ["Why is that driver predicted to win?"],
                          "tool": "explain_prediction_tool", "driver": order[0]["driver_id"],
                          "season": SEASON, "round": int(rnd)})
        else:                                               # answerable from turn 1's tool output
            convs.append({"kind": "second_place", "turns": first + ["Who is predicted to finish second?"],
                          "tool": None, "expect_text": order[1]["driver_full_name"].split()[-1]})
    return convs


def ask(question: str, history: list[dict]) -> dict:
    start = time.perf_counter()
    out = agent.app.invoke({"messages": agent.build_messages(question, history)}, {"recursion_limit": 12})
    seconds = time.perf_counter() - start
    new = out["messages"][len(history) + 1:]                 # this turn's messages only
    calls = [c for m in new if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    llm = [m for m in new if isinstance(m, AIMessage) and m.usage_metadata]
    last = out["messages"][-1]
    flagged = last.additional_kwargs.get("guardrail_flagged")
    answer = str(out["messages"][-2].content) if flagged else str(last.content)
    sources = "\n".join(str(m.content) for m in out["messages"]
                        if isinstance(m, ToolMessage) or m.type == "human" or (m.type == "ai" and m in out["messages"][:len(history) + 1]))
    return {"question": question, "answer": answer, "reply": str(last.content), "calls": calls,
            "seconds": round(seconds, 2), "input_tokens": sum(m.usage_metadata.get("input_tokens", 0) for m in llm),
            "grounded": not ungrounded(answer, sources)}


def turn(question, history):
    for attempt in range(3):
        try:
            return ask(question, history)
        except Exception as e:
            if "tokens per day" in str(e):
                sys.exit("Groq daily token limit reached; no results written.")
            time.sleep(10 * (attempt + 1))
    return {"question": question, "error": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="suffix for the results file, e.g. before/after a change")
    tag = parser.parse_args().tag

    rows = []
    for conv in conversations():
        history = []
        t1 = turn(conv["turns"][0], history)
        history += [{"role": "user", "content": conv["turns"][0]}, {"role": "assistant", "content": t1.get("reply", "")}]
        t2 = turn(conv["turns"][1], history)
        if "error" in t2:
            rows.append({**conv, "error": True}); continue
        used = [c for c in t2["calls"] if c["name"] == conv["tool"]]
        tool_ok = True if conv["tool"] is None else bool(used)   # second place: answerable without a tool
        args_ok = None
        if used:
            a = used[0]["args"]
            args_ok = all([agent.resolve_driver(str(a.get("driver_id", "")))[0] == conv["driver"],
                           "season" not in conv or int(a.get("season", -1)) == conv["season"],
                           "round" not in conv or int(a.get("round_num", -1)) == conv["round"]])
        if "expect_abs_number" in conv:
            correct = any(matches(abs(n), conv["expect_abs_number"]) for n in numbers(t2["answer"]))
        elif "expect_text" in conv:
            correct = conv["expect_text"].lower() in t2["answer"].lower()
        else:
            correct = bool(args_ok)
        rows.append({**conv, "turn1": t1, "turn2": t2, "tool_ok": tool_ok, "args_ok": args_ok,
                     "correct": correct, "grounded": t2["grounded"]})
        time.sleep(1.5)

    history, growth = [], []
    for q in LONG_CONVERSATION:
        r = turn(q, history)
        growth.append({k: r.get(k) for k in ("question", "input_tokens", "seconds")})
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.get("reply", "")}]
        time.sleep(1.5)

    ok = [r for r in rows if "error" not in r]
    def frac(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return f"{sum(vals)}/{len(vals)}"
    summary = {"conversations": len(rows), "errors": len(rows) - len(ok), "tool_ok": frac("tool_ok"),
               "args_ok": frac("args_ok"), "correct": frac("correct"), "grounded": frac("grounded"),
               "input_tokens_turn1": growth[0]["input_tokens"], "input_tokens_turn10": growth[-1]["input_tokens"]}
    import pandas as pd
    m = meta(f"python evals/memory_eval.py{' --tag ' + tag if tag else ''}", pd.read_csv(ROOT / "data" / "features.csv"),
             agent_model=agent.llm.model_name, seed=SEED)
    lines = header(f"Memory evaluation{' (' + tag + ')' if tag else ''}", m) + [
        f"{len(rows)} two-turn conversations (seed {SEED}) whose second question depends on the first, plus one "
        "10-turn conversation for growth. History is passed exactly as the live chat passes it.", "",
        "| Follow-up turn | Result |", "|---|---|",
        f"| Right tool (where one is needed) | {summary['tool_ok']} |", f"| Right arguments | {summary['args_ok']} |",
        f"| Correct | {summary['correct']} |", f"| Grounded | {summary['grounded']} |", f"| Errors | {summary['errors']} |", "",
        "| Turn (10-turn conversation) | Input tokens | Seconds |", "|---|---|---|"]
    lines += [f"| {i} | {g['input_tokens']} | {g['seconds']} |" for i, g in enumerate(growth, 1)]
    fails = [r for r in ok if not (r["correct"] and r["grounded"])]
    if fails:
        lines += ["", "## Failed follow-ups", ""] + [
            f"- {r['kind']}: “{r['turns'][1]}” → {r['turn2']['answer'][:160].replace(chr(10), ' ')}" for r in fails]
    write("memory_eval" + (f"_{tag}" if tag else ""), {"meta": m, "summary": summary, "conversations": rows,
                                                        "growth": growth}, lines)


if __name__ == "__main__":
    main()
