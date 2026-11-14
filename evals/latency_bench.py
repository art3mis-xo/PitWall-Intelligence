"""
Latency benchmark for the chat agent: the same questions, repeated, under different model
settings. Single eval runs swing with Groq load (see agent_eval_history.md), so each question is
run several times and settings are interleaved so they see the same load.

    python evals/latency_bench.py      # needs GROQ_API_KEY; 6 questions x 3 runs x 2 settings
Writes evals/results/latency_bench.json and .md
"""
import statistics
import sys
import time

from common import ROOT, header, meta, write

sys.path.insert(0, str(ROOT))
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from pitwall.backend.services import agent  # noqa: E402

RUNS = 3
SETTINGS = {"default reasoning": None, "reasoning_effort=low": "low"}
QUESTIONS = [
    "What is Lando Norris's teammate head-to-head win percentage?",
    "What is Verstappen's average finish at the circuit of his latest race?",
    "Who does the model predict to win round 9 of the 2026 season?",
    "Who does the model predict to win round 13 of the 2026 season?",
    "Why is Oscar Piastri predicted to finish P8 in round 9 of 2026?",
    "Why is Charles Leclerc predicted where he is in round 12 of 2026?",
]


def use(effort):
    llm = ChatGroq(model=agent.llm.model_name, temperature=0, reasoning_effort=effort, timeout=30, max_retries=2)
    agent.llm, agent.llm_with_tools = llm, llm.bind_tools(agent.TOOLS)


def one(question):
    start = time.perf_counter()
    out = agent.app.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 12})
    seconds = time.perf_counter() - start
    calls = [m for m in out["messages"] if isinstance(m, AIMessage) and m.usage_metadata]
    return {"seconds": round(seconds, 2), "llm_calls": len(calls),
            "input_tokens": sum(m.usage_metadata.get("input_tokens", 0) for m in calls),
            "output_tokens": sum(m.usage_metadata.get("output_tokens", 0) for m in calls),
            "answered": bool(str(out["messages"][-1].content).strip())}


def main():
    runs = {name: [] for name in SETTINGS}
    for r in range(RUNS):
        for q in QUESTIONS:
            for name, effort in SETTINGS.items():   # interleaved: both settings see the same load
                use(effort)
                runs[name].append({"run": r, "question": q, **one(q)})
                time.sleep(1.5)

    def p(xs, q):
        xs = sorted(xs)
        return xs[int(q * (len(xs) - 1))]

    summary = {}
    for name, rows in runs.items():
        secs = [x["seconds"] for x in rows]
        summary[name] = {"n": len(rows), "p50_s": round(p(secs, 0.5), 2), "p90_s": round(p(secs, 0.9), 2),
                         "llm_calls_mean": round(statistics.mean(x["llm_calls"] for x in rows), 2),
                         "input_tokens_mean": int(statistics.mean(x["input_tokens"] for x in rows)),
                         "output_tokens_mean": int(statistics.mean(x["output_tokens"] for x in rows)),
                         "answered": sum(x["answered"] for x in rows)}
    import pandas as pd
    m = meta("python evals/latency_bench.py", pd.read_csv(ROOT / "data" / "features.csv"),
             agent_model=agent.llm.model_name, runs=RUNS, questions=len(QUESTIONS))
    lines = header("Agent latency benchmark", m) + [
        f"{len(QUESTIONS)} questions (stats, prediction, explanation) x {RUNS} runs per setting, settings interleaved. "
        "Latency is end to end (all model and tool calls). Answer quality per setting: see agent_eval_history.md.", "",
        "| Setting | Runs | p50 (s) | p90 (s) | LLM calls / question | Input tokens | Output tokens | Answered |",
        "|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['n']} | {s['p50_s']} | {s['p90_s']} | {s['llm_calls_mean']} | "
                     f"{s['input_tokens_mean']} | {s['output_tokens_mean']} | {s['answered']}/{s['n']} |")
    write("latency_bench", {"meta": m, "summary": summary, "runs": runs}, lines)


if __name__ == "__main__":
    main()
