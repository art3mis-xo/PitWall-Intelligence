"""
Paraphrase robustness: the same question asked in 4 wordings. An agent that only works for one
phrasing passes a templated eval but fails real users.

Expected answers are computed from the data (as in agent_eval.py) and scored with the same rules.
  accuracy      share of all asked questions answered correctly
  consistency   share of items answered correctly in every wording

    python evals/robustness_eval.py     # needs GROQ_API_KEY; 8 items x 4 wordings
Writes evals/results/robustness_eval.json and .md
"""
import random
import sys
import time

from common import ROOT, header, meta, write

sys.path.insert(0, str(ROOT))
import agent_eval  # noqa: E402
from pitwall.backend.services import agent  # noqa: E402
from pitwall.backend.services.race_predictor import load_features, predict_race  # noqa: E402

SEED, SEASON = 11, 2026
STATS_WORDINGS = ["What is {who}'s teammate head-to-head win percentage?",
                  "How often does {who} beat their teammate?",
                  "{who} vs teammate: what's the head-to-head win rate?",
                  "Give me {who}'s head-to-head record against teammates as a percentage."]
PREDICT_WORDINGS = ["Who does the model predict to win round {r} of {s}?",
                    "Predicted winner for {s} round {r}?",
                    "According to PitWall, which driver is favourite in race {r} of the {s} season?",
                    "Who's on top of the predicted order for {s} R{r}?"]


def items() -> list[dict]:
    rng = random.Random(SEED)
    df = load_features()
    season = df[df.season == SEASON]
    names = season.drop_duplicates("driver_id").set_index("driver_id")["driver_full_name"]
    out = []
    for i, code in enumerate(rng.sample(sorted(names.index), 5)):
        latest = df[df.driver_id == code].sort_values(["season", "round"]).iloc[-1]
        who = [names[code], names[code].split()[-1]][i % 2]
        base = {"kind": "stats", "tool": "features_for_stats_tool", "driver": code,
                "expect_number": round(100 * float(latest["teammate_win_pct"]), 1)}
        out.append([{**base, "question": w.format(who=who)} for w in STATS_WORDINGS])
    for rnd in rng.sample(sorted(season["round"].unique()), 3):
        winner = predict_race(SEASON, int(rnd))[0]
        base = {"kind": "predict", "tool": "race_predictor_tool", "season": SEASON, "round": int(rnd),
                "expect_text": [winner["driver_full_name"].split()[-1]]}
        out.append([{**base, "question": w.format(r=rnd, s=SEASON)} for w in PREDICT_WORDINGS])
    return out


def main():
    groups = []
    for group in items():
        rows = []
        for case in group:
            rows.append(agent_eval.run_case(case))
            time.sleep(agent_eval.PAUSE_S)
        groups.append(rows)
    flat = [r for g in groups for r in g if "error" not in r]
    ok_groups = [g for g in groups if all("error" not in r for r in g)]
    summary = {"items": len(groups), "questions": len(flat),
               "errors": sum("error" in r for g in groups for r in g),
               "accuracy": f"{sum(r['correct'] for r in flat)}/{len(flat)}",
               "consistency": f"{sum(all(r['correct'] for r in g) for g in ok_groups)}/{len(ok_groups)}"}
    by_wording = [f"{sum(g[i]['correct'] for g in ok_groups)}/{len(ok_groups)}" for i in range(4)]
    import pandas as pd
    m = meta("python evals/robustness_eval.py", pd.read_csv(ROOT / "data" / "features.csv"),
             agent_model=agent.llm.model_name, seed=SEED)
    lines = header("Paraphrase robustness (live agent)", m) + [
        f"{summary['items']} questions, each asked in 4 wordings (seed {SEED}); answers computed from the data, "
        "scored with agent_eval.py's rules.", "",
        "| Metric | Result |", "|---|---|",
        f"| Accuracy, all wordings | {summary['accuracy']} |",
        f"| Consistency (item right in every wording) | {summary['consistency']} |",
        f"| Accuracy by wording (1-4) | {' · '.join(by_wording)} |",
        f"| Errors | {summary['errors']} |", "", "## Inconsistent items", ""]
    for g in ok_groups:
        if not all(r["correct"] for r in g):
            lines.append(f"- {g[0]['kind']}: " + "; ".join(f"“{r['question']}” → {'✓' if r['correct'] else '✗'}" for r in g))
    write("robustness_eval", {"meta": m, "summary": summary, "by_wording": by_wording, "groups": groups}, lines)


if __name__ == "__main__":
    main()
