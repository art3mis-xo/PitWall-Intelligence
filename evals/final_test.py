"""
Final, one-time test on 2026: train the frozen configs/final.json on 2018-2025,
evaluate every 2026 race in data/features.csv.

The configuration was chosen on 2025 only (evals/experiments.py). This script
refuses to run if configs/final.json has uncommitted changes, so the test
season cannot feed back into the choice.

    python evals/final_test.py
Writes evals/results/final_test_2026.json and .md
"""
import subprocess
import sys

from common import TEST_SEASON, TRAIN_FIRST, ROOT, fmt, grid_baseline_rank, header, meta, per_race, summarise, write

sys.path.insert(0, str(ROOT))
import train  # noqa: E402

REFERENCE = [("configs/clean_xgb.json", "XGBoost, leak-free notebook features (reference, not tuned)")]


def main():
    if subprocess.run(["git", "status", "--porcelain", "--", "configs/final.json"],
                      capture_output=True, text=True, cwd=ROOT).stdout.strip():
        sys.exit("configs/final.json has uncommitted changes; commit the frozen config first.")

    df = train.load_dataset()
    tr = df[(df.season >= TRAIN_FIRST) & (df.season < TEST_SEASON)]
    te = df[df.season == TEST_SEASON]

    base_races = per_race(te, grid_baseline_rank(te))
    baseline = summarise(base_races)
    rows = [{"model": "Grid-position baseline (finish = grid)", **baseline}]
    final = train.load_config(ROOT / "configs" / "final.json")
    for path, label in [("configs/final.json", "Final model: " + final["description"])] + REFERENCE:
        config = train.load_config(ROOT / path)
        races = per_race(te, train.predict_rank(train.fit(config, tr), config, te))
        rows.append({"model": label, "config": path, "n_features": len(config["features"]),
                     "per_race": races.to_dict(orient="records"), **summarise(races, base_races)})

    m = meta("python evals/final_test.py", df, protocol=f"train {TRAIN_FIRST}-{TEST_SEASON - 1}, evaluate {TEST_SEASON} once",
             train_rows=len(tr), test_rows=len(te),
             note=f"Selection used {TEST_SEASON - 1} only. {TEST_SEASON} was seen once before, in evals/backtest_audit.py, "
                  "with the old leaky/leak-removed notebook features, before any of these choices were made.")
    rounds = sorted(te["round"].unique())
    lines = header(f"Final test: {TEST_SEASON} (evaluated once)", m) + [
        f"Train {TRAIN_FIRST}–{TEST_SEASON - 1} ({len(tr)} rows); test {TEST_SEASON} rounds {rounds[0]}–{rounds[-1]} "
        f"({len(te)} rows, {baseline['n_races']} races). Per-race metrics, mean ± std over races. "
        "MAE is on predicted finishing rank.",
        "",
        "| Model | Spearman | Top-10 overlap | MAE | NDCG@10 | Spearman vs grid baseline | Races beating baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = r["spearman_mean"] - baseline["spearman_mean"]
        vs = "—" if r is rows[0] else f"{delta:+.3f}"
        beat = f"{r['races_beating_baseline_spearman']}/{r['n_races']}" if "races_beating_baseline_spearman" in r else "—"
        lines.append(f"| {r['model']} | {fmt(r, 'spearman')} | {fmt(r, 'top10')} | {fmt(r, 'mae')} | "
                     f"{fmt(r, 'ndcg10')} | {vs} | {beat} |")
    lines += ["", m["note"]]
    write(f"final_test_{TEST_SEASON}", {"meta": m, "results": rows}, lines)


if __name__ == "__main__":
    main()
