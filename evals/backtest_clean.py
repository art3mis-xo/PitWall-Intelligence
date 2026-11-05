"""
Leak-free backtest on the VALIDATION season only (train 2018-2024, evaluate 2025).

    python evals/backtest_clean.py
Writes evals/results/backtest_clean.json and .md
"""
import sys

from common import TRAIN_FIRST, VAL_SEASON, ROOT, fmt, grid_baseline_rank, header, meta, per_race, summarise, write

sys.path.insert(0, str(ROOT))
import train  # noqa: E402

CONFIGS = ["configs/clean_xgb.json"]


def main():
    df = train.load_dataset()
    tr = df[(df.season >= TRAIN_FIRST) & (df.season < VAL_SEASON)]
    va = df[df.season == VAL_SEASON]
    assert tr.season.max() < VAL_SEASON and va.season.unique().tolist() == [VAL_SEASON]

    base_races = per_race(va, grid_baseline_rank(va))
    rows = [{"model": "Grid-position baseline (finish = grid)", **summarise(base_races)}]
    for path in CONFIGS:
        for weather in (False, True):
            config = train.load_config(ROOT / path)
            if weather:
                config["features"] = config["features"] + train.RACE_WEATHER
            model = train.fit(config, tr)
            races = per_race(va, train.predict_rank(model, config, va))
            label = f"XGBoost clean, {len(config['features'])} features" + (" (+ race weather)" if weather else "")
            rows.append({"model": label, "config": path, "race_weather": weather,
                         "features": config["features"], **summarise(races, base_races)})

    m = meta("python evals/backtest_clean.py", df, protocol=f"train {TRAIN_FIRST}-{VAL_SEASON - 1}, evaluate {VAL_SEASON} (validation)",
             train_rows=len(tr), eval_rows=len(va))
    lines = header("Leak-free backtest (validation 2025)", m) + [
        f"Train {TRAIN_FIRST}–{VAL_SEASON - 1} ({len(tr)} rows), evaluate {VAL_SEASON} ({len(va)} rows, "
        f"{rows[0]['n_races']} races). Per-race metrics, mean ± std over races. MAE is on predicted finishing rank.",
        "",
        "| Model | Spearman | Top-10 overlap | MAE | NDCG@10 | Races beating baseline (Spearman) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        beat = f"{r['races_beating_baseline_spearman']}/{r['n_races']}" if "races_beating_baseline_spearman" in r else "—"
        lines.append(f"| {r['model']} | {fmt(r, 'spearman')} | {fmt(r, 'top10')} | {fmt(r, 'mae')} | {fmt(r, 'ndcg10')} | {beat} |")
    write("backtest_clean", {"meta": m, "results": rows}, lines)


if __name__ == "__main__":
    main()
