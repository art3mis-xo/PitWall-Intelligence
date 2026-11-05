"""
Model selection on the VALIDATION season only (train 2018-2024, evaluate 2025).
2026 is never loaded here.

Rules, fixed before running:
  - Primary metric: mean per-race Spearman on 2025. Tie-break: NDCG@10.
  - Stage 1: compare objectives on the clean feature set
             (XGB on finish position, XGB on finish - grid, LightGBM lambdarank).
  - Stage 2: with the best objective, add pre-race feature groups one at a time
             in a fixed order; keep a group only if Spearman improves.
  - Stage 3: random hyperparameter search, fixed budget and seed, on the stage-2 winner.
  - The best configuration over all stages is written to configs/final.json.

    python evals/experiments.py
Writes evals/results/experiments.json, evals/results/experiments.md, configs/final.json
"""
import copy
import json
import random
import sys

from common import TRAIN_FIRST, VAL_SEASON, ROOT, fmt, grid_baseline_rank, header, meta, per_race, summarise, write

sys.path.insert(0, str(ROOT))
import train  # noqa: E402

HP_BUDGET, HP_SEED = 20, 0

FEATURE_GROUPS = [  # all known before the race starts
    ("quali", "qualifying position, best-lap gap to pole (%), position vs teammate",
     ["quali_position", "quali_best_gap_pct", "teammate_quali_delta"]),
    ("pit_lane", "pit-lane start flag", ["pit_lane_start"]),
    ("driver_form", "driver average finish over last 3 / 5 races, points over last 5",
     ["driver_avg_finish_last3", "driver_avg_finish_last5", "driver_points_last5"]),
    ("reliability", "driver and team DNF rate over last 10 races",
     ["driver_dnf_rate_last10", "team_dnf_rate_last10"]),
    ("circuit_history", "past position changes at this circuit (all drivers / this driver)",
     ["circuit_abs_pos_change_hist", "driver_circuit_pos_change_hist"]),
]

LGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20}

XGB_SPACE = {"n_estimators": [100, 200, 300, 500], "max_depth": [2, 3, 4, 5], "learning_rate": [0.02, 0.05, 0.1],
             "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0],
             "reg_lambda": [0.5, 1.5, 5.0]}
LGBM_SPACE = {"n_estimators": [100, 200, 300, 500], "num_leaves": [7, 15, 31], "learning_rate": [0.02, 0.05, 0.1],
              "min_child_samples": [10, 20, 40], "colsample_bytree": [0.7, 0.85, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}


def main():
    df = train.load_dataset()
    tr = df[(df.season >= TRAIN_FIRST) & (df.season < VAL_SEASON)]
    va = df[df.season == VAL_SEASON]
    assert tr.season.max() < VAL_SEASON and va.season.unique().tolist() == [VAL_SEASON]
    base_races = per_race(va, grid_baseline_rank(va))
    baseline = summarise(base_races)

    clean = train.load_config(ROOT / "configs" / "clean_xgb.json")
    log = []

    def run(exp_id, stage, description, config):
        model = train.fit(config, tr)
        s = summarise(per_race(va, train.predict_rank(model, config, va)), base_races)
        row = {"id": exp_id, "stage": stage, "description": description, "n_features": len(config["features"]),
               "config": copy.deepcopy(config), **s}
        log.append(row)
        print(f"{exp_id:>4} {description[:70]:70s} spearman {s['spearman_mean']:.3f}")
        return row

    def key(r):
        return (r["spearman_mean"], r["ndcg10_mean"])

    # ── stage 1: objective ───────────────────────────────────────────────────
    objectives = [
        ("E1", "XGB regression on finish position", {**clean}),
        ("E2", "XGB regression on finish - grid (grid residual)", {**clean, "target": "grid_residual"}),
        ("E3", "LightGBM lambdarank (group = race)", {**clean, "model": "lgbm_lambdarank", "params": LGBM_PARAMS}),
    ]
    stage1 = [run(i, 1, d, c) for i, d, c in objectives]
    best = max(stage1, key=key)

    # ── stage 2: pre-race feature groups, kept only if they help ─────────────
    for n, (name, desc, cols) in enumerate(FEATURE_GROUPS, start=4):
        candidate = copy.deepcopy(best["config"])
        candidate["features"] = candidate["features"] + cols
        row = run(f"E{n}", 2, f"{best['id']} + {name}: {desc}", candidate)
        row["kept"] = key(row) > key(best)
        if row["kept"]:
            best = row

    # ── stage 3: hyperparameter search on the stage-2 winner ─────────────────
    space = LGBM_SPACE if best["config"]["model"] == "lgbm_lambdarank" else XGB_SPACE
    rng = random.Random(HP_SEED)
    stage2_best = best
    for n in range(1, HP_BUDGET + 1):
        candidate = copy.deepcopy(stage2_best["config"])
        sampled = {k: rng.choice(v) for k, v in space.items()}
        candidate["params"] = {**candidate.get("params", {}), **sampled}
        row = run(f"H{n}", 3, f"{stage2_best['id']} hyperparameters: " + ", ".join(f"{k}={v}" for k, v in sampled.items()),
                  candidate)
        if key(row) > key(best):
            best = row

    m = meta("python evals/experiments.py", df, protocol=f"train {TRAIN_FIRST}-{VAL_SEASON - 1}, evaluate {VAL_SEASON} (validation)",
             train_rows=len(tr), eval_rows=len(va), hp_budget=HP_BUDGET, hp_seed=HP_SEED)
    final = {**best["config"], "name": "final",
             "description": f"Selected by evals/experiments.py: experiment {best['id']}, highest mean Spearman on validation {VAL_SEASON}."}
    (ROOT / "configs" / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    lines = header("Experiments log (validation 2025 only)", m) + [
        "Rules fixed before running: primary metric is mean per-race Spearman on 2025 (tie-break NDCG@10). "
        "Stage 1 compares objectives; stage 2 adds pre-race feature groups one at a time and keeps a group only if "
        f"Spearman improves; stage 3 is a random hyperparameter search ({HP_BUDGET} draws, seed {HP_SEED}). "
        "2026 is not used. MAE is on predicted finishing rank.",
        "",
        f"Grid-position baseline on 2025: Spearman {fmt(baseline, 'spearman')}, top-10 {fmt(baseline, 'top10')}, "
        f"MAE {fmt(baseline, 'mae')}, NDCG@10 {fmt(baseline, 'ndcg10')}.",
        "",
        "| ID | Stage | Experiment | Features | Spearman | Top-10 | MAE | NDCG@10 | Beats baseline (races) | Kept |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in log:
        kept = {True: "yes", False: "no"}.get(r.get("kept"), "")
        lines.append(f"| {r['id']} | {r['stage']} | {r['description']} | {r['n_features']} | {fmt(r, 'spearman')} | "
                     f"{fmt(r, 'top10')} | {fmt(r, 'mae')} | {fmt(r, 'ndcg10')} | "
                     f"{r['races_beating_baseline_spearman']}/{r['n_races']} | {kept} |")
    lines += ["", f"**Selected:** {best['id']} ({best['description']}), Spearman {fmt(best, 'spearman')} "
                  f"vs baseline {fmt(baseline, 'spearman')}. Written to `configs/final.json`."]
    write("experiments", {"meta": m, "baseline": baseline, "experiments": log, "selected": best["id"]}, lines)


if __name__ == "__main__":
    main()
