"""
Leakage audit: reproduce the backtest from legacy/3Model_training.ipynb and compare it with
a grid-position baseline and with the leaky features removed.

Protocol (same as the notebook): for each holdout season, train on all earlier
seasons, evaluate per race, average over races.

Run from the repo root:
    python evals/backtest_audit.py
Writes evals/results/backtest_before.json and evals/results/backtest_before.md
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "features.csv"
FEATURE_COLS_PKL = ROOT / "models" / "feature_cols.pkl"
RESULTS = ROOT / "evals" / "results"
TARGET = "finish_position"
HOLDOUTS = [2025, 2026]

# Hyperparameters copied from legacy/3Model_training.ipynb cell 7
XGB_PARAMS = dict(objective="reg:squarederror", n_estimators=500, max_depth=4, learning_rate=0.05,
                  subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.5,
                  reg_alpha=0.1, random_state=42, n_jobs=-1)

# Features computed from the race being predicted, or from data that includes future races
LEAKY = ["avg_team_finish_position", "rolling_avg_finish_position", "var_delta_i_norm",
         "teammate_wins", "teammate_losses", "teammate_win_pct", "circuit_wet_race_frequency"]
# Measured during the race itself; dropped in the stricter variant
RACE_WEATHER = ["race_avg_air_temp", "race_avg_track_temp"]


def load() -> pd.DataFrame:
    raw = pd.read_csv(FEATURES_CSV)
    for c in ["q1_time", "q2_time", "q3_time", "pole_q3_time"]:
        raw[c] = pd.to_timedelta(raw[c], errors="coerce").dt.total_seconds()
    raw["had_grid_penalty"] = raw["had_grid_penalty"].astype(int)
    raw["team_id_enc"] = raw["team_id"].astype("category").cat.codes
    raw[TARGET] = raw[TARGET].fillna(20).astype(float)
    return raw


def per_race(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for (season, rnd), g in df.assign(_p=pred).groupby(["season", "round"]):
        a, p, ids = g[TARGET].to_numpy(), g["_p"].to_numpy(), g["driver_id"].to_numpy()
        rows.append({
            "season": int(season), "round": int(rnd),
            "spearman": spearmanr(a, p)[0],
            "top10": len(set(ids[np.argsort(a)[:10]]) & set(ids[np.argsort(p)[:10]])) / 10,
            "mae": mean_absolute_error(a, p),
        })
    return pd.DataFrame(rows)


def summarise(r: pd.DataFrame) -> dict:
    return {
        "n_races": len(r),
        "spearman_mean": round(float(r.spearman.mean()), 3),
        "spearman_std": round(float(r.spearman.std()), 3),
        "top10_mean": round(float(r.top10.mean()), 3),
        "mae_mean": round(float(r.mae.mean()), 3),
    }


def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "evals/backtest_audit.py", "data", "models"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def main():
    raw = load()
    feats = joblib.load(FEATURE_COLS_PKL)
    no_leak = [c for c in feats if c not in LEAKY]
    no_leak_no_weather = [c for c in no_leak if c not in RACE_WEATHER]

    variants = [
        ("Grid-position baseline (finish = grid)", None),
        ("avg_team_finish_position alone (no model)", None),
        (f"XGBoost, all {len(feats)} features (notebook)", feats),
        (f"XGBoost, leaky removed ({len(no_leak)} features, race weather kept)", no_leak),
        (f"XGBoost, leaky + race weather removed ({len(no_leak_no_weather)} features)", no_leak_no_weather),
    ]

    results = []
    for hold in HOLDOUTS:
        train, test = raw[raw.season < hold], raw[raw.season == hold]
        for name, cols in variants:
            if name.startswith("Grid"):
                pred = test["grid_position"].fillna(20).to_numpy()
            elif name.startswith("avg_team"):
                pred = test["avg_team_finish_position"].to_numpy()
            else:
                pred = XGBRegressor(**XGB_PARAMS).fit(train[cols], train[TARGET]).predict(test[cols])
            results.append({"holdout": hold, "model": name, "features": cols,
                            "train_rows": len(train), "test_rows": len(test),
                            **summarise(per_race(test, pred))})

    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": "python evals/backtest_audit.py",
        "model": f"xgboost {xgboost.__version__} XGBRegressor",
        "xgb_params": XGB_PARAMS,
        "dataset": {"file": "data/features.csv", "rows": len(raw),
                    "races": int(raw[["season", "round"]].drop_duplicates().shape[0]),
                    "seasons": f"{raw.season.min()}-{raw.season.max()}"},
        "protocol": "train on seasons < holdout; metrics per race, averaged over races",
        "leaky_features": LEAKY,
        "race_weather_features": RACE_WEATHER,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "backtest_before.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    lines = [
        "# Backtest before the leakage fix",
        "",
        f"Date {meta['date']} · commit `{meta['git_commit']}` · {meta['model']} · "
        f"{meta['dataset']['rows']} rows, {meta['dataset']['races']} races ({meta['dataset']['seasons']}) · "
        f"`{meta['command']}`",
        "",
        "Train on all seasons before the holdout; per-race metrics averaged over the holdout's races.",
        "",
        "| Holdout | Model | Races | Spearman (mean ± std) | Top-10 overlap | MAE |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['holdout']} | {r['model']} | {r['n_races']} | "
                     f"{r['spearman_mean']:.3f} ± {r['spearman_std']:.3f} | {r['top10_mean']:.3f} | {r['mae_mean']:.3f} |")
    (RESULTS / "backtest_before.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
