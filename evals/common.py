"""Shared metrics and result writing for evals/ scripts.

Every metric is computed per race, then averaged over races.
  spearman   rank correlation between predicted and actual finishing order
  top10      share of the actual top 10 that the prediction also puts in its top 10
  mae        mean absolute error of the predicted finishing RANK (positions)
  ndcg10     NDCG@10 with relevance = 21 - finishing position (P1 = 20, P20 = 1)
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
sys.path.insert(0, str(ROOT))

import train  # noqa: E402  (repo-root module)

TRAIN_FIRST, VAL_SEASON, TEST_SEASON = 2018, 2025, 2026


def per_race(df: pd.DataFrame, pred_rank: np.ndarray) -> pd.DataFrame:
    rows = []
    for (season, rnd), g in df.assign(_p=pred_rank).groupby(["season", "round"]):
        actual, pred, ids = g[train.TARGET].to_numpy(), g["_p"].to_numpy(), g["driver_id"].to_numpy()
        actual_rank = pd.Series(actual).rank(method="first").to_numpy()
        relevance = np.clip(21 - actual, 0, None)
        rows.append({
            "season": int(season), "round": int(rnd),
            "spearman": float(spearmanr(actual, pred)[0]),
            "top10": len(set(ids[np.argsort(actual, kind="stable")[:10]]) & set(ids[np.argsort(pred, kind="stable")[:10]])) / 10,
            "mae": float(np.abs(actual_rank - pred).mean()),
            "ndcg10": float(ndcg_score([relevance], [-pred], k=10)),
        })
    return pd.DataFrame(rows)


def summarise(races: pd.DataFrame, baseline: pd.DataFrame | None = None) -> dict:
    out = {"n_races": len(races)}
    for m in ["spearman", "top10", "mae", "ndcg10"]:
        out[f"{m}_mean"] = round(float(races[m].mean()), 3)
        out[f"{m}_std"] = round(float(races[m].std()), 3)
    if baseline is not None:
        out["races_beating_baseline_spearman"] = int((races["spearman"].to_numpy() > baseline["spearman"].to_numpy()).sum())
    return out


def grid_baseline_rank(df: pd.DataFrame) -> np.ndarray:
    grid = df["grid_position"].fillna(train.DNF_POSITION)
    return grid.groupby([df["season"], df["round"]]).rank(method="first").to_numpy()


def git_commit(paths: list[str]) -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def meta(command: str, df: pd.DataFrame, **extra) -> dict:
    import lightgbm
    import xgboost
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(["evals/*.py", "train.py", "2Feature_engineering.py", "configs", "data"]),
        "command": command,
        "libraries": {"xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__},
        "dataset": {"file": "data/features.csv", "rows": len(df),
                    "races": int(df[["season", "round"]].drop_duplicates().shape[0]),
                    "seasons": f"{df.season.min()}-{df.season.max()}"},
        **extra,
    }


def fmt(s: dict, m: str) -> str:
    return f"{s[m + '_mean']:.3f} ± {s[m + '_std']:.3f}"


def write(name: str, payload: dict, md_lines: list[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / f"{name}.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


def header(title: str, m: dict) -> list[str]:
    d = m["dataset"]
    return [f"# {title}", "",
            f"Date {m['date']} · commit `{m['git_commit']}` · xgboost {m['libraries']['xgboost']}, "
            f"lightgbm {m['libraries']['lightgbm']} · {d['rows']} rows, {d['races']} races ({d['seasons']}) · `{m['command']}`",
            ""]
