"""
Train the race predictor from data/features.csv.

    python train.py                              # configs/final.json, all seasons
    python train.py --config configs/clean_xgb.json --train-until 2024

Writes models/race_predictor.pkl, models/feature_cols.pkl and
models/model_config.json (the config the model was trained with).
evals/ scripts import load_dataset / fit / predict_rank from here, so the
evaluated model and the served model are built the same way.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "data" / "features.csv"
MODELS_DIR = ROOT / "models"
TARGET = "finish_position"
DNF_POSITION = 20            # the 3 rows with no finishing position
RACE_WEATHER = ["race_avg_air_temp", "race_avg_track_temp"]   # measured during the race


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    if config.get("race_weather"):
        config["features"] = config["features"] + RACE_WEATHER
    return config


def load_dataset(path: Path = FEATURES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["had_grid_penalty"] = df["had_grid_penalty"].astype(int)
    df["team_id_enc"] = df["team_id"].astype("category").cat.codes
    df[TARGET] = df[TARGET].fillna(DNF_POSITION).astype(float)
    return df


def _relevance(df: pd.DataFrame) -> np.ndarray:
    # Ranking label: P1 -> 20 ... P20 and below -> 0 (higher is better)
    return (21 - df[TARGET]).clip(lower=0).astype(int).to_numpy()


def _grid(df: pd.DataFrame) -> pd.Series:
    return df["grid_position"].fillna(DNF_POSITION)


def fit(config: dict, train: pd.DataFrame):
    X, kind, params = train[config["features"]], config["model"], config.get("params", {})
    if kind == "xgb_regressor":
        from xgboost import XGBRegressor
        # "grid_residual": learn finish - grid, so the grid-position baseline is the starting point
        y = train[TARGET] - _grid(train) if config.get("target") == "grid_residual" else train[TARGET]
        return XGBRegressor(random_state=42, n_jobs=-1, **params).fit(X, y)
    if kind == "lgbm_lambdarank":
        from lightgbm import LGBMRanker
        ordered = train.sort_values(["season", "round"])
        groups = ordered.groupby(["season", "round"], sort=False).size().to_numpy()
        model = LGBMRanker(objective="lambdarank", label_gain=list(range(32)), random_state=42,
                           verbose=-1, **params)
        return model.fit(ordered[config["features"]], _relevance(ordered), group=groups)
    raise ValueError(f"unknown model type {kind!r}")


def predict_score(model, config: dict, df: pd.DataFrame) -> np.ndarray:
    """Lower score = better predicted finish, for every model type."""
    raw = model.predict(df[config["features"]])
    if config["model"] == "lgbm_lambdarank":
        return -raw
    return _grid(df).to_numpy() + raw if config.get("target") == "grid_residual" else raw


def predict_rank(model, config: dict, df: pd.DataFrame) -> np.ndarray:
    """Predicted finishing rank (1 = winner) within each race."""
    score = pd.Series(predict_score(model, config, df), index=df.index)
    return score.groupby([df["season"], df["round"]]).rank(method="first").to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "final.json"))
    parser.add_argument("--train-until", type=int, default=None, help="last season to train on (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_dataset()
    train = df if args.train_until is None else df[df["season"] <= args.train_until]
    model = fit(config, train)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "race_predictor.pkl")
    joblib.dump(config["features"], MODELS_DIR / "feature_cols.pkl")
    (MODELS_DIR / "model_config.json").write_text(json.dumps(
        {**config, "trained_on_rows": len(train),
         "trained_on_seasons": f"{train.season.min()}-{train.season.max()}"}, indent=2))
    print(f"Trained {config['name']} on {len(train)} rows ({train.season.min()}-{train.season.max()}) "
          f"-> models/race_predictor.pkl")


if __name__ == "__main__":
    main()
