
# race_predictor.py
import json
import pandas as pd
import shap
import joblib
from pathlib import Path
from functools import lru_cache

BASE_DIR          = Path(__file__).parent.parent.parent.parent  # one more .parent
MODEL_PATH        = BASE_DIR / "models" / "race_predictor.pkl"
FEAT_COLS_PATH    = BASE_DIR / "models" / "feature_cols.pkl"
MODEL_CONFIG_PATH = BASE_DIR / "models" / "model_config.json"   # written by train.py
FEATURES_PATH     = BASE_DIR / "data"   / "features.csv"

ACTIVE_DRIVERS_2026 = [
    'ALB', 'ALO', 'ANT', 'BEA', 'BOR', 'BOT',
    'COL', 'GAS', 'HAD', 'HAM', 'HUL', 'LAW',
    'LEC', 'LIN', 'NOR', 'OCO', 'PER', 'PIA',
    'RUS', 'SAI', 'STR', 'VER'
]

FEATURE_LABELS = {
    "grid_position":                  "Grid position",
    "grid_penalty_places":            "Grid penalty (places)",
    "had_grid_penalty":               "Has grid penalty",
    "quali_position":                 "Qualifying position",
    "quali_best_gap_pct":             "Quali gap to pole (%)",
    "quali_gap_to_pole":              "Q3 gap to pole (s)",
    "teammate_quali_delta":           "Quali position vs teammate",
    "team_avg_pace_delta":            "Team quali pace delta",
    "rolling_avg_pace_delta":         "Team quali pace (last 3)",
    "pace_trend":                     "Team quali pace trend",
    "team_avg_finish_last3":          "Team avg finish (last 3 races)",
    "team_avg_finish_last5":          "Team avg finish (last 5 races)",
    "team_dnf_rate_last10":           "Team DNF rate (last 10)",
    "driver_avg_finish_last3":        "Driver avg finish (last 3 races)",
    "driver_avg_finish_last5":        "Driver avg finish (last 5 races)",
    "driver_points_last5":            "Driver points (last 5 races)",
    "driver_dnf_rate_last10":         "Driver DNF rate (last 10)",
    "avg_finish_grid_delta":          "Driver avg places gained/lost",
    "teammate_wins":                  "Teammate H2H wins (to date)",
    "teammate_losses":                "Teammate H2H losses (to date)",
    "teammate_win_pct":               "Teammate H2H win % (to date)",
    "career_starts_at_circuit":       "Career starts at circuit",
    "avg_finish_at_circuit":          "Avg finish at circuit",
    "driver_circuit_pos_change_hist": "Places gained at circuit (driver)",
    "circuit_abs_pos_change_hist":    "Circuit position changes (history)",
    "circuit_pos_change_var_hist":    "Circuit overtaking variance (history)",
    "circuit_wet_race_frequency":     "Circuit wet-race frequency",
    "race_avg_air_temp":              "Air temperature",
    "race_avg_track_temp":            "Track temperature",
    "team_id_enc":                    "Team identity",
}


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    config = json.loads(MODEL_CONFIG_PATH.read_text())
    if config.get("target") == "grid_residual":
        raise ValueError("serving a grid_residual model is not supported: SHAP would explain the residual only")
    return config


def _score_sign() -> int:
    # A ranker outputs higher = better; regressors predict a position (lower = better).
    # Flipping the ranker keeps one convention: lower score = better finish.
    return -1 if load_model_config()["model"] == "lgbm_lambdarank" else 1


@lru_cache(maxsize=1)
def load_feature_cols() -> list[str]:
    """Load the exact feature columns the model was trained on."""
    return joblib.load(FEAT_COLS_PATH)


@lru_cache(maxsize=1)
def load_explainer():
    return shap.TreeExplainer(load_model())


def shap_base_value() -> float:
    """SHAP base value on the same scale as predicted_score."""
    return _score_sign() * float(load_explainer().expected_value)


@lru_cache(maxsize=1)
def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    # fix dtypes
    for col in ["q1_time", "q2_time", "q3_time", "pole_q3_time"]:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce").dt.total_seconds()
    df["had_grid_penalty"] = df["had_grid_penalty"].astype(int)
    df["team_id_enc"]      = df["team_id"].astype("category").cat.codes
    return df


def predict_race(season: int, round_num: int) -> list[dict]:
    df         = load_features()
    model      = load_model()
    explainer  = load_explainer()
    feat_cols  = load_feature_cols()

    race_rows = df[(df["season"] == season) & (df["round"] == round_num)]

    upcoming = False
    if race_rows.empty:
        upcoming  = True
        season_df = df[df["season"] == season]

        source_df = season_df if not season_df.empty else df

        latest = (
            source_df.sort_values(["season", "round"])
                     .groupby("driver_id")
                     .last()
                     .reset_index()
        )

        # filter to active 2026 grid — removes retired drivers
        latest    = latest[latest["driver_id"].isin(ACTIVE_DRIVERS_2026)].copy()
        race_rows = latest.copy()
        race_rows["circuit_id"] = f"Round {round_num}"

    X = race_rows[feat_cols].copy()
    X["had_grid_penalty"] = X["had_grid_penalty"].astype(int)

    sign      = _score_sign()
    preds     = sign * model.predict(X)
    shap_vals = sign * explainer.shap_values(X)

    results = []
    for i, (_, row) in enumerate(race_rows.iterrows()):
        shap_dict = {
            col: round(float(shap_vals[i][j]), 4)
            for j, col in enumerate(feat_cols)
        }
        results.append({
            "driver_id":          row["driver_id"],
            "driver_full_name":   row.get("driver_full_name", row["driver_id"]),
            "team_id":            row["team_id"],
            "grid_position":      int(row["grid_position"]) if pd.notna(row["grid_position"]) else None,
            "predicted_score":    round(float(preds[i]), 4),   # lower = better finish
            "shap_values":        shap_dict,
            "top_factors":        _top_factors(shap_dict, feat_cols),
        })

    results.sort(key=lambda x: x["predicted_score"])
    for rank, r in enumerate(results, 1):
        r["predicted_rank"] = rank

    return results


def _top_factors(shap_dict: dict, feat_cols: list, n: int = 5) -> list[dict]:
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   k,
            "label":     FEATURE_LABELS.get(k, k.replace("_", " ").title()),
            "value":     v,
            "shap_value": v,
            # lower score = better finish, so a negative SHAP value moves the driver up the order
            "direction": "improves" if v < 0 else "worsens",
        }
        for k, v in sorted_feats[:n]
    ]