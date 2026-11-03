
# race_predictor.py
import pandas as pd
import shap
import joblib
from pathlib import Path
from functools import lru_cache

BASE_DIR       = Path(__file__).parent.parent.parent.parent  # one more .parent
MODEL_PATH     = BASE_DIR / "models" / "xgb_race_predictor_final.pkl"
FEAT_COLS_PATH = BASE_DIR / "models" / "feature_cols.pkl"
FEATURES_PATH  = BASE_DIR / "data"   / "features.csv"

ACTIVE_DRIVERS_2026 = [
    'ALB', 'ALO', 'ANT', 'BEA', 'BOR', 'BOT',
    'COL', 'GAS', 'HAD', 'HAM', 'HUL', 'LAW',
    'LEC', 'LIN', 'NOR', 'OCO', 'PER', 'PIA',
    'RUS', 'SAI', 'STR', 'VER'
]

FEATURE_LABELS = {
    "grid_position":               "Grid position",
    "rolling_avg_finish_position": "Team rolling avg finish",
    "avg_team_finish_position":    "Team avg finish (season)",
    "quali_gap_to_pole":           "Gap to pole (quali)",
    "rolling_avg_pace_delta":      "Rolling pace delta",
    "team_avg_pace_delta":         "Team pace delta",
    "career_starts_at_circuit":    "Career starts at circuit",
    "avg_finish_at_circuit":       "Avg finish at circuit",
    "avg_finish_grid_delta":       "Avg quali-to-race delta",
    "teammate_win_pct":            "Teammate H2H win %",
    "var_delta_i_norm":            "Overtaking difficulty",
    "circuit_wet_race_frequency":  "Circuit wet frequency",
    "grid_penalty_places":         "Grid penalty (places)",
    "had_grid_penalty":            "Has grid penalty",
    "pace_trend":                  "Pace trend (slope)",
    "race_avg_air_temp":           "Air temperature",
    "race_avg_track_temp":         "Track temperature",
    "teammate_wins":               "Teammate H2H wins",
    "teammate_losses":             "Teammate H2H losses",
    "team_id_enc":                 "Team identity",
}


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_cols() -> list[str]:
    """Load the exact feature columns the model was trained on."""
    if FEAT_COLS_PATH.exists():
        return joblib.load(FEAT_COLS_PATH)
    # fallback: hardcoded list (matches 2feature_engineering.py output)
    return [
        "grid_position", "race_avg_air_temp", "race_avg_track_temp",
        "var_delta_i_norm", "avg_team_finish_position", "rolling_avg_finish_position",
        "career_starts_at_circuit", "avg_finish_at_circuit", "avg_finish_grid_delta",
        "teammate_wins", "teammate_losses", "teammate_win_pct",
        "quali_gap_to_pole", "team_avg_pace_delta", "rolling_avg_pace_delta",
        "pace_trend", "circuit_wet_race_frequency", "grid_penalty_places",
        "had_grid_penalty", "team_id_enc",
    ]


@lru_cache(maxsize=1)
def load_explainer():
    return shap.TreeExplainer(load_model())


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

    preds     = model.predict(X)
    shap_vals = explainer.shap_values(X)

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
            "predicted_position": round(float(preds[i]), 2),
            "shap_values":        shap_dict,
            "top_factors":        _top_factors(shap_dict, feat_cols),
        })

    results.sort(key=lambda x: x["predicted_position"])
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
            "direction": "positive" if v < 0 else "negative",
        }
        for k, v in sorted_feats[:n]
    ]