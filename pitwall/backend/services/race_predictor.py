
# race_predictor.py
import importlib.util
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
BASE_RESULTS_PATH = BASE_DIR / "data"   / "driver_race_results.csv"
FEATURE_PIPELINE  = BASE_DIR / "2Feature_engineering.py"

# Before qualifying these are unknown. The model never saw them missing in training
# (LightGBM would treat a missing quali position like 0, i.e. better than pole), so a
# not-yet-qualified race uses each driver's average over their last 3 races instead,
# and the API marks these inputs as estimated.
QUALI_ESTIMATED = ["quali_position", "quali_best_gap_pct", "quali_gap_to_pole",
                   "teammate_quali_delta", "team_avg_pace_delta"]
GRID_ASSUMED = ["grid_position", "grid_penalty_places", "had_grid_penalty"]
ESTIMATED_INPUTS = set(QUALI_ESTIMATED + GRID_ASSUMED)
RECENT_RACES = 3

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


@lru_cache(maxsize=1)
def _feature_pipeline():
    # 2Feature_engineering.py can't be imported by name (it starts with a digit)
    spec = importlib.util.spec_from_file_location("feature_engineering", FEATURE_PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_name(season: int, round_num: int) -> str:
    """Event name as used for circuit_id in the data, e.g. 'São Paulo Grand Prix'."""
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        match = schedule[schedule["RoundNumber"] == round_num]
        if not match.empty:
            return str(match.iloc[0]["EventName"])
    except Exception as e:
        print(f"[PREDICT] schedule lookup failed for {season} R{round_num}: {e}")
    return f"Round {round_num}"   # unknown circuit: circuit-history features stay missing


@lru_cache(maxsize=8)
def _upcoming_race_rows(season: int, round_num: int) -> pd.DataFrame:
    """Features for a race not in the data yet, built by the same pipeline as training.

    The race is appended to the results history with no outcome, so every history
    feature uses all earlier races and circuit features use the actual circuit.
    Qualifying hasn't happened: qualifying features are estimated from each driver's
    last 3 races, the grid is taken as the estimated qualifying position, and no grid
    penalty is assumed. Rows are flagged pre_qualifying so the API can say so."""
    fe = _feature_pipeline()
    base = fe.load_base(BASE_RESULTS_PATH)
    latest = (base.sort_values(["season", "round"]).groupby("driver_id").last().reset_index())
    entrants = latest[latest["driver_id"].isin(ACTIVE_DRIVERS_2026)]
    race = pd.DataFrame({
        "season": season, "round": round_num, "circuit_id": _event_name(season, round_num),
        "driver_id": entrants["driver_id"].to_numpy(),
        "driver_full_name": entrants["driver_full_name"].to_numpy(),
        "team_id": entrants["team_id"].to_numpy(),
        "race_had_rain": False,   # an outcome; only read by races after this one
    })
    built = fe.build_features(pd.concat([base, race], ignore_index=True))
    is_race = (built["season"] == season) & (built["round"] == round_num)
    rows = built[is_race].copy()

    recent = (built[~is_race].sort_values(["season", "round"])
                             .groupby("driver_id")[QUALI_ESTIMATED]
                             .apply(lambda g: g.tail(RECENT_RACES).mean()))
    rows[QUALI_ESTIMATED] = recent.reindex(rows["driver_id"]).to_numpy()
    rows["grid_position"] = rows["quali_position"]
    rows[["grid_penalty_places", "had_grid_penalty"]] = 0
    rows["pre_qualifying"] = True

    team_codes = {t: i for i, t in enumerate(load_features()["team_id"].astype("category").cat.categories)}
    rows["team_id_enc"] = rows["team_id"].map(team_codes).fillna(-1).astype(int)
    return rows


def predict_race(season: int, round_num: int) -> list[dict]:
    df         = load_features()
    model      = load_model()
    explainer  = load_explainer()
    feat_cols  = load_feature_cols()

    race_rows = df[(df["season"] == season) & (df["round"] == round_num)]
    if race_rows.empty:
        race_rows = _upcoming_race_rows(season, round_num)

    X = race_rows[feat_cols].astype(float)

    sign      = _score_sign()
    preds     = sign * model.predict(X)
    shap_vals = sign * explainer.shap_values(X)

    pre_qualifying = bool(race_rows.get("pre_qualifying", pd.Series(dtype=bool)).any())
    estimated = ESTIMATED_INPUTS if pre_qualifying else set()

    results = []
    for i, (_, row) in enumerate(race_rows.iterrows()):
        shap_dict = {
            col: round(float(shap_vals[i][j]), 4)
            for j, col in enumerate(feat_cols)
        }
        known_grid = not pre_qualifying and pd.notna(row["grid_position"])
        results.append({
            "driver_id":          row["driver_id"],
            "driver_full_name":   row.get("driver_full_name", row["driver_id"]),
            "team_id":            row["team_id"],
            "grid_position":      int(row["grid_position"]) if known_grid else None,
            "pre_qualifying":     pre_qualifying,
            "predicted_score":    round(float(preds[i]), 4),   # lower = better finish
            "shap_values":        shap_dict,
            "top_factors":        _top_factors(shap_dict, X.iloc[i], estimated),
        })

    results.sort(key=lambda x: x["predicted_score"])
    for rank, r in enumerate(results, 1):
        r["predicted_rank"] = rank

    return results


def _top_factors(shap_dict: dict, feature_values: pd.Series, estimated: set = frozenset(),
                 n: int = 5) -> list[dict]:
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "feature":   k,
            "label":     FEATURE_LABELS.get(k, k.replace("_", " ").title()),
            # the feature's own value for this driver (None when unknown, e.g. no Q3 time)
            "value":     round(float(feature_values[k]), 4) if pd.notna(feature_values[k]) else None,
            "estimated": k in estimated,   # pre-qualifying stand-in, not the actual value
            "shap_value": v,
            # lower score = better finish, so a negative SHAP value moves the driver up the order
            "direction": "improves" if v < 0 else "worsens",
        }
        for k, v in sorted_feats[:n]
    ]