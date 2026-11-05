"""
Phase 2: Feature Engineering
Reads:  data/driver_race_results.csv   (output of 1Build_dataset_base.py + add_grid_penalties.py)
Writes: data/features.csv              (input to train.py)

Run order:
  1. 1Build_dataset_base.py
  2. add_grid_penalties.py
  3. this file

Every feature for race t is built only from information available before race t
starts: qualifying and the starting grid of race t, and results of races before t.
tests/test_features_no_leak.py checks this by changing the results of one race and
asserting no feature of that race or any earlier race moves.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_CSV  = DATA_DIR / "driver_race_results.csv"
OUTPUT_CSV = DATA_DIR / "features.csv"
CACHE_FILE = DATA_DIR / "penalty_cache.json"

RACE_KEYS = ["season", "round"]
FINISHED_STATUSES = {"Finished", "Lapped"}   # plus "+N Lap(s)", see is_dnf()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_base(path: Path = INPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["q1_time", "q2_time", "q3_time"]:
        df[col] = pd.to_timedelta(df[col])
    return df


def is_dnf(status: pd.Series) -> pd.Series:
    finished = status.isin(FINISHED_STATUSES) | status.fillna("").str.match(r"^\+\d+ Laps?$")
    return (~finished).astype(float)


def past_rolling(df: pd.DataFrame, by, col: str, window: int | None) -> pd.Series:
    """Mean of `col` over earlier rows of the same group (rows sorted by season, round).

    shift(1) drops the current row, so race t only sees races before t.
    window=None means all earlier rows (expanding)."""
    ordered = df.sort_values(RACE_KEYS)
    shifted = ordered.groupby(by)[col].shift(1)
    grouped = shifted.groupby([ordered[b] for b in ([by] if isinstance(by, str) else by)])
    if window is None:
        out = grouped.transform(lambda x: x.expanding().mean())
    else:
        out = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
    return out.reindex(df.index)


def per_race_level(df: pd.DataFrame, level_keys: list[str], values: dict) -> pd.DataFrame:
    """Aggregate to one row per (race, level) e.g. per team-race or per race."""
    return df.groupby(RACE_KEYS + level_keys, as_index=False).agg(**values)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def build_features(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["dnf"] = is_dnf(df["classified_status"])
    field_size = df.groupby(RACE_KEYS)["driver_id"].transform("count")

    # ── starting grid ─────────────────────────────────────────────────────────
    # FastF1 reports a pit-lane start as grid 0; it starts behind everyone.
    df["pit_lane_start"] = (df["grid_position"] == 0).astype(int)
    df.loc[df["grid_position"] == 0, "grid_position"] = field_size
    df["finish_grid_delta"] = df["finish_position"] - df["grid_position"]

    # ── qualifying (known before the race) ────────────────────────────────────
    # quali_position: furthest session reached first, then lap time within it.
    stage = np.select([df["q3_time"].notna(), df["q2_time"].notna(), df["q1_time"].notna()], [3, 2, 1], 0)
    stage_time = df["q3_time"].where(stage == 3, df["q2_time"].where(stage == 2, df["q1_time"]))
    df["_quali_key"] = -stage * 1e6 + stage_time.dt.total_seconds().fillna(1e5)
    df["quali_position"] = df.groupby(RACE_KEYS)["_quali_key"].rank(method="first")

    best_lap = df[["q1_time", "q2_time", "q3_time"]].min(axis=1).dt.total_seconds()
    pole_best = best_lap.groupby([df["season"], df["round"]]).transform("min")
    df["quali_best_gap_pct"] = (best_lap - pole_best) / pole_best * 100

    pole_q3 = df["q3_time"].groupby([df["season"], df["round"]]).transform("min")
    df["quali_gap_to_pole"] = (df["q3_time"] - pole_q3).dt.total_seconds()

    team_quali = df.groupby(RACE_KEYS + ["team_id"])["quali_position"]
    teammate_quali = (team_quali.transform("sum") - df["quali_position"]).where(team_quali.transform("count") == 2)
    df["teammate_quali_delta"] = df["quali_position"] - teammate_quali

    # team quali pace: this race's quali (pre-race), plus a 3-race rolling mean and slope
    def rolling_slope(x):
        y = x.values
        if len(y) < 2:
            return np.nan
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    team_pace = (per_race_level(df, ["team_id"], {"team_avg_pace_delta": ("quali_gap_to_pole", "mean")})
                 .sort_values(["team_id"] + RACE_KEYS))
    team_pace["rolling_avg_pace_delta"] = (team_pace.groupby("team_id")["team_avg_pace_delta"]
                                           .transform(lambda x: x.rolling(window=3, min_periods=1).mean()))
    team_pace["pace_trend"] = (team_pace.groupby("team_id")["team_avg_pace_delta"]
                               .transform(lambda x: x.rolling(window=3, min_periods=2).apply(rolling_slope, raw=False)))
    df = df.merge(team_pace, on=RACE_KEYS + ["team_id"], how="left")

    # ── driver form (previous races only) ─────────────────────────────────────
    df["driver_avg_finish_last3"] = past_rolling(df, "driver_id", "finish_position", 3)
    df["driver_avg_finish_last5"] = past_rolling(df, "driver_id", "finish_position", 5)
    df["driver_points_last5"]     = past_rolling(df, "driver_id", "points", 5)
    df["driver_dnf_rate_last10"]  = past_rolling(df, "driver_id", "dnf", 10)
    df["avg_finish_grid_delta"]   = past_rolling(df, "driver_id", "finish_grid_delta", None)

    # ── team form (previous races only) ───────────────────────────────────────
    team_race = per_race_level(df, ["team_id"], {"team_finish": ("finish_position", "mean"),
                                                 "team_dnf": ("dnf", "mean")})
    team_race["team_avg_finish_last3"] = past_rolling(team_race, "team_id", "team_finish", 3)
    team_race["team_avg_finish_last5"] = past_rolling(team_race, "team_id", "team_finish", 5)
    team_race["team_dnf_rate_last10"]  = past_rolling(team_race, "team_id", "team_dnf", 10)
    df = df.merge(team_race.drop(columns=["team_finish", "team_dnf"]), on=RACE_KEYS + ["team_id"], how="left")

    # ── teammate head-to-head (previous races only) ───────────────────────────
    def teammate_result(group):
        group = group.sort_values("finish_position")
        if len(group) < 2:
            return None
        return pd.DataFrame({"winner": [group.iloc[0]["driver_id"]], "loser": [group.iloc[1]["driver_id"]]})

    h2h = (df.groupby(RACE_KEYS + ["team_id"], group_keys=True)[["driver_id", "finish_position"]]
             .apply(teammate_result).reset_index(level=RACE_KEYS))
    wins   = h2h.groupby(RACE_KEYS + ["winner"]).size().rename("win").rename_axis(RACE_KEYS + ["driver_id"])
    losses = h2h.groupby(RACE_KEYS + ["loser"]).size().rename("loss").rename_axis(RACE_KEYS + ["driver_id"])
    df = df.join(wins, on=RACE_KEYS + ["driver_id"]).join(losses, on=RACE_KEYS + ["driver_id"])
    df[["win", "loss"]] = df[["win", "loss"]].fillna(0)
    ordered = df.sort_values(RACE_KEYS)
    for col, out in [("win", "teammate_wins"), ("loss", "teammate_losses")]:
        df[out] = ordered.groupby("driver_id")[col].transform(lambda x: x.cumsum().shift(1, fill_value=0)).reindex(df.index)
    h2h_total = df["teammate_wins"] + df["teammate_losses"]
    df["teammate_win_pct"] = (df["teammate_wins"] / h2h_total).where(h2h_total > 0)

    # ── driver history at this circuit (previous visits only) ─────────────────
    df["career_starts_at_circuit"] = df.sort_values(RACE_KEYS).groupby(["driver_id", "circuit_id"]).cumcount().reindex(df.index)
    df["avg_finish_at_circuit"] = past_rolling(df, ["driver_id", "circuit_id"], "finish_position", None)
    df["driver_circuit_pos_change_hist"] = past_rolling(df, ["driver_id", "circuit_id"], "finish_grid_delta", None)

    # ── circuit character (previous races at this circuit only) ───────────────
    df["_delta_norm"] = df["finish_grid_delta"] / field_size
    race_level = per_race_level(df, ["circuit_id"], {"pos_change_var": ("_delta_norm", "var"),
                                                     "abs_pos_change": ("finish_grid_delta", lambda s: s.abs().mean()),
                                                     "rain": ("race_had_rain", "max")})
    race_level["rain"] = race_level["rain"].astype(float)
    race_level["circuit_pos_change_var_hist"] = past_rolling(race_level, "circuit_id", "pos_change_var", None)
    race_level["circuit_abs_pos_change_hist"] = past_rolling(race_level, "circuit_id", "abs_pos_change", None)
    race_level["circuit_wet_race_frequency"]  = past_rolling(race_level, "circuit_id", "rain", None)
    df = df.merge(race_level[RACE_KEYS + ["circuit_id", "circuit_pos_change_var_hist",
                                          "circuit_abs_pos_change_hist", "circuit_wet_race_frequency"]],
                  on=RACE_KEYS + ["circuit_id"], how="left")

    return df.drop(columns=["dnf", "finish_grid_delta", "_quali_key", "_delta_norm", "win", "loss"])


# ─────────────────────────────────────────────────────────────────────────────
# GRID PENALTIES (Jolpica API, cached in data/penalty_cache.json)
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL    = "https://api.jolpi.ca/ergast/f1"
SLEEP_BASE  = 1.5
MAX_RETRIES = 5

def fetch_with_retry(url):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 3
                print(f"    [429] waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"    [ERROR] {e}")
            return None
    return None

def get_penalty_data(year, round_num):
    q = fetch_with_retry(f"{BASE_URL}/{year}/{round_num}/qualifying.json")
    r = fetch_with_retry(f"{BASE_URL}/{year}/{round_num}/results.json")
    if not q or not r:
        return None
    q_races = q["MRData"]["RaceTable"]["Races"]
    r_races = r["MRData"]["RaceTable"]["Races"]
    if not q_races or not r_races:
        return None
    quali_pos = {x["Driver"]["code"]: int(x["position"]) for x in q_races[0].get("QualifyingResults", [])}
    grid_pos  = {}
    for x in r_races[0].get("Results", []):
        g = x.get("grid", "0")
        grid_pos[x["Driver"]["code"]] = int(g) if g != "0" else None
    out = {}
    for code in set(quali_pos) | set(grid_pos):
        q_p, g_p = quali_pos.get(code), grid_pos.get(code)
        out[code] = max(0, g_p - q_p) if (q_p and g_p) else 0
    return out

def refresh_penalties(df: pd.DataFrame) -> pd.DataFrame:
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    races = df[RACE_KEYS].drop_duplicates().sort_values(RACE_KEYS)
    print(f"Grid penalties: {len(races)} races, {len(cache)} cached")

    for _, row in races.iterrows():
        year, rnd = int(row["season"]), int(row["round"])
        key = f"{year}_{rnd}"
        if key in cache:
            continue
        print(f"    fetching {year} R{rnd}...")
        result = get_penalty_data(year, rnd)
        if result is not None:
            cache[key] = result
            CACHE_FILE.write_text(json.dumps(cache))
        time.sleep(SLEEP_BASE)

    def lookup_penalty(row):
        key = f"{int(row['season'])}_{int(row['round'])}"
        val = cache.get(key, {}).get(row["driver_id"], 0)
        if isinstance(val, dict):
            return val.get("grid_penalty_places", 0)
        return val

    df = df.copy()
    df["grid_penalty_places"] = df.apply(lookup_penalty, axis=1)
    df["had_grid_penalty"]    = df["grid_penalty_places"] > 0
    print(f"    penalised rows: {df['had_grid_penalty'].sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    base = refresh_penalties(load_base())
    print(f"Loaded {len(base)} rows")
    df = build_features(base)

    print("\n── Final null counts ──")
    print(df.isnull().sum())
    print(f"\nShape: {df.shape}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
