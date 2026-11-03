"""
Phase 2: Feature Engineering
Reads:  data/driver_race_results.csv   (output of 1Build_dataset_base.py)
Writes: data/features.csv              (input to Phase 3 modeling)

Run order:
  1. 1Build_dataset_base.py
  2. add_grid_penalties.py   (or run the penalty cells inline — same logic)
  3. this file

All features are derived from the base CSV + Jolpica API (penalties).
No lap-level FastF1 data is needed here — that comes in Phase 3 (FP2 pace).
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("./data")
INPUT_CSV  = DATA_DIR / "driver_race_results.csv"
OUTPUT_CSV = DATA_DIR / "features.csv"
CACHE_FILE = DATA_DIR / "penalty_cache.json"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["q1_time"] = pd.to_timedelta(df["q1_time"])
df["q2_time"] = pd.to_timedelta(df["q2_time"])
df["q3_time"] = pd.to_timedelta(df["q3_time"])
print(f"Loaded {len(df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: overtaking_difficulty  (var_delta_i_norm)
# Variance of normalised grid-to-finish delta per race.
# High variance = easy to overtake. Low = Monaco-style processional.
# ─────────────────────────────────────────────────────────────────────────────
df["delta_i"] = df["finish_position"] - df["grid_position"]

drivers_per_race = (
    df.groupby(["season", "round", "circuit_id"])["driver_id"]
      .count()
      .reset_index(name="drivers_count")
)
df2 = df.merge(drivers_per_race, on=["season", "circuit_id", "round"], how="left")
df2["delta_i_norm"] = df2["delta_i"] / df2["drivers_count"]

race_variance = (
    df2.groupby(["season", "round", "circuit_id"])["delta_i_norm"]
       .var()
       .reset_index(name="var_delta_i_norm")
)
df = df.merge(race_variance, on=["season", "round", "circuit_id"], how="left")
print(f"[1] overtaking_difficulty (var_delta_i_norm) — nulls: {df['var_delta_i_norm'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: team pace — quali-based (full 2018-2025 coverage)
# Uses quali_gap_to_pole as the pace proxy since it covers all seasons
# unlike the df_laps lap-time approach which only had 2018-2022.
# NOTE: quali_gap_to_pole is computed in Feature 7 below.
# This section is intentionally placed AFTER Feature 7 in the .py.
# In the notebook they were separate sections; here we reorder for correctness.
# ─────────────────────────────────────────────────────────────────────────────
# (computed after quali_gap_to_pole is available — see after Feature 7)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: rolling team finishing position
# ─────────────────────────────────────────────────────────────────────────────
df_avg_finish = (
    df.groupby(["season", "round", "team_id"])["finish_position"]
      .mean()
      .reset_index(name="avg_team_finish_position")
      .sort_values(["team_id", "season", "round"])
)
df_avg_finish["rolling_avg_finish_position"] = (
    df_avg_finish.groupby("team_id")["avg_team_finish_position"]
                 .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)
df = df.merge(
    df_avg_finish[["season", "round", "team_id", "avg_team_finish_position", "rolling_avg_finish_position"]],
    on=["season", "round", "team_id"], how="left"
)
print(f"[3] rolling team finish — nulls: {df['rolling_avg_finish_position'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: driver circuit history
# career_starts_at_circuit, avg_finish_at_circuit
# ─────────────────────────────────────────────────────────────────────────────
df_sorted = df.sort_values(["driver_id", "circuit_id", "season", "round"])

df["career_starts_at_circuit"] = (
    df_sorted.groupby(["driver_id", "circuit_id"]).cumcount()  # 0-indexed count before this race
)

df["avg_finish_at_circuit"] = (
    df_sorted.groupby(["driver_id", "circuit_id"])["finish_position"]
             .transform(lambda x: x.shift(1).expanding().mean())
)
print(f"[4] driver circuit history — avg_finish nulls: {df['avg_finish_at_circuit'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 5: quali-to-race delta (finish_grid_delta, avg_finish_grid_delta)
# ─────────────────────────────────────────────────────────────────────────────
df["finish_grid_delta"] = df["finish_position"] - df["grid_position"]

df_sorted2 = df.sort_values(["driver_id", "season", "round"])
df["avg_finish_grid_delta"] = (
    df_sorted2.groupby("driver_id")["finish_grid_delta"]
              .transform(lambda x: x.shift(1).expanding().mean())
)
print(f"[5] quali-to-race delta — avg nulls: {df['avg_finish_grid_delta'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 6: teammate head-to-head
# teammate_wins, teammate_losses, teammate_win_pct
# ─────────────────────────────────────────────────────────────────────────────
def teammate_result(group):
    group = group.sort_values("finish_position")
    if len(group) < 2:
        return None
    winner = group.iloc[0]["driver_id"]
    loser  = group.iloc[1]["driver_id"]
    margin = group.iloc[1]["finish_position"] - group.iloc[0]["finish_position"]
    return pd.DataFrame({
        "season":   [group.iloc[0]["season"]],
        "round":    [group.iloc[0]["round"]],
        "team_id":  [group.iloc[0]["team_id"]],
        "winner":   [winner],
        "loser":    [loser],
        "margin":   [margin],
    })

# group_keys=False prevents pandas from adding group keys to the index
results = (
    df.groupby(["season", "round", "team_id"], group_keys=False)
      .apply(teammate_result)
      .dropna()
      .reset_index(drop=True)
)

wins   = results.groupby("winner").size().reset_index(name="teammate_wins")
losses = results.groupby("loser").size().reset_index(name="teammate_losses")

driver_h2h = pd.merge(wins, losses, left_on="winner", right_on="loser", how="outer")
driver_h2h = driver_h2h.rename(columns={"winner": "driver_id"}).fillna(0)
driver_h2h["teammate_win_pct"] = (
    driver_h2h["teammate_wins"] / (driver_h2h["teammate_wins"] + driver_h2h["teammate_losses"])
)

df = df.merge(
    driver_h2h[["driver_id", "teammate_wins", "teammate_losses", "teammate_win_pct"]],
    on="driver_id", how="left"
)
print(f"[6] teammate H2H — nulls: {df['teammate_win_pct'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 7: quali gap to pole
# ─────────────────────────────────────────────────────────────────────────────
pole_times = (
    df.dropna(subset=["q3_time"])
      .groupby(["season", "round"])["q3_time"]
      .min()
      .rename("pole_q3_time")
)
df = df.join(pole_times, on=["season", "round"])
df["quali_gap_to_pole"] = (
    df["q3_time"] - df["pole_q3_time"]
).dt.total_seconds()
print(f"[7] quali_gap_to_pole — nulls: {df['quali_gap_to_pole'].isna().sum()} (expected: drivers elim in Q1/Q2)")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 (continued): team pace delta — quali-based, full coverage
# Placed here because it depends on quali_gap_to_pole (Feature 7)
# ─────────────────────────────────────────────────────────────────────────────
def rolling_slope(x):
    y = x.values
    if len(y) < 2:
        return np.nan
    return np.polyfit(np.arange(len(y)), y, 1)[0]

team_pace = (
    df.groupby(["season", "round", "team_id"])["quali_gap_to_pole"]
      .mean()
      .reset_index(name="team_avg_pace_delta")
      .sort_values(["team_id", "season", "round"])
)
team_pace["rolling_avg_pace_delta"] = (
    team_pace.groupby("team_id")["team_avg_pace_delta"]
             .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)
team_pace["pace_trend"] = (
    team_pace.groupby("team_id")["team_avg_pace_delta"]
             .transform(lambda x: x.rolling(window=3, min_periods=2).apply(rolling_slope, raw=False))
)

df = df.merge(
    team_pace[["season", "round", "team_id", "team_avg_pace_delta", "rolling_avg_pace_delta", "pace_trend"]],
    on=["season", "round", "team_id"], how="left"
)
print(f"[2] team pace delta — nulls: team_avg={df['team_avg_pace_delta'].isna().sum()}, "
      f"rolling={df['rolling_avg_pace_delta'].isna().sum()}, trend={df['pace_trend'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 8: circuit wet race frequency
# ─────────────────────────────────────────────────────────────────────────────
wet_freq = (
    df.drop_duplicates(subset=["season", "round"])
      .groupby("circuit_id")["race_had_rain"]
      .mean()
      .rename("circuit_wet_race_frequency")
)
df = df.join(wet_freq, on="circuit_id")
print(f"[8] circuit_wet_race_frequency — nulls: {df['circuit_wet_race_frequency'].isna().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 9: grid penalties (Jolpica API)
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

cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
print(f"\n[9] Grid penalties: {len(races)} races, {len(cache)} cached")

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

df["grid_penalty_places"] = df.apply(lookup_penalty, axis=1)
df["had_grid_penalty"]    = df["grid_penalty_places"] > 0
print(f"    penalised rows: {df['had_grid_penalty'].sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL: null check + save
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final null counts ──")
print(df.isnull().sum())
print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
