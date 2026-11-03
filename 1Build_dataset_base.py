"""
Phase 1: Build the base driver-race results table from FastF1.
Run this LOCALLY (not in a sandboxed env) since it needs to hit F1's live timing API.

Output: data/driver_race_results.csv
One row = one driver's result at one race.
"""

import random
import fastf1
import pandas as pd
import time
from pathlib import Path

# --- setup ---
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

SEASONS = range(2018, 2027)  # adjust start year based on how far back you want
MAX_RETRIES = 6
RETRY_BASE_DELAY = 3.0


def get_season_rounds(year: int) -> pd.DataFrame:
    """Get the event schedule for a season, excluding testing."""
    for attempt in range(MAX_RETRIES):
        try:
            schedule = fastf1.get_event_schedule(year)
            # drop pre-season testing rows (RoundNumber == 0)
            return schedule[schedule["RoundNumber"] > 0]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1.5)
                print(f"  [SCHEDULE RETRY {attempt+1}/{MAX_RETRIES}] {year}: {e} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                raise


def get_quali_times(year: int, round_num: int, event_name: str) -> dict:
    """
    Pull Q1/Q2/Q3 times from the actual Qualifying session.
    Race-session Q1/Q2/Q3 columns are unreliable/often empty, so we
    fetch them from the 'Q' session and join by driver abbreviation.
    Returns {driver_abbreviation: {"Q1": ..., "Q2": ..., "Q3": ...}}.
    """
    for attempt in range(MAX_RETRIES):
        try:
            quali = fastf1.get_session(year, round_num, "Q")
            quali.load(laps=False, telemetry=False, weather=False, messages=False)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1.5)
                print(f"  [QUALI RETRY {attempt+1}/{MAX_RETRIES}] {year} R{round_num} {event_name}: {e} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"  [QUALI SKIP] {year} R{round_num} {event_name}: {e}")
                return {}

    try:
        q_results = quali.results
    except Exception:
        q_results = None

    if q_results is None or q_results.empty:
        print(f"  [QUALI EMPTY] {year} R{round_num} {event_name}")
        return {}

    out = {}
    for _, r in q_results.iterrows():
        out[r.get("Abbreviation")] = {
            "Q1": r.get("Q1"),
            "Q2": r.get("Q2"),
            "Q3": r.get("Q3"),
        }
    return out


def extract_race_rows(year: int, round_num: int, event_name: str) -> list[dict]:
    """Pull one race session (+ matching quali session) and return one dict per driver."""
    rows = []
    for attempt in range(MAX_RETRIES):
        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(laps=False, telemetry=False, weather=True, messages=False)
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1.5)
                print(f"  [RACE RETRY {attempt+1}/{MAX_RETRIES}] {year} R{round_num} {event_name}: {e} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"  [SKIP] {year} R{round_num} {event_name}: {e}")
                return rows

    try:
        results = session.results
    except Exception:
        results = None
    if results is None or results.empty:
        print(f"  [EMPTY] {year} R{round_num} {event_name}")
        return rows

    # weather summary for the race session (averaged across the session)
    weather = session.weather_data
    avg_air_temp = weather["AirTemp"].mean() if weather is not None and not weather.empty else None
    avg_track_temp = weather["TrackTemp"].mean() if weather is not None and not weather.empty else None
    rain_flag = bool(weather["Rainfall"].any()) if weather is not None and not weather.empty else None

    # pull quali times separately and join by driver abbreviation
    quali_times = get_quali_times(year, round_num, event_name)

    for _, r in results.iterrows():
        driver_abbr = r.get("Abbreviation")
        q = quali_times.get(driver_abbr, {})

        rows.append({
            "season": year,
            "round": round_num,
            "circuit_id": event_name,
            "driver_id": driver_abbr,
            "driver_full_name": r.get("FullName"),
            "team_id": r.get("TeamName"),
            "grid_position": r.get("GridPosition"),
            "finish_position": r.get("Position"),
            "classified_status": r.get("Status"),
            "points": r.get("Points"),
            "q1_time": q.get("Q1"),
            "q2_time": q.get("Q2"),
            "q3_time": q.get("Q3"),
            "race_avg_air_temp": avg_air_temp,
            "race_avg_track_temp": avg_track_temp,
            "race_had_rain": rain_flag,
        })

    return rows


def main():
    all_rows = []

    for year in SEASONS:
        print(f"\n=== Season {year} ===")
        try:
            schedule = get_season_rounds(year)
        except Exception as e:
            print(f"  Could not load schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            round_num = event["RoundNumber"]
            event_name = event["EventName"]
            print(f" Round {round_num}: {event_name}")

            rows = extract_race_rows(year, round_num, event_name)
            all_rows.extend(rows)

            time.sleep(1.5 + random.uniform(0, 0.8))  # be polite to the API

    df = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "driver_race_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()