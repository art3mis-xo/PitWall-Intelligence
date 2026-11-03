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

SEASONS = range(2018, 2026)  # adjust start year based on how far back you want


def get_season_rounds(year: int) -> pd.DataFrame:
    """Get the event schedule for a season, excluding testing."""
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["RoundNumber"] > 0]


def get_lap_data(year: int, round_num: int, event_name: str) -> pd.DataFrame:
    """Pull lap data for one race session and return a DataFrame with one row per lap per driver."""
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  [LAP DATA SKIP] {year} R{round_num} {event_name}: {e}")
        return pd.DataFrame()

    try:
        laps = session.laps.copy()
    except fastf1.exceptions.DataNotLoadedError:
        print(f"  [LAP DATA NOT LOADED] {year} R{round_num} {event_name}")
        return pd.DataFrame()

    if laps.empty:
        print(f"  [LAP DATA EMPTY] {year} R{round_num} {event_name}")
        return pd.DataFrame()

    laps["season"] = year
    laps["round"] = round_num
    laps["circuit_id"] = event_name
    laps["driver_id"] = laps["Driver"]
    laps["team_id"] = laps["Team"]

    return laps


def main():
    all_laps = []
    for year in SEASONS:
        try:
            rounds_df = get_season_rounds(year)
        except Exception as e:
            print(f"Could not load schedule for {year}: {e}")
            continue

        for _, row in rounds_df.iterrows():
            round_num = row["RoundNumber"]
            event_name = row["EventName"]
            print(f"Processing {year} R{round_num} {event_name}...")

            laps_df = get_lap_data(year, round_num, event_name)
            if not laps_df.empty:
                all_laps.append(laps_df)

            time.sleep(0.5)  # be polite to the API

    if all_laps:
        df = pd.concat(all_laps, ignore_index=True)
        out_path = OUTPUT_DIR / "lap_race_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to {out_path}")
    else:
        print("No lap data collected.")


if __name__ == "__main__":
    main()
