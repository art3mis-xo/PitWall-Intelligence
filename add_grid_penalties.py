"""
Phase 1 add-on: Derive grid penalty features from Jolpica F1 API.

Logic: compare a driver's qualifying position vs their actual grid position.
If grid > quali position, they took a penalty. Delta = penalty size.

Adds two columns to your existing driver_race_results.csv:
  - grid_penalty_places (int): how many places they dropped (0 = no penalty)
  - had_grid_penalty (bool): convenience flag

Run AFTER build_base_table.py.
Handles 429 rate limiting with exponential backoff + retry.
Skips races already successfully fetched (safe to re-run after interruption).
"""

import random
import requests
import pandas as pd
import time
import json
from pathlib import Path

DATA_DIR    = Path("./data")
INPUT_CSV   = DATA_DIR / "driver_race_results.csv"
OUTPUT_CSV  = DATA_DIR / "driver_race_results.csv"   # overwrites in place
CACHE_FILE  = DATA_DIR / "penalty_cache.json"        # saves progress between runs

BASE_URL    = "https://api.jolpi.ca/ergast/f1"
SLEEP_BASE  = 3.0    # seconds between normal requests — Jolpica allows ~4 req/sec but be safe
MAX_RETRIES = 8      # how many times to retry a 429 before giving up on that race


def fetch_with_retry(url: str) -> dict | None:
    """
    GET a URL, retrying on 429 with exponential backoff.
    Returns parsed JSON or None on permanent failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt * 3 + random.uniform(0, 2.0)   # 3s, 6s, 12s, 24s, 48s, 96s, 192s
                print(f"    [429] Rate limited. Waiting {wait:.1f}s before retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code != 429:
                print(f"    [HTTP ERROR] {e}")
                return None
        except Exception as e:
            print(f"    [ERROR] {e}")
            return None
    print(f"    [GIVE UP] Failed after {MAX_RETRIES} retries: {url}")
    return None


def get_quali_positions(year: int, round_num: int) -> dict:
    url  = f"{BASE_URL}/{year}/{round_num}/qualifying.json"
    data = fetch_with_retry(url)
    if not data:
        return {}
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    return {
        r["Driver"]["code"]: int(r["position"])
        for r in races[0].get("QualifyingResults", [])
    }


def get_grid_positions(year: int, round_num: int) -> dict:
    url  = f"{BASE_URL}/{year}/{round_num}/results.json"
    data = fetch_with_retry(url)
    if not data:
        return {}
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    out = {}
    for r in races[0].get("Results", []):
        code = r["Driver"]["code"]
        grid = r.get("grid", "0")
        out[code] = int(grid) if grid != "0" else None
    return out


def compute_penalties(year: int, round_num: int) -> dict | None:
    """
    Returns {driver_code: {"grid_penalty_places": int, "had_grid_penalty": bool}}.
    Returns None if either API call completely failed (so we can skip caching it).
    """
    quali_pos = get_quali_positions(year, round_num)
    grid_pos  = get_grid_positions(year, round_num)

    # if BOTH calls returned empty, something went wrong — don't cache
    if not quali_pos and not grid_pos:
        return None

    out = {}
    for code in set(quali_pos) | set(grid_pos):
        q = quali_pos.get(code)
        g = grid_pos.get(code)
        penalty = max(0, g - q) if (q is not None and g is not None) else 0
        out[code] = {
            "grid_penalty_places": penalty,
            "had_grid_penalty":    penalty > 0,
        }
    return out


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def normalise_penalty(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (int, float)):
        penalty = int(value)
        return {
            "grid_penalty_places": penalty,
            "had_grid_penalty":    penalty > 0,
        }
    return {"grid_penalty_places": 0, "had_grid_penalty": False}


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    print(f"{len(races)} unique race weekends to process")

    # load any previously cached results so we don't re-fetch on re-run
    cache = load_cache()
    print(f"Cache has {len(cache)} races already fetched\n")

    penalty_lookup = {}

    for _, row in races.iterrows():
        year      = int(row["season"])
        round_num = int(row["round"])
        cache_key = f"{year}_{round_num}"

        if cache_key in cache:
            # already fetched — just load from cache
            for driver_code, pen in cache[cache_key].items():
                penalty_lookup[(year, round_num, driver_code)] = normalise_penalty(pen)
            continue

        print(f"  Fetching {year} R{round_num}...")
        penalties = compute_penalties(year, round_num)

        if penalties is None:
            print(f"    [SKIP] Both API calls failed for {year} R{round_num} — will retry next run")
            time.sleep(SLEEP_BASE)
            continue

        # store in cache and lookup
        cache[cache_key] = penalties
        for driver_code, pen in penalties.items():
            penalty_lookup[(year, round_num, driver_code)] = normalise_penalty(pen)

        save_cache(cache)   # save after every race so progress survives interruption
        time.sleep(SLEEP_BASE)

    # map onto dataframe
    df["grid_penalty_places"] = df.apply(
        lambda r: penalty_lookup.get(
            (int(r["season"]), int(r["round"]), r["driver_id"]), {}
        )["grid_penalty_places"],
        axis=1
    )
    df["had_grid_penalty"] = df["grid_penalty_places"] > 0

    penalised = df[df["had_grid_penalty"]]
    total_races = len(cache)
    print(f"\nFound {len(penalised)} penalised driver-race rows across {total_races} successfully fetched races")
    print(penalised[["season","round","circuit_id","driver_id",
                      "grid_penalty_places","grid_position"]].head(10))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved updated table to {OUTPUT_CSV}")

    # report any races we failed to fetch
    fetched_keys = set(cache.keys())
    all_keys = {f"{int(r.season)}_{int(r.round)}" for r in races.itertuples()}
    missing = all_keys - fetched_keys
    if missing:
        print(f"\n[WARNING] {len(missing)} races not fetched (run again to retry):")
        for k in sorted(missing):
            print(f"  {k}")


if __name__ == "__main__":
    main()