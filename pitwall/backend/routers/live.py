"""
routers/live.py
Live F1 data endpoints — fetches current season data at request time.
  GET /live/next-race          — next upcoming race from FastF1 schedule
  GET /live/standings/{season} — WDC + constructors championship from Jolpica
  GET /live/last-race/{season} — full results of most recent completed race
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
import fastf1
import requests
import time

router = APIRouter(prefix="/api/live", tags=["Live"])

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
SLEEP        = 0.5   # polite delay between Jolpica calls

# ── helpers ────────────────────────────────────────────────────────────────

def jolpica_get(url: str) -> dict:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── next race ──────────────────────────────────────────────────────────────

@router.get("/next-race")
async def get_next_race():
    """
    Returns the next upcoming race based on today's date.
    Pulls from FastF1 event schedule for the current year.
    Falls back to next year if current season is over.
    """
    try:
        now  = datetime.now(timezone.utc)
        year = now.year

        schedule = fastf1.get_event_schedule(year, include_testing=False)

        # find first event whose date is in the future
        upcoming = None
        for _, row in schedule.iterrows():
            event_date = row["EventDate"]
            # make timezone-aware if needed
            if event_date.tzinfo is None:
                import pandas as pd
                event_date = event_date.tz_localize("UTC")
            if event_date > now:
                upcoming = row
                break

        # if season is over, peek at next year
        if upcoming is None:
            schedule = fastf1.get_event_schedule(year + 1, include_testing=False)
            upcoming = schedule.iloc[0]

        # try to get race session datetime (more precise than EventDate)
        try:
            event     = fastf1.get_event(int(upcoming["RoundNumber"]) and year or year + 1,
                                          int(upcoming["RoundNumber"]))
            race_time = event.get_session("Race").date
            if race_time.tzinfo is None:
                race_time = race_time.replace(tzinfo=timezone.utc)
            race_iso  = race_time.isoformat()
        except Exception:
            race_iso = str(upcoming["EventDate"].date()) + "T13:00:00Z"

        return {
            "season":     year,
            "round":      int(upcoming["RoundNumber"]),
            "name":       upcoming["EventName"],
            "circuit":    upcoming["Location"],
            "country":    upcoming["Country"],
            "race_date":  race_iso,
            "format":     upcoming.get("EventFormat", "conventional"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── championship standings ──────────────────────────────────────────────────

@router.get("/standings/{season}")
async def get_standings(season: int):
    """
    Returns current WDC and constructors championship standings
    for the given season, fetched live from Jolpica.
    """
    try:
        # drivers
        d_data    = jolpica_get(f"{JOLPICA_BASE}/{season}/driverStandings.json")
        d_list    = d_data["MRData"]["StandingsTable"]["StandingsLists"]
        drivers   = []
        if d_list:
            for entry in d_list[0]["DriverStandings"]:
                drv = entry["Driver"]
                con = entry["Constructors"][0] if entry["Constructors"] else {}
                drivers.append({
                    "position":   int(entry["position"]),
                    "driver_id":  drv.get("code", drv["driverId"].upper()[:3]),
                    "full_name":  f"{drv['givenName']} {drv['familyName']}",
                    "team_id":    con.get("name", ""),
                    "points":     float(entry["points"]),
                    "wins":       int(entry["wins"]),
                })

        time.sleep(SLEEP)

        # constructors
        c_data  = jolpica_get(f"{JOLPICA_BASE}/{season}/constructorStandings.json")
        c_list  = c_data["MRData"]["StandingsTable"]["StandingsLists"]
        constructors = []
        if c_list:
            for entry in c_list[0]["ConstructorStandings"]:
                con = entry["Constructor"]
                constructors.append({
                    "position": int(entry["position"]),
                    "team_id":  con.get("name", con["constructorId"]),
                    "points":   float(entry["points"]),
                    "wins":     int(entry["wins"]),
                })

        return {
            "season":       season,
            "drivers":      drivers,
            "constructors": constructors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── last race results ───────────────────────────────────────────────────────

@router.get("/last-race/{season}")
async def get_last_race(season: int):
    """
    Returns full finishing order of the most recently completed race,
    including DNFs and classified status. Fetched from Jolpica.
    """
    try:
        data  = jolpica_get(f"{JOLPICA_BASE}/{season}/last/results.json")
        races = data["MRData"]["RaceTable"]["Races"]

        if not races:
            raise HTTPException(
                status_code=404,
                detail=f"No completed races found for {season}"
            )

        race    = races[0]
        results = []

        for r in race["Results"]:
            drv = r["Driver"]
            con = r["Constructor"]
            fastest = r.get("FastestLap", {})

            results.append({
                "position":        r.get("position", "NC"),
                "classified_pos":  r.get("positionText", "NC"),   # NC = not classified
                "driver_id":       drv.get("code", drv["driverId"].upper()[:3]),
                "full_name":       f"{drv['givenName']} {drv['familyName']}",
                "team_id":         con.get("name", ""),
                "grid":            int(r.get("grid", 0)),
                "laps":            int(r.get("laps", 0)),
                "status":          r.get("status", ""),           # "Finished", "+1 Lap", "DNF" etc
                "points":          float(r.get("points", 0)),
                "time":            r.get("Time", {}).get("time", None),
                "fastest_lap":     fastest.get("Time", {}).get("time", None),
                "fastest_lap_rank": int(fastest.get("rank", 0)) if fastest else None,
            })

        return {
            "season":       int(race["season"]),
            "round":        int(race["round"]),
            "race_name":    race["raceName"],
            "circuit":      race["Circuit"]["circuitName"],
            "date":         race["date"],
            "results":      results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))