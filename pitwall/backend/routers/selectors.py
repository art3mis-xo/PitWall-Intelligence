# pitwall/backend/routers/selectors.py
from fastapi import APIRouter, HTTPException
import fastf1
import pandas as pd
from ..services import FastF1Service

router = APIRouter(prefix="/api/selectors", tags=["Selectors"])
f1_service = FastF1Service()


@router.get("/years")
async def get_years():
    """Returns available seasons — FastF1 has data from 2018 onwards for telemetry"""
    return {"years": list(range(2024, 2017, -1))}  # 2024 down to 2018


@router.get("/events/{year}")
async def get_events(year: int):
    """Returns all race weekends for a given year — from actual F1 calendar"""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = []
        for _, row in schedule.iterrows():
            events.append({
                "round":      int(row["RoundNumber"]),
                "name":       row["EventName"],
                "country":    row["Country"],
                "location":   row["Location"],
                "date":       str(row["EventDate"].date()),
                "format":     row["EventFormat"],  # conventional or sprint_shootout
            })
        return {"events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{year}/{round}")
async def get_sessions(year: int, round: int):
    """Returns available session types for a specific event"""
    try:
        event = fastf1.get_event(year, round)
        sessions = []

        # FastF1 stores sessions as Session1..Session5
        for i in range(1, 6):
            session_name = event.get(f"Session{i}")
            if session_name and session_name != "None" and pd.notna(session_name):
                sessions.append({
                    "number": i,
                    "name":   session_name,   # e.g. "Practice 1", "Qualifying", "Race"
                })
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drivers/{year}/{event}/{session}")
async def get_drivers(year: int, event: str, session: str):
    """Returns drivers who actually participated in this specific session"""
    try:
        s = await f1_service.get_session(year, event, session)
        drivers = []
        for drv in s.drivers:
            drv_info = s.get_driver(drv)
            drivers.append({
                "number":       drv,
                "abbreviation": drv_info["Abbreviation"],
                "full_name":    drv_info["FullName"],
                "team":         drv_info["TeamName"],
                "team_color":   "#" + drv_info["TeamColor"],
            })
        # Sort by driver number
        drivers.sort(key=lambda x: int(x["number"]))
        return {"drivers": drivers}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")   # ← add this
        import traceback
        traceback.print_exc()                       # ← and this — prints full stack
        raise HTTPException(status_code=500, detail=str(e))
