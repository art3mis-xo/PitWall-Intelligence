# pitwall/backend/services/fastf1_services.py
import asyncio
import fastf1
import pandas as pd
from pathlib import Path
from functools import lru_cache
try:
    from fastf1.exceptions import DataNotLoadedError
except ImportError:
    # Fallback for older FastF1 versions
    from fastf1.core import DataNotLoadedError

CACHE_PATH = Path(__file__).resolve().parents[3] / "cache_f1"

fastf1.Cache.enable_cache(str(CACHE_PATH))


class FastF1Service:
    _session_cache: dict = {}
    _load_locks: dict = {}

    async def get_session(self, year: int, gp: str, session_type: str):
        key = f"{year}_{gp}_{session_type}"
        
        if key in self._session_cache:
            return self._session_cache[key]

        if key not in self._load_locks:
            self._load_locks[key] = asyncio.Lock()

        async with self._load_locks[key]:
            # Double check after acquiring lock
            if key in self._session_cache:
                return self._session_cache[key]

            def _load():
                s = fastf1.get_session(year, gp, session_type)
                s.load(telemetry=True, laps=True, weather=False)
                return s
            
            self._session_cache[key] = await asyncio.to_thread(_load)
            return self._session_cache[key]

    async def get_race_laps(self, year: int, gp: str,
                            session_type: str = "R") -> pd.DataFrame:
        s = await self.get_session(year, gp, session_type)
        laps = s.laps.copy()
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        return laps

    async def get_telemetry(self, year: int, gp: str,
                                driver: str, session_type: str = 'Q'):
        s = await self.get_session(year, gp, session_type)
        try:
            lap = s.laps.pick_drivers(driver).pick_fastest()
            tel = lap.get_car_data().add_distance()
            return tel
        except DataNotLoadedError:
            return pd.DataFrame()

    async def get_schedule(self, year: int):
        return fastf1.get_event_schedule(year)

    async def get_stints(self, year: int, gp: str, session_type: str) -> pd.DataFrame:
        s = await self.get_session(year, gp, session_type)
        # Stints are computed by grouping laps by Driver and Stint number
        stints = s.laps.groupby(["Driver", "Stint", "Compound"]).agg(
            StintLength=("LapNumber", "count")
        ).reset_index()
        return stints