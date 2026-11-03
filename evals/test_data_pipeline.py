# evals/test_data_pipeline.py
import pytest
import httpx

BASE = "http://localhost:8000"

class TestSelectors:
    def test_years_returns_list(self):
        r = httpx.get(f"{BASE}/selectors/years")
        assert r.status_code == 200
        years = r.json()["years"]
        assert isinstance(years, list)
        assert 2024 in years
        assert 2018 in years

    def test_cancelled_races_absent(self):
        """Bahrain and Saudi Arabia cancelled in 2026 — must not appear"""
        r = httpx.get(f"{BASE}/selectors/events/2026")
        names = [e["name"] for e in r.json()["events"]]
        assert "Bahrain Grand Prix"      not in names
        assert "Saudi Arabian Grand Prix" not in names

    def test_drivers_are_year_correct(self):
        """Antonelli wasn't racing in 2022 — must not appear"""
        r = httpx.get(f"{BASE}/selectors/drivers/2022/Monaco Grand Prix/Race")
        abbrs = [d["abbreviation"] for d in r.json()["drivers"]]
        assert "ANT" not in abbrs   # Kimi Antonelli
        assert "VER" in abbrs       # Verstappen was there

    def test_position_changes_returns_plotly(self):
        r = httpx.get(
            f"{BASE}/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "VER,LEC"}
        )
        assert r.status_code == 200
        body = r.json()
        assert "data"   in body
        assert "layout" in body
        assert len(body["data"]) >= 2   # at least one trace per driver

    def test_position_changes_respects_driver_filter(self):
        """Only requested drivers should appear in traces"""
        r = httpx.get(
            f"{BASE}/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "VER,LEC"}
        )
        trace_names = [t["name"] for t in r.json()["data"]
                       if t.get("mode") == "lines"]
        assert "VER" in trace_names
        assert "LEC" in trace_names
        assert "NOR" not in trace_names   # wasn't requested

    def test_practice_session_loads(self):
        """Practice sessions should work — not just Race"""
        r = httpx.get(
            f"{BASE}/telemetry/position-changes/2023/Japanese Grand Prix",
            params={"session": "Practice 1", "drivers": "VER,LEC"}
        )
        assert r.status_code == 200

class TestEdgeCases:
    def test_invalid_year_handled(self):
        r = httpx.get(f"{BASE}/selectors/events/2010")
        # Either 200 with empty list or 500 with clear message
        # Should not hang or crash silently
        assert r.status_code in [200, 404, 500]

    def test_nonexistent_driver_returns_404(self):
        r = httpx.get(
            f"{BASE}/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "ZZZ"}
        )
        assert r.status_code == 404