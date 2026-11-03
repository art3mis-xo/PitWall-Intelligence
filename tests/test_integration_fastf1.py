"""Selectors and telemetry against real FastF1 data.

As of 2026-11-03 F1 live timing answers uncached requests with HTTP 401, so these
pass only where cache_f1/ (gitignored) already holds the sessions used here.
"""
import pytest

pytestmark = pytest.mark.integration


class TestSelectors:
    def test_years_returns_list(self, client):
        r = client.get("/api/selectors/years")
        assert r.status_code == 200
        years = r.json()["years"]
        assert isinstance(years, list)
        assert 2024 in years
        assert 2018 in years

    def test_2026_calendar_changes(self, client):
        """Saudi cancelled in 2026; Bahrain moved from its March slot to October"""
        r = client.get("/api/selectors/events/2026")
        events = {e["name"]: e for e in r.json()["events"]}
        assert "Saudi Arabian Grand Prix" not in events
        assert events["Bahrain Grand Prix"]["date"].startswith("2026-10")

    def test_drivers_are_year_correct(self, client):
        """Antonelli wasn't racing in 2022 — must not appear"""
        r = client.get("/api/selectors/drivers/2022/Monaco Grand Prix/Race")
        abbrs = [d["abbreviation"] for d in r.json()["drivers"]]
        assert "ANT" not in abbrs   # Kimi Antonelli
        assert "VER" in abbrs       # Verstappen was there

    def test_position_changes_returns_plotly(self, client):
        r = client.get(
            "/api/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "VER,LEC"}
        )
        assert r.status_code == 200
        body = r.json()
        assert "data"   in body
        assert "layout" in body
        assert len(body["data"]) >= 2   # at least one trace per driver

    def test_position_changes_respects_driver_filter(self, client):
        """Only requested drivers should appear in traces"""
        r = client.get(
            "/api/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "VER,LEC"}
        )
        trace_names = [t["name"] for t in r.json()["data"]
                       if t.get("mode") == "lines"]
        assert "VER" in trace_names
        assert "LEC" in trace_names
        assert "NOR" not in trace_names   # wasn't requested

    def test_practice_session_loads(self, client):
        """Practice sessions should work — not just Race"""
        r = client.get(
            "/api/telemetry/position-changes/2024/Japanese Grand Prix",
            params={"session": "Practice 2", "drivers": "VER,LEC"}
        )
        assert r.status_code == 200

class TestEdgeCases:
    def test_invalid_year_handled(self, client):
        r = client.get("/api/selectors/events/2010")
        # Either 200 with empty list or 500 with clear message
        # Should not hang or crash silently
        assert r.status_code in [200, 404, 500]

    def test_nonexistent_driver_returns_404(self, client):
        r = client.get(
            "/api/telemetry/position-changes/2024/Monaco Grand Prix",
            params={"session": "Race", "drivers": "ZZZ"}
        )
        assert r.status_code == 404
