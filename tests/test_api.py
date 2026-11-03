"""HTTP layer through FastAPI's TestClient. No network, no Groq calls."""
import pytest
import requests

from pitwall.backend.routers import intelligence, telemetry
from pitwall.backend.services import news
from pitwall.backend.services.fastf1_services import DataNotLoadedError


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint(client):
    r = client.get("/api/predict/2026/14")
    assert r.status_code == 200
    predictions = r.json()["predictions"]
    assert predictions[0]["predicted_rank"] == 1
    assert {f["direction"] for p in predictions for f in p["top_factors"]} <= {"improves", "worsens"}


def test_explain_endpoint_grounds_llm_narrative(client, monkeypatch):
    async def fake_chat(message, history=None):
        return "Starts from pole position and converts it."

    monkeypatch.setattr(intelligence, "chat", fake_chat)
    predictions = client.get("/api/predict/2026/14").json()["predictions"]
    driver = next(p for p in predictions if p["grid_position"] != 1)

    r = client.get(f"/api/intelligence/explain/2026/14/{driver['driver_id']}")
    assert r.status_code == 200
    body = r.json()
    assert "pole position" not in body["narrative"].lower()
    assert body["top_factors"] == driver["top_factors"]


def test_explain_endpoint_falls_back_when_llm_fails(client, monkeypatch):
    async def broken_chat(message, history=None):
        raise RuntimeError("Groq unavailable")

    monkeypatch.setattr(intelligence, "chat", broken_chat)
    r = client.get("/api/intelligence/explain/2026/14/VER")
    assert r.status_code == 200
    assert "unavailable" in r.json()["narrative"]


def test_explain_unknown_driver_is_404(client):
    r = client.get("/api/intelligence/explain/2026/14/ZZZ")
    assert r.status_code == 404


def test_ground_narrative_keeps_pole_claim_only_for_p1():
    text = "A pole position and front-row start."
    assert intelligence.ground_narrative(text, 1) == text
    grounded = intelligence.ground_narrative(text, 4).lower()
    assert "pole position" not in grounded
    assert "front-row" not in grounded


@pytest.mark.parametrize("endpoint", ["position-changes", "tyre-strategy", "gear-shifts", "speed-traces"])
def test_telemetry_returns_503_when_live_timing_has_no_data(client, monkeypatch, endpoint):
    async def no_data(*args, **kwargs):
        raise DataNotLoadedError("The data you are trying to access has not been loaded yet.")

    monkeypatch.setattr(telemetry.f1_service, "get_session", no_data)
    r = client.get(f"/api/telemetry/{endpoint}/2024/Monaco Grand Prix", params={"session": "Practice 1"})
    assert r.status_code == 503
    assert r.json()["detail"] == telemetry.SESSION_UNAVAILABLE


def test_news_survives_all_feeds_failing(client, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(news.requests, "get", offline)
    monkeypatch.setitem(news._cache, "fetched_at", 0.0)
    r = client.get("/api/intelligence/news")
    assert r.status_code == 200
    assert r.json()["articles"] == []
