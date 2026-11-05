"""Deployment settings: health check, CORS, telemetry switch and Groq rate limits. No network, no Groq."""
import pytest

from pitwall.backend import main
from pitwall.backend.routers import intelligence, telemetry


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    async def fake_chat(message, history=None):
        calls.append(message)
        return "A grounded explanation."

    monkeypatch.setattr(intelligence, "chat", fake_chat)
    return calls


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://pitwall.vercel.app/, http://localhost:5173")
    assert main.cors_origins() == ["https://pitwall.vercel.app", "http://localhost:5173"]


def test_cors_allows_configured_origin_only(client):
    allowed = client.options("/api/health", headers={"Origin": "http://localhost:5173",
                                                     "Access-Control-Request-Method": "GET"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    other = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


@pytest.mark.parametrize("endpoint", ["position-changes", "tyre-strategy", "gear-shifts", "speed-traces"])
def test_telemetry_switched_off(client, monkeypatch, endpoint):
    monkeypatch.setenv("TELEMETRY_ENABLED", "false")
    r = client.get(f"/api/telemetry/{endpoint}/2024/Monaco Grand Prix")
    assert r.status_code == 503
    assert r.json()["detail"] == telemetry.TELEMETRY_DISABLED


def test_chat_per_ip_limit(client, monkeypatch, fake_llm):
    monkeypatch.setenv("GROQ_LIMIT_PER_IP", "2")
    codes = [client.post("/api/intelligence/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert len(fake_llm) == 2   # the refused request never reached the LLM


def test_chat_daily_cap_applies_across_ips(client, monkeypatch, fake_llm):
    monkeypatch.setenv("GROQ_DAILY_CAP", "2")
    codes = [client.post("/api/intelligence/chat", json={"message": "hi"},
                         headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_forged_forwarded_for_prefix_does_not_bypass_limit(client, monkeypatch, fake_llm):
    # Only the last X-Forwarded-For entry (added by the hosting proxy) identifies the client
    monkeypatch.setenv("GROQ_LIMIT_PER_IP", "1")
    first = client.post("/api/intelligence/chat", json={"message": "hi"}, headers={"X-Forwarded-For": "1.1.1.1, 9.9.9.9"})
    second = client.post("/api/intelligence/chat", json={"message": "hi"}, headers={"X-Forwarded-For": "2.2.2.2, 9.9.9.9"})
    assert (first.status_code, second.status_code) == (200, 429)


def test_explain_still_returns_factors_when_limited(client, monkeypatch, fake_llm):
    monkeypatch.setenv("GROQ_DAILY_CAP", "0")
    r = client.get("/api/intelligence/explain/2026/14/VER")
    assert r.status_code == 200
    body = r.json()
    assert "daily limit" in body["narrative"]
    assert len(body["top_factors"]) == 5
    assert fake_llm == []
