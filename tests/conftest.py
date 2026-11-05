import os

# The agent module builds a Groq client at import time. A placeholder key lets it
# import without a real one, and any test that accidentally reaches Groq fails on
# auth instead of spending quota. load_dotenv() does not override existing vars.
os.environ["GROQ_API_KEY"] = "test-key-not-used"

import pytest
from fastapi.testclient import TestClient

from pitwall.backend.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


EVENT_NAMES = {(2026, 15): "Azerbaijan Grand Prix", (2026, 20): "São Paulo Grand Prix"}


@pytest.fixture(autouse=True)
def offline_event_names(monkeypatch):
    # Upcoming-race predictions look up the circuit in the FastF1 schedule (network)
    from pitwall.backend.services import race_predictor
    monkeypatch.setattr(race_predictor, "_event_name",
                        lambda season, rnd: EVENT_NAMES.get((season, rnd), f"Round {rnd}"))


@pytest.fixture(autouse=True)
def fresh_rate_limits():
    from pitwall.backend.services.rate_limit import limiter
    limiter.reset()
