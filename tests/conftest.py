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
