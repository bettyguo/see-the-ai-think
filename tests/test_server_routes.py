"""Route tests using FastAPI's TestClient. lazy=True skips model load."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import RuntimeConfig
from backend.server.app import create_app


@pytest.fixture
def client():
    app = create_app(RuntimeConfig(), lazy=True)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "model" in body


def test_models_lists_default(client: TestClient):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert any(m["name"] == "gpt2-small" and m["is_default"] for m in body)


def test_examples_route(client: TestClient):
    r = client.get("/api/examples")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) >= 3
    assert all("prompt" in e and "title" in e for e in body)


def test_feature_route_known(client: TestClient):
    r = client.get("/api/feature/gpt2-small/6/12")
    assert r.status_code == 200
    body = r.json()
    assert body["layer"] == 6
    assert body["feature"] == 12
    assert body["label"] is not None
    assert body["label"]["tier"] in {"MEASURED", "SOURCED", "AUTO-LABEL"}
    assert "honesty_note" in body


def test_feature_route_unknown_returns_null_label(client: TestClient):
    r = client.get("/api/feature/gpt2-small/99/99999")
    assert r.status_code == 200
    body = r.json()
    assert body["label"] is None
    assert body["top_corpus_examples"] == []


def test_generate_returns_503_when_engine_missing(client: TestClient):
    r = client.post("/api/generate", json={"prompt": "hi"})
    assert r.status_code == 503


def test_analyze_returns_503_when_engine_missing(client: TestClient):
    r = client.post("/api/analyze", json={"prompt": "hi"})
    assert r.status_code == 503
