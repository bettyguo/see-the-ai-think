"""The frontend is static assets served from the FastAPI app — make sure they
all load and that demo_capture.json parses to the schema the UI expects.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT, RuntimeConfig
from backend.server.app import create_app


@pytest.fixture
def client():
    app = create_app(RuntimeConfig(), lazy=True)
    with TestClient(app) as c:
        yield c


STATIC_FILES = [
    "/static/app.js",
    "/static/styles.css",
    "/static/heatmap.js",
    "/static/logit_lens.js",
    "/static/feature_panel.js",
    "/static/demo_capture.json",
    "/static/assets/favicon.svg",
    "/static/index.html",
]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "see-the-ai-think" in r.text


@pytest.mark.parametrize("path", STATIC_FILES)
def test_static_assets_reachable(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} → {r.status_code}"


def test_demo_capture_shape():
    path = PROJECT_ROOT / "frontend" / "demo_capture.json"
    assert path.exists(), "frontend/demo_capture.json missing — run `python -m backend.demo` to generate"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "meta" in data and "tokens" in data
    assert data["meta"]["model"] == "gpt2-small"
    assert data["meta"]["n_layers"] == 12
    assert isinstance(data["tokens"], list) and len(data["tokens"]) >= 8
    first = data["tokens"][0]
    assert {"position", "token_id", "text", "top_features", "logits_per_layer", "attn_top_per_layer"} <= set(first.keys())
    # The logit-lens scrubber expects one entry per layer including embedding (n_layers + 1).
    assert len(first["logits_per_layer"]) == data["meta"]["n_layers"] + 1
    # Top features should sort by activation magnitude.
    acts = [f["act"] for f in first["top_features"]]
    assert acts == sorted(acts, reverse=True)
