"""Shared fixtures.

The tests deliberately split into two layers:

  * Unit tests run **without** torch / transformers / sae-lens installed —
    they exercise the data structures, server routing, and label logic only.
  * Integration tests (marked `@pytest.mark.integration`) require a real
    GPT-2 Small download and exercise the full forward pass. They are
    opt-in: run with `pytest -m integration`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `backend.*` importable when running from a clean checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fixture_prompt() -> str:
    return (ROOT / "tests" / "fixtures" / "prompt.txt").read_text(encoding="utf-8").strip()


@pytest.fixture
def expected_shapes() -> dict:
    import json
    return json.loads(
        (ROOT / "tests" / "fixtures" / "expected_shapes.json").read_text(encoding="utf-8")
    )
