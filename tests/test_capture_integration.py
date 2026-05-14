"""Integration tests against a real GPT-2 Small.

Marked `integration` — opt-in via `pytest -m integration`. CI runs the unit
suite by default; the integration suite runs on tagged releases or manual
dispatch (it downloads ~500 MB of weights).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_capture_residual_shapes(fixture_prompt, expected_shapes):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from backend.models.capture import CaptureEngine

    engine = CaptureEngine(spec_name="gpt2-small", device="cpu", enable_sae=False)
    engine.attach()
    try:
        result = engine.analyze(fixture_prompt, top_k_features=8, top_k_logits=5)
        assert result.n_layers == expected_shapes["n_layers"]
        assert len(result.tokens) == expected_shapes["tokens_in_prompt"]
        # Each token has a logit-lens entry for every layer (1 embedding + 12 blocks).
        assert len(result.tokens[0].logits_per_layer) == expected_shapes["n_layers"] + 1
    finally:
        engine.detach()


def test_capture_final_layer_predicts_dog(fixture_prompt, expected_shapes):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from backend.models.capture import CaptureEngine

    engine = CaptureEngine(spec_name="gpt2-small", device="cpu", enable_sae=False)
    engine.attach()
    try:
        result = engine.analyze(fixture_prompt, top_k_features=4, top_k_logits=5)
        final = result.tokens[-1].logits_per_layer[-1]
        assert expected_shapes["expected_top1_final"] in final.tokens
    finally:
        engine.detach()


def test_generate_stream_yields_tokens(fixture_prompt):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from backend.models.capture import CaptureEngine

    engine = CaptureEngine(spec_name="gpt2-small", device="cpu", enable_sae=False)
    engine.attach()
    try:
        captures = list(
            engine.generate_stream(
                fixture_prompt, max_new_tokens=3, top_k_features=4, top_k_logits=3, temperature=0.0
            )
        )
        assert len(captures) >= 1
        # First emitted positions cover the prompt; total >= prompt_len + max_new_tokens.
        assert captures[-1].position >= 8 + 3 - 1
    finally:
        engine.detach()
