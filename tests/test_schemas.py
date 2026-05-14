"""Schema round-trips. No torch required."""

from __future__ import annotations

import json

from backend.server.schemas import (
    AttnTopOut,
    DoneEvent,
    FeatureHitOut,
    FeatureLabelOut,
    LogitLensOut,
    MetaEvent,
    TokenEvent,
)


def test_token_event_serializes_with_event_tag():
    ev = TokenEvent(
        position=0,
        token_id=42,
        text="The",
        top_features=[FeatureHitOut(layer=6, feature=12, act=3.4)],
        logits_per_layer=[LogitLensOut(layer=11, tokens=[" dog"], probs=[0.42])],
        attn_top_per_layer=[[AttnTopOut(head=0, src=0, weight=0.9)]],
    )
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "token"
    assert payload["top_features"][0]["feature"] == 12
    assert payload["logits_per_layer"][0]["tokens"] == [" dog"]


def test_meta_event_carries_sae_flag():
    meta = MetaEvent(model="gpt2-small", n_layers=12, sae_loaded=False, notes=["fallback"])
    payload = json.loads(meta.model_dump_json())
    assert payload["sae_loaded"] is False
    assert "fallback" in payload["notes"]


def test_done_event_has_total():
    ev = DoneEvent(total_tokens=10)
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "done"
    assert payload["total_tokens"] == 10


def test_feature_label_tier_constrained():
    fl = FeatureLabelOut(text="x", tier="AUTO-LABEL", source="src")
    assert fl.tier == "AUTO-LABEL"
