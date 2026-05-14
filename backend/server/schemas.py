"""Pydantic request/response schemas. Keep flat — they double as the SSE wire format."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k_features: int = Field(default=32, ge=1, le=128)
    top_k_logits: int = Field(default=5, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    top_k_features: int = Field(default=32, ge=1, le=128)
    top_k_logits: int = Field(default=5, ge=1, le=20)


class FeatureHitOut(BaseModel):
    layer: int
    feature: int
    act: float


class LogitLensOut(BaseModel):
    layer: int
    tokens: list[str]
    probs: list[float]


class AttnTopOut(BaseModel):
    head: int
    src: int
    weight: float


class TokenEvent(BaseModel):
    """One SSE event payload for one position."""

    event: Literal["token"] = "token"
    position: int
    token_id: int
    text: str
    top_features: list[FeatureHitOut]
    logits_per_layer: list[LogitLensOut]
    attn_top_per_layer: list[list[AttnTopOut]]


class MetaEvent(BaseModel):
    event: Literal["meta"] = "meta"
    model: str
    n_layers: int
    sae_loaded: bool
    notes: list[str]


class DoneEvent(BaseModel):
    event: Literal["done"] = "done"
    total_tokens: int


class FeatureLabelOut(BaseModel):
    text: str
    tier: Literal["MEASURED", "SOURCED", "AUTO-LABEL"]
    source: str


class TriggerExampleOut(BaseModel):
    text: str
    activating_index: int
    activation: float


class FeatureDetail(BaseModel):
    model: str
    layer: int
    feature: int
    label: FeatureLabelOut | None
    top_corpus_examples: list[TriggerExampleOut]
    honesty_note: str
