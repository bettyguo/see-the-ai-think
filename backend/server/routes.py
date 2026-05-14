"""HTTP routes for see-the-ai-think.

Endpoints:
  GET  /health                              liveness
  GET  /models                              available models
  GET  /examples                            pre-baked prompts
  GET  /feature/{model}/{layer}/{feature}   label + triggering examples
  POST /generate                            SSE stream of token captures
  POST /analyze                             non-streaming one-shot
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from backend.config import EXAMPLES_DIR
from backend.features.labels import load_labels
from backend.features.labels import lookup as lookup_label
from backend.features.triggers import load_triggers
from backend.features.triggers import lookup as lookup_triggers
from backend.models.registry import list_models
from backend.server.schemas import (
    AnalyzeRequest,
    FeatureDetail,
    FeatureLabelOut,
    GenerateRequest,
    MetaEvent,
    TriggerExampleOut,
)
from backend.server.stream import captures_to_sse, token_to_event

HONESTY_NOTE = (
    "This panel shows activations measured by running this model on this prompt. "
    "Any text label is auto-generated or community-sourced (see the tier badge) "
    "and may be wrong. The activation magnitudes are real; the meanings are "
    "interpretations."
)


def make_router(state: dict[str, Any]) -> APIRouter:
    """Build the router. `state` holds the (singleton) CaptureEngine + caches."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": state.get("model_name"),
            "sae_loaded": bool(state.get("engine") and state["engine"].sae_loaded),
        }

    @router.get("/models")
    def models() -> list[dict]:
        return list_models()

    @router.get("/examples")
    def examples() -> list[dict]:
        path = EXAMPLES_DIR / "prompts.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    @router.get("/feature/{model}/{layer}/{feature}")
    def feature(model: str, layer: int, feature: int) -> FeatureDetail:
        labels = state.setdefault("labels", load_labels())
        triggers = state.setdefault("triggers", load_triggers())
        label = lookup_label(labels, model, layer, feature)
        examples_ = lookup_triggers(triggers, model, layer, feature)
        return FeatureDetail(
            model=model,
            layer=layer,
            feature=feature,
            label=(
                FeatureLabelOut(text=label.text, tier=label.tier, source=label.source)
                if label is not None
                else None
            ),
            top_corpus_examples=[
                TriggerExampleOut(
                    text=t.text,
                    activating_index=t.activating_index,
                    activation=t.activation,
                )
                for t in examples_
            ],
            honesty_note=HONESTY_NOTE,
        )

    @router.post("/generate")
    async def generate(req: GenerateRequest, request: Request) -> EventSourceResponse:
        engine = state.get("engine")
        if engine is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        meta = MetaEvent(
            model=engine.spec.name,
            n_layers=engine.spec.n_layers,
            sae_loaded=engine.sae_loaded,
            notes=[] if engine.sae_loaded else ["SAE weights unavailable — showing raw neuron activations."],
        )
        cap_iter = engine.generate_stream(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            top_k_features=req.top_k_features,
            top_k_logits=req.top_k_logits,
            temperature=req.temperature,
        )
        return EventSourceResponse(captures_to_sse(meta, cap_iter))

    @router.post("/analyze")
    def analyze(req: AnalyzeRequest) -> JSONResponse:
        engine = state.get("engine")
        if engine is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        result = engine.analyze(
            prompt=req.prompt,
            top_k_features=req.top_k_features,
            top_k_logits=req.top_k_logits,
        )
        return JSONResponse(
            content={
                "model": result.model_name,
                "n_layers": result.n_layers,
                "sae_loaded": result.sae_loaded,
                "notes": result.notes,
                "tokens": [token_to_event(t).model_dump() for t in result.tokens],
            }
        )

    return router
