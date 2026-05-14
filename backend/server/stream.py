"""Translate CaptureEngine output to SSE events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from backend.models.capture import TokenCapture
from backend.server.schemas import (
    AttnTopOut,
    DoneEvent,
    FeatureHitOut,
    LogitLensOut,
    MetaEvent,
    TokenEvent,
)


def token_to_event(tc: TokenCapture) -> TokenEvent:
    return TokenEvent(
        position=tc.position,
        token_id=tc.token_id,
        text=tc.text,
        top_features=[
            FeatureHitOut(layer=h.layer, feature=h.feature_id, act=h.activation)
            for h in tc.top_features
        ],
        logits_per_layer=[
            LogitLensOut(layer=ll.layer, tokens=ll.tokens, probs=ll.probs)
            for ll in tc.logits_per_layer
        ],
        attn_top_per_layer=[
            [AttnTopOut(head=h, src=s, weight=w) for (h, s, w) in layer]
            for layer in tc.attn_top_per_layer
        ],
    )


def encode_sse(event: TokenEvent | MetaEvent | DoneEvent) -> dict[str, str]:
    """Format for sse-starlette: {'event': name, 'data': json}."""
    return {"event": event.event, "data": event.model_dump_json()}


async def captures_to_sse(
    meta: MetaEvent,
    captures: Iterator[TokenCapture],
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE events: 1 meta, N tokens, 1 done.

    The captures iterator is synchronous (generator from CaptureEngine). We
    surface it through an async generator so FastAPI's SSE handling works
    naturally without blocking the loop on long token ticks. For CPU-bound
    work, the per-tick latency budget (<100ms) is small enough that we don't
    need to thread it out.
    """
    yield encode_sse(meta)
    count = 0
    for tc in captures:
        yield encode_sse(token_to_event(tc))
        count += 1
    yield encode_sse(DoneEvent(total_tokens=count))
