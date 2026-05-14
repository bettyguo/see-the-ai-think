# Architecture

`see-the-ai-think` is two things stitched together:

1. A **Python backend** that hooks a small HuggingFace causal LM and captures per-layer residuals, attention patterns, SAE-feature firings, and per-layer logit-lens predictions — then streams them over Server-Sent Events.
2. A **single-page frontend** (no build step) that renders the stream as a feature heatmap, a logit-lens scrubber, attention arcs, and a surprise ribbon.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (frontend/)                                        │
│  - index.html + ES modules                                  │
│  - heatmap (Canvas), logit lens + attention (SVG)           │
│  - SSE client; falls back to /static/demo_capture.json      │
└─────────────────────────────────────────────────────────────┘
              ▲                              ▲
              │ SSE: per-token events        │ HTTP: /feature, /examples
┌─────────────┴──────────────────────────────┴────────────────┐
│  Python backend (backend/, FastAPI)                          │
│  POST /api/generate           → text/event-stream            │
│  GET  /api/feature/.../...    → label + triggering examples  │
│  GET  /api/models, /examples, /health                        │
│                                                              │
│  CaptureEngine (models/capture.py)                           │
│  - forward hooks on each transformer block                   │
│  - per-token forward pass with KV cache                      │
│  - logit lens via final LayerNorm + lm_head                  │
│  - SAE encode + top-K via SAELens                            │
└─────────────────────────────────────────────────────────────┘
```

## Capture contract

For a tokenized prompt of length T through an L-layer transformer of width D:

| Quantity | Shape | Source |
|---|---|---|
| `tokens` | T ids + decoded text | tokenizer |
| `residuals[layer]` | `T × D` for each of L+1 layers | forward-hook on each block |
| `attn[layer]` | `H × T × T` | `output_attentions=True` |
| `sae_acts[layer]` | sparse top-K per token | SAE.encode via SAELens |
| `logit_lens[layer]` | `T × 5` (top-5 next-token probs) | unembed(LN_f(residual)) |

The streaming payload is one SSE event per token position, plus a `meta` event up front and a `done` event at the end. Per-token payload is ~10–30 KB.

## Hooks

We use raw `torch.nn.Module.register_forward_hook` (not TransformerLens) for portability. The block list is located via the architecture-specific map in `backend/models/capture.py::_block_modules`. For GPT-2 family this is `model.transformer.h[i]`. Pre-hooks capture the residual entering each block; post-hooks capture the residual leaving each block. Attention weights come from `output_attentions=True` (cheap on small models).

## SAE bundle

`backend/features/sae.py::try_load_saes` loads SAEs lazily with SAELens. **Any failure degrades gracefully** — the `SAEBundle.is_loaded` flag stays `False`, the UI shows a banner ("SAE unavailable — showing raw neuron activations"), and the capture engine falls back to raw top-K residual coordinates.

## Honesty enforcement

Every label served by `/api/feature/...` carries a `tier` field (`MEASURED` | `SOURCED` | `AUTO-LABEL`). The UI renders the appropriate badge pill. Labels are loaded from `backend/data/feature_labels.json`; a tiny in-source fallback exists for offline use.

## Performance

| Stage | Budget (CPU laptop, GPT-2 Small) |
|---|---|
| Forward pass per token | 30–60 ms |
| SAE encode (12 layers, top-K=32) | 10–20 ms |
| Logit lens (12 layers × 5) | 3–5 ms |
| JSON + SSE | <7 ms |
| **End-to-end token tick** | **<100 ms** (>10 Hz, feels live) |

The browser caps heatmap rendering at the canvas-frame budget (~16 ms) by drawing cells directly to the Canvas 2D context — no DOM nodes per cell.

## File-size discipline

Every `.py` and `.js` source file is ≤ 500 lines. Where logic threatens that limit, modules are split (e.g., `backend/server/routes.py` and `backend/server/stream.py` are deliberately separate).

## What we don't ship in v0.1

- **Feature steering** (toggle a feature, watch the output change). Tested in prototype; GPT-2 Small features don't produce dramatic-enough behavioral shifts to demo convincingly. Will revisit with Gemma-2-2B + Gemma Scope in v0.2.
- **Cross-layer feature trajectory** (visualizing how a token's representation moves through feature space). Cool but visually abstract; cut for v1.
- **Real-time auto-interp** of features the bundled snapshot doesn't cover. Auto-interp generation needs an LLM call; we link out to Neuronpedia instead.
