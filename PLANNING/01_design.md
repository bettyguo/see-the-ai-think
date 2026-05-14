# PHASE 1 — DESIGN

> Architecture and contracts. Implementation begins in Phase 2.
> Date: 2026-05-14. Author: Betty Guo.

---

## 1. Repo layout

```
see-the-ai-think/
├── README.md                       # GIF first, then one-sentence pitch, then 1-command quickstart
├── LICENSE                         # MIT, © 2026 Dongxin Guo (Betty Guo)
├── CONTRIBUTING.md
├── pyproject.toml                  # pinned deps; entry point `see-the-ai-think`
├── Makefile                        # `make run` — the one command
├── run.sh                          # POSIX one-command (calls Makefile)
├── run.ps1                         # Windows one-command
│
├── backend/
│   ├── __init__.py
│   ├── config.py                   # paths, model registry, defaults
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py             # name → (loader, default_device)
│   │   ├── hf_loader.py            # HuggingFace transformers loader + tokenizer
│   │   └── capture.py              # forward hooks → per-layer residual + attention
│   ├── features/
│   │   ├── __init__.py
│   │   ├── sae.py                  # SAE loading + top-K feature encoding
│   │   ├── logit_lens.py           # per-layer unembed → top-K next-token probs
│   │   ├── labels.py               # bundled feature labels (Neuronpedia snapshot)
│   │   └── triggers.py             # indexed triggering-examples corpus lookup
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app factory
│   │   ├── routes.py               # /generate (SSE), /feature, /models, /examples, /health
│   │   ├── schemas.py              # pydantic request/response models
│   │   └── stream.py               # async per-token streaming pipeline
│   └── __main__.py                 # `python -m backend` launches uvicorn
│
├── frontend/
│   ├── index.html                  # single-page, no build step
│   ├── app.js                      # SSE client + state, ~300 lines
│   ├── heatmap.js                  # Canvas SAE-firing heatmap
│   ├── logit_lens.js               # SVG layer scrubber
│   ├── feature_panel.js            # right-side panel (top tokens, label, badge)
│   ├── attention.js                # Phase 4 — attention grid
│   ├── steering.js                 # Phase 4 — feature steering controls
│   ├── styles.css                  # dark theme, designed to screenshot well
│   └── assets/
│       └── favicon.svg
│
├── data/                           # NOT committed; populated by first-run setup
│   └── README.md                   # explains the cache layout
│
├── examples/
│   └── prompts.json                # pre-baked prompts (see §9)
│
├── assets/
│   ├── logo.svg
│   ├── demo.gif                    # the hero (recorded manually — see RECORD_DEMO.md)
│   └── RECORD_DEMO.md              # exact shot list for the demo recording
│
├── tests/
│   ├── conftest.py
│   ├── test_capture.py             # backbone hooks return correct shapes for fixture prompt
│   ├── test_logit_lens.py
│   ├── test_sae.py                 # SAE top-K stable for fixture
│   ├── test_server_routes.py       # /generate, /feature, /examples
│   ├── test_stream.py              # SSE payload shape + ordering
│   └── fixtures/
│       ├── prompt.txt              # "The quick brown fox jumps over the lazy"
│       └── expected_shapes.json    # shape contract for capture
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── HOSTED_DEMO.md              # HF Spaces / Fly.io recipe
│   ├── HONESTY.md                  # the rules from PHASE 0 §5
│   ├── LAUNCH.md                   # Show HN, X, reddit, newsletters
│   └── PROFILE_SNIPPET.md          # for bettyguo's GitHub profile README
│
└── .github/
    └── workflows/
        ├── ci.yml                  # pytest, lint, smoke-test one-command-start
        └── release.yml             # tag → wheel + container (deferred)
```

**Source-file size rule:** every `.py` and `.js` ≤ 500 lines. Where logic threatens to exceed it, split (e.g., `routes.py` and `stream.py` are deliberately separate).

---

## 2. Capture contract

The backend must, for one forward pass on a tokenized prompt of length T tokens through an L-layer Transformer with H heads and D residual width:

| Quantity | Shape | Source |
|---|---|---|
| `tokens` | `T` ids + decoded strings | tokenizer |
| `residuals[layer]` | `T × D` per layer 0..L (incl. embedding = "layer 0") | forward hook on each block's residual-stream input |
| `attn[layer]` | `H × T × T` per layer | forward hook on attention weights output |
| `sae_acts[layer]` | sparse `T × F_layer` — only top-K per token | SAE encode of residual, then top-K mask |
| `logit_lens[layer]` | `T × 5` (top-5 token ids + probs) per layer | unembed(LN(residual)) → softmax → topk |

### `CaptureResult` (Python dataclass)

```python
@dataclass
class TokenCapture:
    token_id: int
    text: str
    top_features: list[FeatureHit]      # cross-layer top 32, sorted by act
    logits_per_layer: list[list[tuple[str, float]]]  # L × 5
    attn_top_per_layer: list[list[tuple[int,int,float]]]  # sparse summary

@dataclass
class FeatureHit:
    layer: int
    feature_id: int
    activation: float

@dataclass
class CaptureResult:
    model_name: str
    prompt: str
    tokens: list[TokenCapture]
    n_layers: int
    sae_loaded: bool                    # False → UI shows graceful degradation banner
```

### Hooks implementation rule

Use **raw `torch.nn.Module.register_forward_hook`** on the HF model — no TransformerLens dependency. Locate residual injection points by walking `model.transformer.h[i]` for GPT-2 family; provide a per-architecture map in `backend/models/registry.py`. SAE encode is done outside the hook to keep the forward pass clean.

### Streaming vs batched

For autoregressive generation we run **one token at a time** so the UI sees the actual stream. Each generated token triggers a new forward pass with the KV cache — captures are emitted before sampling the next token. For pure "analyze this prompt" mode (no generation), we do one batched forward.

### Fixture contract

`tests/fixtures/prompt.txt` = `"The quick brown fox jumps over the lazy"` (8 tokens with GPT-2 BPE). Expected:
- `n_layers == 12`
- `residuals[0].shape == (8, 768)`, `residuals[12].shape == (8, 768)`
- `attn[0].shape == (12, 8, 8)`
- top-1 logit-lens prediction at final layer's last position is `" dog"` (well-known GPT-2 completion)
- SAE acts present at all 12 layers if SAE weights downloaded; `sae_loaded=False` otherwise

Tests assert these shapes plus the `" dog"` prediction (regression catch).

---

## 3. UI interaction contracts (LAUNCH-CRITICAL set)

### Interaction A — Live SAE feature firing heatmap

**User action:** Types a prompt, presses ⌘↵ (or clicks **Run**).

**What they see:**
- The prompt appears as a horizontal row of token chips (already tokenized, BPE pieces visible on hover).
- Below it, a **heatmap canvas**: rows = top-N (32) most-active features across the prompt, columns = tokens. Cells colored by activation magnitude (perceptual viridis-like ramp, dark theme: `#0b0d14` → `#3a86ff` → `#ffd166`).
- As generation streams, new columns slide in from the right and the row ranking re-sorts smoothly (250ms ease) when a new feature enters the top-32.
- Each row's left label shows `L{layer}/F{feature}` plus its short auto-label (if available) — with the appropriate badge pill (`MEASURED`/`AUTO-LABEL`/`SOURCED`).
- Hovering a cell shows tooltip: token, feature, activation, layer.

**Data needed (per token, SSE payload):** `top_features` list (≤32) + `tokens` cumulative.

**Animation rule:** ≤16ms per frame. Use Canvas, not DOM cells. Re-rank with a CSS-transform fallback when rows reorder.

### Interaction B — Click-a-feature → triggering tokens + label

**User action:** Clicks any row label or cell in the heatmap.

**What they see:** A right-side panel slides in (300ms) showing:
- Header: `L8 · Feature 3127`, badge pill for label trust tier.
- Auto-label (if any), with citation: e.g., "Neuronpedia auto-interp · model gpt2-small · 2024-12 snapshot". A `Speculative — auto-generated` notice in small text.
- "Top 10 activating tokens (in this prompt)" — list of (token, activation) bars.
- "Top 10 activating examples (corpus)" — short text snippets from the bundled triggering corpus with the activating token highlighted.
- "What this means" expander: a fixed honesty paragraph (we never invent meanings).

**Data needed:** `GET /feature/{layer}/{feature_id}` returns:
```json
{
  "layer": 8, "feature_id": 3127,
  "label": {"text": "...", "source": "neuronpedia", "tier": "AUTO-LABEL"} | null,
  "top_corpus_examples": [{"text": "...", "tokens": [...], "acts": [...]}],
  "top_in_session": [...]   // computed live from current capture
}
```

**Animation rule:** Panel uses `transform: translateX(...)` + opacity; no layout thrash.

### Interaction C — Layer-by-layer logit lens scrubber

**User action:** Drags a horizontal slider labeled "Layer 0 → 12" below the heatmap.

**What they see:** A row of token chips representing the **most recently generated position**. Above each chip, a vertical stack of the top-5 next-token predictions from the logit-lens at the slider's layer — bars sized by probability, color faded by rank. As the user drags, predictions morph through layers in real time. The final layer (slider rightmost) shows the model's actual final prediction.

**Data needed:** `logits_per_layer` (already in the SSE payload).

**Animation rule:** Slider drives a CSS variable; bars transition `width 100ms`.

---

## 4. Performance contract

**Targets (CPU laptop, GPT-2 Small, ≤T=64 tokens):**

| Stage | Budget | Implementation |
|---|---|---|
| Forward pass per generated token | 30–60ms | HF `transformers` with `use_cache=True` |
| SAE encode (12 layers, top-32 features) | 10–20ms | precomputed encoder weight matmul; sparse top-K via `torch.topk` |
| Logit lens (12 layers × 5) | 3–5ms | `model.transformer.ln_f` reused + `lm_head` × residual; cache `lm_head.weight` |
| Payload build + JSON | <5ms | flat lists; numpy → list once |
| SSE send | <2ms | uvicorn with `loop=uvloop` (POSIX), `asyncio` selector elsewhere |
| **End-to-end token tick** | **<100ms** | feels live (>10 Hz) |
| Frontend frame | <16ms | Canvas heatmap, no DOM reflow |

**First-paint:** ≤200ms after Run is pressed. First two tokens may be slower (caches warming) — the UI shows a one-second skeleton state if no token has arrived in 250ms.

**Demo mode:** A precomputed `CaptureResult` for one example prompt is bundled (`examples/demo_capture.json`, ~150 KB). On first paint, if the model hasn't loaded yet, the UI animates this precomputed run so the visitor sees something alive within ~50ms. A small banner reads "Demo capture — your model is still loading."

---

## 5. One-command-start contract

`make run` (Makefile target) — equivalently `./run.sh` (POSIX) and `.\run.ps1` (Windows) — does **exactly** the following, with progress visible:

1. **Check Python.** Require Python 3.11+; print a clear error otherwise.
2. **Create venv** at `.venv/` if absent. Use the stdlib `venv` module (no pyenv/uv requirement).
3. **Install deps** from `pyproject.toml` (pinned). Progress visible via pip's output.
4. **Warm the cache** (idempotent). On first run only:
   - Download GPT-2 Small weights via `transformers` (`~/.cache/huggingface/`).
   - Download SAELens release `gpt2-small-res-jb` (12 SAEs × ~80MB = ~1GB; show a progress bar). If download fails or is rate-limited, continue with `sae_loaded=False` and surface a banner in the UI.
   - Download bundled triggering-examples corpus (`~5MB` from a release-attached parquet on the project's GitHub releases page; if release unavailable, fall back to a tiny embedded sample shipped in the wheel).
   - Build the triggering-examples index (`~30s`).
5. **Start server.** `uvicorn backend.server.app:app --host 127.0.0.1 --port 8000` in the foreground.
6. **Open browser.** `python -m webbrowser http://127.0.0.1:8000` after a 1s health-check loop. Skip on `--no-browser`.

Total cold start: ~3 min on a decent connection (dominated by the 1 GB SAE download). Warm restart: <5 s.

`make run-fast` skips the SAE download — degrades to neuron-activations + logit lens + attention. Useful when bandwidth is limited.

---

## 6. Hosted-demo plan

We cannot deploy from this environment. Ship `docs/HOSTED_DEMO.md` with a precise recipe:

**Primary path: HuggingFace Spaces (Docker, CPU-basic tier — free).**

`docs/HOSTED_DEMO.md` contains:
- Dockerfile template (multi-stage: base = `python:3.11-slim`, copy `backend/` + `frontend/`, `CMD uvicorn ...`).
- `app.py` shim file Spaces expects.
- README front-matter for Spaces (`sdk: docker`, `app_port: 7860`).
- One-time SAE pre-stage in the image so cold-start is bearable.
- Rate-limit middleware snippet (1 req/sec/IP).
- "If CPU-basic is too slow for SAE: switch to CPU-upgrade tier ($0.03/hr)."

**Backup path: Fly.io** (Dockerfile already compatible). Sample `fly.toml` included.

**Backup path 2: Modal.** A `modal_app.py` shim (commented as optional) shows how to mount the same FastAPI app under a Modal `@asgi_app()`.

The README's "live demo" badge links to a placeholder `https://huggingface.co/spaces/bettyguo/see-the-ai-think` URL; human deploys the Space and updates if the slug differs.

---

## 7. README wireframe

```
[hero image: assets/demo.gif — the most mesmerizing 8 seconds]

# see-the-ai-think

> Watch an LLM think. Interactive, runs on your laptop, no GPU required —
> load a model, type a prompt, see the features light up.

[badges: MIT · CI · no-GPU · live-demo · python-3.11+]

## Live demo
[ ▶  https://huggingface.co/spaces/bettyguo/see-the-ai-think  ]

## Quickstart (60 seconds)
```bash
git clone https://github.com/bettyguo/see-the-ai-think.git
cd see-the-ai-think
make run         # or ./run.sh  (Windows: .\run.ps1)
```
This installs deps, downloads GPT-2 Small + its SAEs (~1 GB on first run), starts the server, and opens your browser at http://127.0.0.1:8000.

## What you're seeing
[3 small screenshots, one per LAUNCH-CRITICAL interaction, with one-line captions]

## What this tool does — and does not — claim
[the honesty note from docs/HONESTY.md, abbreviated]

## How it works
[one paragraph + a link to docs/ARCHITECTURE.md]

## Models supported
- **GPT-2 Small (124M)** — default, ships with Joseph Bloom's residual-stream SAEs (12 layers).
- **Gemma-2-2B** — opt-in, uses Gemma Scope SAEs. Slower; better features.

## Credits & references
- Sparse autoencoders: Bricken et al. 2023 (Anthropic), Templeton et al. 2024 (Anthropic),
  Cunningham et al. 2023 (arXiv:2309.08600).
- Gemma Scope: Lieberum et al. 2024 (arXiv:2408.05147).
- GPT-2 Small SAEs: Joseph Bloom, 2024.
- Logit lens: nostalgebraist, 2020.
- Built on TransformerLens, SAELens, Neuronpedia auto-interp labels.

## Star history
[![Star History](https://api.star-history.com/svg?repos=bettyguo/see-the-ai-think&type=Date)](https://star-history.com/#bettyguo/see-the-ai-think)

## Citing
[bibtex block — Betty Guo, HKU]

## License
MIT © 2026 Dongxin Guo (Betty Guo) · ORCID 0009-0000-2388-1072
Final-year PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu.
```

The GIF is **above the H1**. The stranger sees movement before they see a word.

---

## 8. Test plan

| Test | What it checks |
|---|---|
| `test_capture.py::test_residual_shapes` | residuals[i].shape == (T, 768) for all i, fixture prompt |
| `test_capture.py::test_attention_shapes` | attn[i].shape == (12, T, T) |
| `test_capture.py::test_final_layer_prediction` | last-position top-1 == " dog" (regression catch) |
| `test_logit_lens.py::test_layer_progression` | logit-lens entropy monotonically decreases on average from layer 0 → 12 |
| `test_sae.py::test_top_k_shape` | top-K returns ≤K features and all activations > 0 |
| `test_sae.py::test_sae_unavailable_path` | with no SAE weights, capture still succeeds, sae_loaded=False |
| `test_server_routes.py::test_models_route` | /models returns at least gpt2-small |
| `test_server_routes.py::test_examples_route` | /examples returns ≥3 prompts |
| `test_server_routes.py::test_feature_route` | /feature/8/3127 returns label tier in {MEASURED,AUTO-LABEL,SOURCED,null} |
| `test_stream.py::test_sse_event_order` | /generate yields events in token order, schema-valid |
| `test_stream.py::test_sse_payload_shape` | each event has tokens, top_features, logits_per_layer keys |
| `tests/smoke_oneshot.sh` (CI only) | from clean clone, `make run-fast` starts and serves /health within 90s |

CI matrix: Ubuntu-latest + macOS-latest, Python 3.11 and 3.12. Windows runs the unit subset; full smoke test runs on Linux only.

---

## CHECKPOINT 1 SUMMARY

- **Done:** Repo layout fixed. Capture/UI/perf contracts written with concrete shapes, payloads, and budgets. One-command-start spec exact. README wireframe drafted (GIF-first). Test plan enumerated. Hosted-demo path documented (HF Spaces primary).
- **Files:** `PLANNING/01_design.md`.
- **Next:** Phase 2 — scaffold repo, implement capture engine for GPT-2 Small, implement streaming server, write unit tests against fixture.

Proceeding to Phase 2 per autonomous-execution directive.
