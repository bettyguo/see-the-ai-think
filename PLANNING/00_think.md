# PHASE 0 — THINK

> Pre-implementation research and decision log. No project code written in this phase.
> Date: 2026-05-14. Author: Betty Guo (Dongxin Guo, HKU, advised by Prof. Siu-Ming Yiu).

---

## 1. Landscape scan — what exists, what's missing

### Existing tools (verified May 2026)

| Tool | What it is | What it does well | Why it's NOT the product we're building |
|---|---|---|---|
| **TransformerLens** ([TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)) | Python library for hooking GPT-style models | Best-in-class for caching activations and running interventions from a notebook | Notebook-only, no UI, requires you to already know what you're looking for. HookedSAE was removed in v2.0 and moved to SAELens. |
| **SAELens** ([decoderesearch/SAELens](https://github.com/decoderesearch/SAELens), `sae-lens` on PyPI) | Library to train and load Sparse Autoencoders on LLMs | Standardized SAE loading; integrates with TransformerLens and nnsight | Programmer's tool. No browser UI. No "type a prompt and watch." |
| **Neuronpedia** ([github.com/hijohnnylin/neuronpedia](https://github.com/hijohnnylin/neuronpedia), [neuronpedia.org](https://neuronpedia.org)) | Hosted web platform for browsing SAE features | Searchable, auto-interp labels, gorgeous feature pages, has Gemma Scope hosted | **Browse-first, not prompt-first.** You explore precomputed features. You can't easily type your own prompt and watch features fire live across layers as the model generates. It's a microscope catalog; we're building a live oscilloscope. |
| **nnsight / nnterp** ([NDIF](https://nnsight.net)) | Transparent interventions on PyTorch models, with remote execution | Powerful API for cross-model interventions | Library API for researchers. No casual-visitor UI. |
| **BertViz / CircuitsVis** | JS components for attention heads / neuron activations | Pretty, embeddable | Component libraries — building blocks, not a runnable app. Attention-only or single-view. |
| **Inseq, Captum** | Feature attribution / saliency | Strong for token-attribution research | Not feature-level mechanistic; not "watch it think." |

### The gap (this is the product)

None of the above gives a **casual visitor** an instant, in-browser, "I typed a prompt and now I'm watching the model's internal features light up across layers and tokens" experience. Neuronpedia is the closest, but it is a **catalog of pre-mined features**, not a live forward-pass visualizer. SAELens + TransformerLens can do everything technically, but only via Python notebooks.

**The product is the missing UX layer between a pre-trained SAE and a non-researcher's curiosity.** That layer has enormous latent demand (mech interp is the most-discussed AI safety topic of the last 18 months) and no shipped artifact. Exactly the position `nanochat` occupied for "train a real model from scratch cheaply."

### Small open models with public SAEs (verified)

| Model | Params | SAEs available | Laptop CPU runnable? | Notes |
|---|---|---|---|---|
| **GPT-2 Small** | 124M | Joseph Bloom's residual-stream SAEs, all 12 layers, ~25k features each — loadable via SAELens (`jbloom/GPT2-Small-SAEs-Reformatted`). Neuronpedia has auto-interp labels for many features. | **Yes — easily, ~30 tok/s on a modern laptop CPU.** | The canonical mech-interp model. Not instruction-tuned, so no refusal features — but generation is still legible. **Our launch model.** |
| **Pythia-70M / 160M** | 70M / 160M | EleutherAI SAEs | Yes | Even smaller; less famous; coverage thinner than GPT-2 small. Backup. |
| **Gemma-2-2B** | 2B (base + IT) | **Gemma Scope** ([google/gemma-scope](https://huggingface.co/google/gemma-scope)) — JumpReLU SAEs on every layer, base + IT. Neuronpedia has rich auto-interps. | Yes on CPU at int8/fp16 (~3–10 tok/s) or any modest GPU. | Bigger and more impressive features (the IT model has real refusal/safety features). **Our stretch / "advanced" model.** |
| Llama-3.2-1B / 3B | 1B / 3B | Community SAEs, less comprehensive | Yes (1B comfortably) | Skip for v1 — uneven SAE coverage. |

**Decision:** Ship with **GPT-2 Small + Bloom SAEs** as the default (fastest, fewest dependencies, runs everywhere). Add **Gemma-2-2B + Gemma Scope** as an opt-in "advanced" model in Phase 4 if time permits. Document both.

---

## 2. The "wow" inventory — 10 candidate interactions

Scored 1–5: **W** = how mesmerizing the resulting GIF is. **E** = effort (1 = trivial, 5 = hard). Ratio is W/E (higher = better wow-to-effort).

| # | Interaction | W | E | W/E | Notes |
|---|---|---|---|---|---|
| 1 | **SAE feature firing heatmap over the prompt** — top-N active features (rows) × tokens (cols), colored by activation magnitude; animates as new tokens stream in | 5 | 2 | **2.5** | **LAUNCH-CRITICAL.** This is the iconic shot. Genuinely novel as a live thing — Neuronpedia shows static feature pages, not live heatmaps over your own prompt. |
| 2 | **Click-a-feature → triggering tokens** — clicking any active feature opens a panel showing its top-K activating examples (from a small bundled corpus) and, when available, its auto-interp label sourced from Neuronpedia | 5 | 2 | **2.5** | **LAUNCH-CRITICAL.** Gives the heatmap *meaning*. The labels are the "aha!" moment. |
| 3 | **Layer-by-layer logit lens scrubber** — slider for layer index 0..N; for each layer, show the top-5 predicted next tokens (probabilities visible). Watch the prediction "develop" through the network. | 4 | 2 | **2.0** | **LAUNCH-CRITICAL.** Classic mesmerizing visual; technically a well-known technique (`logit lens`, nostalgebraist 2020); easy to implement and very pretty. |
| 4 | **Feature steering** — toggle a feature on/off (or dial its activation up/down), re-run the forward pass, watch the output text change | 5 | 4 | 1.25 | The single most viral interaction *if* it works convincingly. Risk: GPT-2 Small features are subtle — steering might produce gibberish instead of clean shifts. **Phase 4 stretch.** |
| 5 | **Attention pattern grid** (heads × tokens) with hover highlight | 3 | 1 | 3.0 | High ratio but everyone has seen this. Include as a side panel, not as the hero shot. |
| 6 | **Refusal/safety feature spotlight** | 5 | 4 | 1.25 | Only works on instruction-tuned models with identified safety features. Possible with Gemma-2-2B-IT + Gemma Scope auto-interp labels. **Phase 4 stretch, advanced-model only.** |
| 7 | **Cross-layer feature trajectory** — pick a token, show how its representation moves through feature space layer by layer | 4 | 4 | 1.0 | Cool but visually abstract. Cut for v1. |
| 8 | **Top-k features per token sparkline** (compact view) | 2 | 1 | 2.0 | Cheap; useful as a secondary chart. |
| 9 | **Neuron ablation visualization** | 3 | 5 | 0.6 | Too heavy for the demo budget. Cut. |
| 10 | **"Surprise" map** — entropy/perplexity per token, color-coded | 3 | 1 | 3.0 | Low effort, looks nice; secondary feature. |

### LAUNCH-CRITICAL set (Phase 3)

1. **SAE feature firing heatmap over the prompt** (#1)
2. **Click-a-feature → triggering tokens + (sourced) label** (#2)
3. **Layer-by-layer logit lens scrubber** (#3)

### Phase 4 stretch goals

- **Feature steering** (#4) — the moonshot. If it works, it's the most viral 8 seconds in the GIF.
- **Attention pattern grid** (#5) — cheap, complements the SAE view.
- "Surprise" / entropy ribbon under the token row (#10) — cheap polish.

### Cut from v1

#4 unless Phase 4 delivers convincingly; #6 (needs advanced model); #7, #9.

---

## 3. Model + SAE choice

**Default (ships as primary):** GPT-2 Small (124M, [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)) + Bloom residual-stream SAEs (loaded via SAELens release `gpt2-small-res-jb`).

- ✅ Runs on a CPU laptop, ~30 tok/s.
- ✅ SAE coverage on all 12 layers.
- ✅ Auto-interp labels available via Neuronpedia for many features.
- ⚠️ Not instruction-tuned — generations are short-range and a bit weird. Mitigation: pre-baked prompts in `examples/` chosen to produce legible completions.
- ⚠️ "Feature 3127 = deception" claims are absent for GPT-2 Small — features are mostly linguistic / syntactic / topic. Honesty rules (§5) handle this.

**Optional / advanced (Phase 4 stretch):** Gemma-2-2B-IT + Gemma Scope.

- ✅ Real refusal / safety / role features with rich auto-interp.
- ⚠️ Heavier — ~5GB download, slower on CPU, may need int8 quant.
- ⚠️ Gemma Scope SAEs are large (per-layer JumpReLU, many features). We'll load only 1–2 layers at a time.

**Fallback if SAE loading fails on the user's machine:** Raw neuron activations from MLPs + attention head patterns + logit lens — still produces a great demo. The UI should gracefully degrade and label what's missing ("SAE features unavailable — showing raw activations").

---

## 4. Runtime architecture sketch

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (frontend/)                                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  HTML + minimal JS (no build step)                     │ │
│  │  ├─ Heatmap (Canvas)                                   │ │
│  │  ├─ Logit-lens scrubber (SVG)                          │ │
│  │  ├─ Feature panel (DOM)                                │ │
│  │  └─ SSE/WS client                                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              ▲                              ▲
              │ SSE: token+features payload  │ HTTP: /feature/{id}
              │ (~10–50 KB per token)        │
┌─────────────┴──────────────────────────────┴────────────────┐
│  Python backend (backend/server/, FastAPI)                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  POST /generate           → SSE stream of token events │ │
│  │  GET  /feature/{layer}/{id} → triggering examples      │ │
│  │  GET  /models             → list available             │ │
│  │  GET  /examples           → pre-baked prompts          │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                  │
│  ┌───────────────────────┴────────────────────────────────┐ │
│  │  Capture engine (backend/models/, backend/features/)   │ │
│  │  ├─ HuggingFace transformers (model + tokenizer)       │ │
│  │  ├─ Forward hooks on residual stream per layer         │ │
│  │  ├─ SAE forward pass (SAELens) → top-K feature acts    │ │
│  │  ├─ Logit lens (per-layer unembed)                     │ │
│  │  ├─ Attention pattern capture                          │ │
│  │  └─ Local cache: features.parquet + triggering corpus  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Per-token streaming payload (target ~20 KB):**
```json
{
  "token_id": 7,
  "text": " world",
  "logits_top5_per_layer": [[("the",0.31),...], ...],   // 13 layers × 5
  "top_features": [
    {"layer": 8, "feature": 3127, "act": 4.2},
    ...                                                  // top 32 across all layers
  ],
  "attention_top_per_layer": [...]                       // sparse, summarized
}
```

**Latency budget (CPU laptop, GPT-2 Small):**
- Forward pass for 1 token: ~30–60 ms.
- SAE encode for 12 layers × top-K: ~10–20 ms (sparse top-K).
- JSON serialize + SSE send: <5 ms.
- **Total: ~50–80 ms per token** → comfortably under the 100 ms "feels live" threshold.

**Caching strategy:** Triggering-examples corpus pre-computed at install time (one-time ~30 s indexing pass over a small text corpus). Per-feature panel loads are then a parquet lookup, <10 ms.

**Local-first, hosted optional:** The tool is local-first. Hosted demo path uses a small VM/container (HF Spaces or a tiny VPS) with the same backend. We do **not** ship a "send your prompt to our server" SaaS — users run their own backend. The hosted demo is one user/session for casual visitors.

---

## 5. Honesty rules (the rule for what the UI may claim)

This product makes mech interp **visible**. It must not make it look like it knows things it doesn't.

| ✅ The UI MAY say | ❌ The UI MUST NOT say |
|---|---|
| "Feature L8/F3127 fires with magnitude 4.2 on token ` world`." | "Feature L8/F3127 is the *world-knowledge* feature." (unless sourced) |
| "Top-10 tokens that activate this feature, from corpus X" — and show them. | "This feature *means* X" without evidence. |
| "Auto-label (from Neuronpedia): *time-of-day expressions*. [Speculative — auto-generated.]" — with a clearly visible badge. | A label rendered as if it were ground truth. |
| "Layer 11 predicts ` world` next with p=0.42 (logit lens)." | "The model has *decided* to say ` world` by layer 11." (causal language) |

**Concrete enforcement:**
- Every interpretive label in the UI is rendered with one of three pill badges:
  - `MEASURED` (green) — direct activation/probability values.
  - `AUTO-LABEL` (amber, dashed border) — auto-interp output. Link to source. Tooltip explains "generated by an LLM, may be wrong."
  - `SOURCED` (blue) — sourced from a paper or human-labeled artifact. Cite next to the label.
- A persistent "How to read this" link in the footer expands the honesty note.
- README has a dedicated "What this tool does and does not claim" section.

---

## 6. Risk log

| Risk | Severity | Mitigation |
|---|---|---|
| **Setup friction kills virality.** | Critical | One-command `run.sh` / `make run`. First-run downloads ~500 MB (model + SAEs + corpus). Show progress. Cache aggressively. Bundle a "demo mode" with pre-computed activations for one example prompt so the UI is *immediately* alive even before the model loads. |
| **GPU requirement kills virality.** | Critical | Default model is 124M and runs on CPU. Hardcoded `device="cpu"` with `--gpu` opt-in. Badge: "no GPU required" in README and UI header. |
| **Looks cool but claims nonsense → reputation hit.** | High | Honesty rules (§5) enforced in code. No claim shipped in the UI without a source pill. Pre-launch hostile review (Phase 6) explicitly checks every UI claim. |
| **Chosen model has no good SAEs.** | Low (GPT-2 Small SAEs are well-established) | GPT-2 Small + Bloom SAEs are public, in SAELens, and integrated with Neuronpedia. Fallback to raw neuron activations if SAEs fail to download. |
| **Frontend perf — heatmap stutters as tokens stream.** | Medium | Canvas (not DOM) for the heatmap; cap top-N features at 32; throttle UI updates at 30 fps; debounce streaming. |
| **HF model download fails or is rate-limited at demo time.** | Medium | Cache to `~/.cache/see-the-ai-think/`. Document offline use. Pre-stage models in the hosted demo image. |
| **Feature steering produces nonsense and looks broken.** | Medium (Phase 4 only) | Restrict to known-good features (curated list of ~10 features for the default model with documented effects). Show "before / after" side-by-side; user can verify the shift themselves. Cut from v1 if results are unconvincing. |
| **SAELens or TransformerLens breaking changes between now and launch.** | Low | Pin versions in `pyproject.toml`. CI runs on the pinned versions. |
| **Windows/Linux/macOS divergence in `run.sh`.** | Medium | Provide both `run.sh` (Linux/macOS) and `run.ps1` / `make.bat` (Windows), plus a fallback `python -m see_the_ai_think` invocation. |
| **License/data risks from bundled corpus for triggering examples.** | Low | Use a public-domain or permissively-licensed small corpus (e.g., a subset of the Pile-CC slice, or Wikipedia first paragraphs of N articles). Document source. |
| **Hosted demo cost / abuse.** | Medium | If we deploy: rate-limit per IP; no API keys exposed; one model only; auto-shutdown after idle. If we can't host reliably, ship `docs/HOSTED_DEMO.md` with a deploy recipe instead. |

---

## 7. Open questions (proceeding with logged assumptions per the operating contract)

1. **Backend hook library — TransformerLens vs raw HuggingFace hooks?**
   *Assumption (proceeding):* Use raw HuggingFace `transformers` + custom forward hooks for portability (works with any HF model, no extra dependency surface). Use SAELens only for SAE loading. Reason: TransformerLens reshapes models, which limits future model support; we want the option to add Gemma-2-2B without fighting TL's coverage.

2. **Frontend stack — build step or no build step?**
   *Assumption:* **No build step.** A single `frontend/index.html` plus a small handful of vanilla JS modules. Tailwind via Play CDN for styling, Alpine.js or htmx for reactivity, D3 + Canvas for visuals. Reason: zero install friction for anyone hacking on the frontend; matches the "trivial setup" promise.

3. **Streaming protocol — SSE or WebSocket?**
   *Assumption:* **SSE.** Server → client only; we don't need duplex. Easier to debug, plays nicely with FastAPI.

4. **Auto-interp labels — pull from Neuronpedia API at runtime, or pre-bundle a snapshot?**
   *Assumption:* **Pre-bundle a snapshot** of GPT-2 Small feature labels at build time, fall back to the Neuronpedia API if the user wants advanced models. Reason: works offline; reproducible demo; respects Neuronpedia's load.

5. **Triggering-examples corpus — what text?**
   *Assumption:* A ~5 MB curated mix of: top 5k Wikipedia first paragraphs (CC-BY-SA), TinyStories sample (cdcacc), and a few literary public-domain excerpts. Indexed at first run.

6. **Hosted demo — where?**
   *Assumption:* HuggingFace Spaces (CPU tier, free) for v1. If too slow, document a Fly.io / Modal deploy in `docs/HOSTED_DEMO.md`. The actual deploy is a manual human step (we cannot deploy from this environment).

7. **Browser support — Chrome only, or all evergreens?**
   *Assumption:* Target evergreen Chrome / Firefox / Safari. No IE/legacy.

8. **Recording the demo GIF — automated or manual?**
   *Assumption:* **Manual.** Browser-based recording cannot be automated from this environment. Ship `assets/RECORD_DEMO.md` with an exact shot list (prompt, interactions, timing).

---

## CHECKPOINT 0 SUMMARY

- **Done:** Landscape scan complete; gap clearly identified (live, prompt-first, browser-native is empty). Wow inventory scored. 3 LAUNCH-CRITICAL interactions chosen. Primary model = GPT-2 Small + Bloom SAEs; stretch = Gemma-2-2B + Gemma Scope. Architecture sketched (FastAPI + SSE + no-build frontend). Honesty rules drafted with three label tiers. Risks logged with mitigations. 8 open questions resolved with documented assumptions.
- **Files:** `PLANNING/00_think.md`.
- **Next:** Phase 1 — write `PLANNING/01_design.md`: repo layout, capture/UI/perf contracts, README wireframe, test plan.

Per the autonomous-execution directive, proceeding to Phase 1 without waiting for human confirmation. Assumptions logged above.
