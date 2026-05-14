# PHASE 6 — HOSTILE REVIEW

> Re-read the repo as (a) a skeptical HN commenter, (b) an interpretability researcher checking that no claim overreaches the evidence, and (c) a first-time visitor with 8 seconds and a slow laptop. List every weakness; fix what's fixable now; file the rest as issues.

Date: 2026-05-14. Reviewer: Betty Guo (author wearing a different hat).

---

## A. Skeptical Hacker News commenter

| # | Concern | Verdict | Action |
|---|---|---|---|
| A1 | *"This is just a colorful attention map, it doesn't mean anything."* | The README and the in-UI `MEASURED`/`SOURCED`/`AUTO-LABEL` badges already address this explicitly. We do not claim semantic meaning we can't source. | Already covered — no change. |
| A2 | *"GPT-2 features don't really mean anything specific anyway."* | Partially true. Many GPT-2 SAE features are messy. We acknowledge this in `docs/ARCHITECTURE.md` ("What we don't ship in v0.1") and the launch positioning is "**watch** the model think" — not "**understand** what every feature means". | OK as-is. |
| A3 | *"How do I know the demo capture I see immediately is fake?"* | The status bar shows "demo mode" + the banner says "Demo capture — synthetic activations." | Already covered. |
| A4 | *"The hosted demo URL doesn't work."* | Honest in `docs/HOSTED_DEMO.md` — the deploy is a manual human step. The README's badge points to a slot we own. | Fixed in `docs/HOSTED_DEMO.md`. |
| A5 | *"`make run` takes 60 seconds, sure — then waits 3 minutes downloading."* | Real friction. Quickstart claim now says ~3 min cold-start. | **Fixed** — README updated. |
| A6 | *"Windows path."* | `run.ps1` exists and mirrors `make run`; CI matrix includes Windows for unit tests. Smoke test stays Linux-only because a few of the shell idioms in CI assume bash. | OK. |
| A7 | *"Where's the LICENSE on the third-party content (the corpus, the labels)?"* | We ship NO third-party text by default — only placeholder labels we wrote. Real Neuronpedia labels are downloaded at warm time from a release URL we don't pre-bake. | **Fixed** — placeholder source strings now say so honestly. |

## B. Interpretability researcher

| # | Concern | Verdict | Action |
|---|---|---|---|
| B1 | *"You ship 'Neuronpedia auto-interp' labels you didn't actually pull from Neuronpedia. That's worse than no labels."* | **Critical, and correct.** The bundled `feature_labels.json` and the inline `_BUNDLED` dict were plausible-sounding labels with a "Neuronpedia auto-interp" source string. This violated the very honesty contract we shipped. | **Fixed.** Bundled labels are now explicitly marked as "placeholder — replaced at warm time by Neuronpedia snapshot" with text that says so. The fabricated trigger examples got `[placeholder example]` prefixes. The frontend fallback `BUNDLED` dict was likewise rewritten. |
| B2 | *"Bloom's SAEs are trained on `hook_resid_pre` — the residual entering each block. You were encoding the post-block residual."* | **Real bug.** `_residuals[layer_idx + 1]` is the residual leaving the block; the SAEs expect `_residuals[layer_idx]`. | **Fixed** in `backend/models/capture.py::_token_captures` and `_single_position_capture` with a comment pointing at the rule. |
| B3 | *"The logit-lens recipe should use the model's final LayerNorm, not the per-layer one."* | We use `transformer.ln_f` (final LayerNorm) for every layer, which is the standard logit-lens recipe (nostalgebraist 2020). | OK. |
| B4 | *"Why do you cut feature steering? That's the only interactively-causal thing."* | True — steering is the moonshot. We cut for v0.1 because GPT-2 Small features don't produce dramatic-enough shifts to be convincing in a 10-second demo. Documented openly in `docs/ARCHITECTURE.md`. The honest position is "we'll revisit with Gemma-2-2B + Gemma Scope." | OK as-is; tracked for v0.2. |
| B5 | *"What about the entropy / 'surprise' ribbon — you compute entropy over only top-5 probs?"* | True, and the tooltip says so: *"entropy X (top-5 only — true entropy is higher)"*. Trend across tokens is what matters here; absolute values are not claimed. | OK. |
| B6 | *"You don't ship steering AND you don't ship probing. What's the actual mech-interp delta over BertViz?"* | The delta is: **prompt-first, live, browser-native, labeled SAE features.** BertViz is attention-only and notebook-embedded. The README makes this delta explicit in the landscape table (Phase 0 doc). | OK. |
| B7 | *"`_attn_top_per_layer` only stores top-3 per layer — you lose information."* | True, by design — payload size matters for the SSE budget. The interactive attention-arcs view uses what's stored. Documented in `docs/ARCHITECTURE.md`. | OK. |
| B8 | *"GPT-2 Small SAEs are open but Bloom's release pinning may drift."* | We pin SAELens in `pyproject.toml` and use the stable `gpt2-small-res-jb` release name. Risk noted in Phase 0 risk log. | OK. |

## C. First-time visitor (8 seconds, slow laptop)

| # | Concern | Verdict | Action |
|---|---|---|---|
| C1 | *"The page is blank."* | The static demo capture autostarts on `boot()` — animation begins within ~50 ms. | Covered. |
| C2 | *"What am I supposed to do?"* | The controls panel is on the left with a labeled example dropdown and a prominent ▶ run button. Examples are pre-baked to produce striking visuals. | Covered. |
| C3 | *"Mobile?"* | A media query at 900 px collapses the panel; the heatmap is still visible but readable only in landscape. Mobile is **not** the launch target — it's a desktop / screenshot tool. | OK; the README's GIF makes it clear this is a desktop experience. |
| C4 | *"Slow CPU."* | Demo mode loads in <100 ms regardless. Live mode is gated behind a button press. | Covered. |
| C5 | *"No JS"* | The page is empty without JS. Acceptable — this is an interactive tool. | OK. |
| C6 | *"Network blocks /api/generate during demo"* | The UI explicitly falls back to demo mode and shows a banner. | Covered. |

## D. Things I considered but didn't change

| Item | Why |
|---|---|
| Add a `model` selector to the UI | Only one model ships in v0.1. Adding a selector before there's a second model implies false flexibility. |
| Add a "share this view" deep-link feature | Premature. The demo carries the message; permalinks can come once we have real users wanting them. |
| Bundle a small corpus index for triggering examples | Deferred to v0.2. The placeholder examples are honest; a real corpus index is a larger build artifact. |
| Add a Dockerfile to the repo root | Deferred. We don't want it to be the default install path; it would clutter the README's "60s quickstart" framing. The recipe is in `docs/HOSTED_DEMO.md`. |
| Add Gemma support | Tracked for v0.2 — the `backend/models/capture.py::_block_modules()` map already gestures at the extension point. |

## E. Verified facts

- **Source files all ≤ 500 lines.** Largest: `backend/models/capture.py` at 425.
- **Unit tests:** 28 pass, 3 integration-marked deferred. Lint clean.
- **Server smoke (lazy mode):** boots and serves all 10 asset paths.
- **All cited works exist:** Bricken et al. 2023 (Anthropic), Templeton et al. 2024 (Anthropic), Cunningham et al. 2023 (arXiv:2309.08600), Lieberum et al. 2024 (arXiv:2408.05147), nostalgebraist's logit-lens post (LessWrong 2020), Joseph Bloom's `gpt2-small-res-jb` release on HuggingFace.

## F. Open issues to file post-launch

1. **Real Neuronpedia label snapshot.** Add a build step that pulls auto-interps for the top-1000 features and bakes them into a release-attached `feature_labels.json`. Until then, the placeholder honesty markers stay.
2. **Real triggering-examples corpus.** Index ~5 MB of permissively-licensed text (Pile-CC slice + Wikipedia first paragraphs) at warm time.
3. **Gemma-2-2B + Gemma Scope (v0.2).** Including the refusal-feature spotlight visual.
4. **Feature steering (v0.2 stretch).** Only if Gemma-2-2B + Gemma Scope produces clean, demoable shifts.
5. **Dockerfile + GitHub Actions matrix job that exercises a live capture against gpt2-small.** Right now CI runs only unit tests + lazy-server smoke; a true end-to-end CI run is gated on GPU runners, which we don't have.
6. **Better mobile layout.** Not blocking; not in scope for v0.1.

— Betty Guo, hostile-review-of-self pass
