# Contributing to see-the-ai-think

Thanks for considering a contribution. This project lives or dies on its **demo quality** — every PR should make the GIF better, the truth-claim tighter, or the setup friction lower.

## Quick development setup

```bash
git clone https://github.com/bettyguo/see-the-ai-think.git
cd see-the-ai-think
make dev          # installs dev deps + runs the server with --reload
```

In another terminal:

```bash
make test         # unit tests (fast, no network)
make lint         # ruff
```

The integration suite is opt-in (it downloads ~500 MB of GPT-2 weights):

```bash
.venv/bin/pip install -e ".[sae]"
.venv/bin/pytest -m integration
```

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full map. Tl;dr:

- `backend/models/capture.py` — forward hooks + the `CaptureEngine`.
- `backend/features/{sae,logit_lens,labels,triggers}.py` — feature extraction.
- `backend/server/` — FastAPI app, SSE stream, JSON routes.
- `frontend/` — single-page app, no build step.
- `examples/prompts.json` — the demo-driver prompts. Add yours here if it produces a striking visual.

## What we love in a PR

- A new pre-baked prompt in `examples/prompts.json` that produces a *visibly different* heatmap from existing ones.
- Labels with `SOURCED` tier — i.e., feature interpretations backed by a paper you can cite. Add them to `backend/data/feature_labels.json`.
- Frontend polish — smoother animation, better tooltip wording, prettier color ramps.
- New model support. Today we only support GPT-2 family in `backend/models/capture.py::_block_modules`; adding Gemma or Llama mostly means another `hasattr` branch.

## What we will push back on

- **Claims without sources.** If a PR adds a label that asserts what a feature "means" without a citation, it lands as `AUTO-LABEL` at most — and we'll want the source link.
- **Setup friction.** A new dependency, a new download step, a new env var — each one costs us users. Make the case in the PR description.
- **Code that doesn't run on a laptop CPU.** GPU-only paths are fine as opt-in flags; they cannot become defaults.

## Honesty rules (load-bearing)

This is the most important norm here. Read [docs/HONESTY.md](docs/HONESTY.md). The TL;DR is: measurements are real, interpretations get badges, nothing claims more than the evidence supports.

## License

By contributing you agree your changes are MIT-licensed.

— Betty Guo
