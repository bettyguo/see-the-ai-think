# `data/` — runtime cache

This directory is **populated at runtime** and is not committed.

`make run` (via `python -m backend.warm`) downloads:

- GPT-2 Small weights → `~/.cache/huggingface/`
- Joseph Bloom's GPT-2 Small SAEs → `~/.cache/huggingface/`
- Feature-labels snapshot (Neuronpedia auto-interp) → `data/feature_labels.json` (when a release is published; otherwise the in-package fallback at `backend/data/feature_labels.json` is used)
- Triggering-examples corpus index → `data/feature_triggers.json`

If `data/feature_labels.json` exists when the server starts, it overrides the in-package fallback. This lets us ship a richer labels snapshot via GitHub Releases without re-packaging the wheel.

To wipe and re-warm:
```bash
rm -rf data/ ~/.cache/see-the-ai-think
make run
```
