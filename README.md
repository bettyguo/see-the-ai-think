<p align="center">
  <img src="assets/demo.gif" alt="see-the-ai-think — watch an LLM think" width="820" />
  <br>
  <em>(If the GIF isn't here yet, follow <a href="assets/RECORD_DEMO.md">assets/RECORD_DEMO.md</a> to record one.)</em>
</p>

# see-the-ai-think

> **Watch an LLM think.** Interactive, runs on your laptop, **no GPU required** — load a model, type a prompt, see the features light up.

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/bettyguo/see-the-ai-think/actions/workflows/ci.yml/badge.svg)](https://github.com/bettyguo/see-the-ai-think/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![no GPU required](https://img.shields.io/badge/no%20GPU-required-06d6a0.svg)](#what-youre-seeing)
[![live demo](https://img.shields.io/badge/live-demo-3a86ff.svg)](https://huggingface.co/spaces/bettyguo/see-the-ai-think)

---

## Live demo

▶ **[huggingface.co/spaces/bettyguo/see-the-ai-think](https://huggingface.co/spaces/bettyguo/see-the-ai-think)** (deploy recipe in [docs/HOSTED_DEMO.md](docs/HOSTED_DEMO.md))

If the Space is asleep, the static demo capture animates immediately so you see the visualization while the model wakes up.

## Quickstart

```bash
git clone https://github.com/bettyguo/see-the-ai-think.git
cd see-the-ai-think
make run            # Linux / macOS
# Windows: .\run.ps1
```

`make run` creates a `.venv`, installs deps (~30 s), downloads **GPT-2 Small** (~500 MB) and **Joseph Bloom's residual-stream SAEs** (~1 GB) on first run, starts the server, and opens your browser at **http://127.0.0.1:8000**. Cold start: **~3 min** on a decent connection. Warm restart: **<5 s**.

**While the model warms,** the UI animates a small bundled demo capture so you see the visualization within ~50 ms regardless.

**Slow network?** `make run-fast` skips the SAE download. The UI falls back to raw neuron activations + logit lens + attention — still beautiful, just without the labeled features.

## What you're seeing

1. **Feature firing heatmap** — top features (rows) by activation magnitude across each token (cols). New columns slide in as the model generates; rows re-rank as new features dominate.
2. **Click any feature → triggering tokens** — opens a panel with the feature's auto-label (if one exists), its tier badge, and the corpus snippets that activate it most strongly.
3. **Logit-lens scrubber** — slide through layers and watch the next-token prediction "develop" from gibberish at layer 0 to the model's actual choice at layer 12.
4. **Attention arcs** — click any token chip to see what it most attended to per layer.
5. **Surprise ribbon** — a thin entropy bar under the prompt, color-coded to show where the model was certain vs. surprised.

## What this tool does — and does not — claim

This product surfaces **measured** internal activity. Any interpretation gets one of three pill badges so you can tell at a glance what to trust:

| Badge | Meaning |
|---|---|
| <kbd>MEASURED</kbd> | Direct activation magnitude or probability. Never speculative. |
| <kbd>SOURCED</kbd> | Label from a paper or human-labeled artifact, with citation shown. |
| <kbd>AUTO-LABEL</kbd> | Auto-interpretation (e.g. Neuronpedia auto-interp). Marked speculative, may be wrong. |

The tool will tell you *"feature L8/F1024 fires with magnitude 4.2 on token ` Marie`"*. It will not tell you *"feature L8/F1024 is the proper-noun feature"* unless that label exists in a source we can cite.

Full rules: [docs/HONESTY.md](docs/HONESTY.md).

## How it works

The backend hooks every transformer block via `register_forward_hook`, captures per-layer residual streams + attention patterns + SAE encodings, and streams them over SSE as the model generates token-by-token. The frontend is a no-build single-page app rendering on Canvas + SVG.

Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Models

| Model | Status | SAEs | Notes |
|---|---|---|---|
| **GPT-2 Small (124M)** | shipped | Joseph Bloom's residual-stream SAEs, 12 layers | Default. Runs ~30 tok/s on a laptop CPU. |
| **Gemma-2-2B** | planned (`v0.2`) | Gemma Scope (Lieberum et al. 2024) | Heavier; better safety / refusal features. |

## Credits & references

- **Sparse autoencoders.** Bricken et al. 2023 — *Towards Monosemanticity* (Anthropic). Templeton et al. 2024 — *Scaling Monosemanticity* (Anthropic). Cunningham et al. 2023 — *Sparse Autoencoders Find Highly Interpretable Features in Language Models* ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600)).
- **Gemma Scope.** Lieberum et al. 2024 ([arXiv:2408.05147](https://arxiv.org/abs/2408.05147)).
- **GPT-2 Small SAEs.** Joseph Bloom, 2024 — public release `gpt2-small-res-jb`.
- **Logit lens.** nostalgebraist, 2020 — "interpreting GPT: the logit lens" (LessWrong).
- Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [SAELens](https://github.com/decoderesearch/SAELens), and [Neuronpedia](https://neuronpedia.org) auto-interpretation snapshots.

## Star history

[![Star History](https://api.star-history.com/svg?repos=bettyguo/see-the-ai-think&type=Date)](https://star-history.com/#bettyguo/see-the-ai-think)

## Citing

```bibtex
@software{guo2026seetheaithink,
  author       = {Guo, Dongxin},
  title        = {{see-the-ai-think}: Watch an LLM think.},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/bettyguo/see-the-ai-think}
}
```

## License & attribution

MIT © 2026 **Dongxin Guo (Betty Guo)** · ORCID [0009-0000-2388-1072](https://orcid.org/0009-0000-2388-1072)

Final-year PhD candidate in Computer Science, University of Hong Kong. Advised by Prof. Siu-Ming Yiu.
