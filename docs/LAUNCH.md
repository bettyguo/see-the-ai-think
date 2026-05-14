# Launch playbook — `see-the-ai-think`

Timing, copy, and channel-by-channel sequencing for the 90-day launch window. The one rule: **the GIF leads everywhere.** Text is in service of the GIF.

## D-day (Tuesday or Wednesday, 9:30 AM Pacific = 18:30 CET = 12:30 ET)

These are the historically best Show HN slots: US west coast pre-work, EU still online, EA / interpretability crowd available.

### 1. GitHub release

- Tag `v0.1.0` with the README's pitch as the release note.
- Confirm CI is green, Space is live.
- Pin the repo on `github.com/bettyguo`.

### 2. Show HN (the lead channel)

**Title:** `Show HN: see-the-ai-think – watch an LLM think, in your browser, no GPU`

Keep the title under 80 chars; HN truncates long titles. "Show HN:" prefix is load-bearing — it routes to a friendlier audience.

**First comment (auto-post 2 min after submission):**

> I built this because every interpretability tool I love (TransformerLens, SAELens, Neuronpedia) is either notebook-only or a static catalog. I wanted a thing where you type a prompt and *watch the features fire*, live, in a browser, on a laptop. So that's what this is — GPT-2 Small + Joseph Bloom's SAEs + logit lens + attention arcs, all streamed over SSE.
>
> The honesty rules are load-bearing (docs/HONESTY.md): activations are real; labels carry a tier badge. Nothing claims to "mean" something without a citation.
>
> Demo: <Space URL>. Source: <repo URL>. Happy to answer questions about the hook plumbing, the SAE choice, or why I cut feature steering for v0.1.

### 3. X (Twitter) — the GIF leads

Thread structure (5 tweets):

1. The GIF + one sentence: *"watch an LLM think — sparse-autoencoder features firing live, in your browser, no GPU. open-source, MIT."*
2. *what you're seeing*: heatmap rows = features, columns = tokens, color = activation. Drag the layer slider to watch the logit lens "develop" toward the final prediction.
3. *the honesty rule*: every label gets a tier badge — MEASURED / SOURCED / AUTO-LABEL. We do not say what a feature "means" without a citation.
4. *what's under the hood*: HuggingFace transformers + forward hooks + SAELens + a tiny FastAPI SSE stream. Frontend is one HTML file plus 6 vanilla-JS modules.
5. *call to action*: `make run` and you have it running on a laptop in 60 seconds. Repo + demo links. Tagging @NeelNanda5 @anthropic @decoderesearch.

### 4. Reddit

**r/MachineLearning** — `[P] see-the-ai-think: watch sparse-autoencoder features fire live in your browser, on a laptop`. Body: short, mirror the README's "what you're seeing" + "what it does not claim" sections. GIF embedded.

**r/LocalLLaMA** — `Watch what your model is doing, live, no GPU required.` Emphasize the laptop story.

**r/learnmachinelearning** — *not* on D-day; D+3 with a "I built this and learned X about SAEs" framing.

### 5. Newsletter outreach (sent the morning of)

Three short pitches via email — three sentences each. The GIF + the one-sentence pitch + a link.

- **AlphaSignal** (`@alphasignalai`) — usually picks up open-source ML tools the same week.
- **Latent Space** (`swyx`) — interpretability is squarely on their beat.
- **The Batch** (`andrewyng@deeplearning.ai`) — longer lead time; still send.
- **Ben Lorica's Gradient Flow** — interpretability angle.
- **Import AI** (Jack Clark) — interp is one of his recurring beats.

### 6. Personal network DMs (the day before)

- Anyone who has retweeted a mech-interp paper in the last 30 days and might amplify a clean demo: Neel Nanda, Jess Smith, Trenton Bricken, Connor Kissane, Joseph Bloom (because we use his SAEs — say thanks explicitly), Decode Research folks.
- *Don't* ask for an RT — share the GIF and "this is shipping tomorrow." People who like it amplify it themselves.

## D+1 through D+7

- Track stars hourly the first 6 hours, then daily. Stable spike pattern: 500–1500 on day one, settling at ~3× day-one over 7 days if the GIF lands.
- Reply to every Show HN comment within 90 minutes — top-of-page for 8+ hours is the usual reward.
- If the Space is overloaded: bump it to `cpu-upgrade` for the first 48 hours. The Show HN top comment about a broken demo is brutal.
- File one polish PR per day in the open — visible activity helps the project look alive.

## D+8 through D+30

- **Two follow-ups** (interpretability has natural amplifiers — keep showing them new things):
  1. **Gemma-2-2B + Gemma Scope** support (v0.2). Refusal-feature spotlight visual is the lead GIF.
  2. **Feature steering** if v0.2 makes it convincing. Toggle a feature, watch the output shift — if it works cleanly, this is the most viral 8 seconds the project produces.

- Talk slot: pitch to ICLR / NeurIPS demo tracks, the EleutherAI Interp reading group, and any open mech-interp meetup. A 10-min screencast > a 30-min slide deck.

## D+30 through D+90

- Track GitHub stars as the success metric; keep the launch-readiness checklist (`PLANNING/06_review.md`) updated.
- Convert at least one piece of public coverage into a permanent README section (under "Press" / "In the wild").
- If a citing paper or a teaching course uses this, link to them.

## Channels to skip (and why)

- **LinkedIn**: ML researcher network here is small and unaffectionate toward "watch an LLM think" framing.
- **Mastodon**: amplifies if the project has a Bluesky thread; not worth a separate post.
- **TikTok / YouTube Shorts**: doable, but the GIF is already optimized for the same dwell-time loop on X.

## What success looks like

- ~2000 stars in 7 days (Show HN top + X amplification).
- ~5000 stars in 30 days (newsletters + follow-ups).
- One academic/interpretability blog mention in 30 days.
- One teaching reference in 90 days.

— Betty Guo
