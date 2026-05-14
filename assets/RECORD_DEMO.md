# Recording the demo GIF

The hero of the README is `assets/demo.gif`. This file lists exactly which moments to capture so the recording is consistent and replicable. **Total target: 10–15 seconds.**

## Setup

1. `make run` (or `make run-fast` for laptops with slow bandwidth — SAE off).
2. Open `http://127.0.0.1:8000` in a clean Chromium window at **1280 × 800**.
3. System dark mode on. Hide bookmarks bar, taskbar, dock.
4. Use the **fox prompt** (`The quick brown fox jumps over the lazy`) — it is the most iconic and produces the cleanest logit-lens convergence to ` dog`.

## Recording tool

- macOS: built-in screen recorder (Cmd-Shift-5), then convert to GIF with `gifski`:
  ```
  gifski --fps 24 --width 1280 -o assets/demo.gif demo.mov
  ```
- Linux: `peek` or `OBS` → mp4 → `gifski`.
- Windows: ShareX or OBS → mp4 → `gifski`.

Target file size ≤ 6 MB so GitHub displays it inline.

## Shot list

| t (s) | Action | What viewers should see |
|---|---|---|
| 0.0 | Page already loaded, demo capture animating | Cells lighting up across the heatmap as tokens appear left-to-right |
| 2.0 | Click the **fox prompt** in the example dropdown (or just press ▶ run with it pre-filled) | Token row populates; heatmap *re-ranks* with rows reshuffling |
| 4.0 | Hover one bright cell on the `fox` column | Tooltip appears: `· fox L9/F11400 · act 5.10` |
| 5.5 | Click the row label for the brightest feature | Right panel slides in showing the auto-label with the amber `AUTO-LABEL` badge and a list of triggering tokens |
| 8.0 | Drag the **layer slider** from 12 → 0 → 12 | Logit-lens bars morph through the layers; final layer locks onto ` dog` at 40%+ |
| 12.0 | Pause on layer 12 showing top prediction ` dog` | Last frame is screenshot-worthy: clearly says ` dog` |

## Stills to also capture (for the README's "what you're seeing" section)

1. `assets/shot-heatmap.png` — heatmap fully populated, no panel.
2. `assets/shot-panel.png` — feature panel open showing label + tier badge + triggering examples.
3. `assets/shot-lens.png` — logit-lens scrubber at layer 12 showing ` dog` on top.

## What the GIF needs to convey in 10 seconds

1. **Something is moving** (heatmap cells lighting up).
2. **It's a model thinking** (token-by-token, layers, attention-style colors).
3. **You can interact** (click, hover, slide).
4. **It says something meaningful** (the panel surfaces a labeled feature; the lens locks onto a recognizable word).

If a frame in the recording doesn't push at least one of those four forward, cut it.
