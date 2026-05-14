# Honesty rules

A tool that visualizes the internals of a neural network is one click from claiming things it cannot back up. The whole genre of interpretability has a credibility problem precisely because demos overreach what the methods show.

This document is the contract that this tool honors. Every change to the UI is checked against it.

## Three tiers

Every interpretive claim served by the backend or rendered in the UI carries one of three pills:

| Pill | When to use it | Example |
|---|---|---|
| `MEASURED` | A direct activation value or probability. No interpretation. | "Feature L8/F1024 activates with magnitude 4.2 on token ` Marie`." |
| `SOURCED` | A label backed by a peer-reviewed paper, technical report, or human-labeled artifact. **Citation shown next to the label.** | "Feature 11/23800: end-of-sentence position. [Bricken et al. 2023, §4.2]" |
| `AUTO-LABEL` | A label produced by an automated interpretation pipeline (Neuronpedia auto-interp, etc.). Visibly marked speculative. | "Feature 6/12: activates on the first token of a word that follows a sentence-ending period. (Auto-generated, may be wrong.)" |

A label without a tier may not be shipped. Period.

## What the UI may say

- "The activation at L8/F1024 on this token is 4.2."
- "The top-5 layer-12 logits include ` dog` at 41% probability."
- "Feature L6/F12 fired on positions 3, 5, 12 of your prompt."
- "Auto-label (Neuronpedia, may be wrong): activates on first tokens of words following a period."
- "Layer 11 attended most strongly from position 7 to position 3 with weight 0.42."

## What the UI must never say

- "Feature L8/F1024 is the proper-noun feature." (Without a citation.)
- "The model has *decided* to say ` dog` by layer 11." (Causal language — only correlational evidence is shown.)
- "This activation means the model is being deceptive." (Speculative; not supported by the measurements.)
- "Turning off this feature makes the model refuse." (We don't ship feature steering for this reason in v0.1.)

## Enforcement

- The schema (`backend/server/schemas.py::FeatureLabelOut.tier`) is constrained to the three tier values. A label without a tier cannot be serialized.
- Labels are loaded from `backend/data/feature_labels.json`; the loader (`backend/features/labels.py::load_labels`) drops any entry with an invalid tier.
- The frontend always renders the tier pill before the label text.
- The right-side panel includes a fixed "What this means" paragraph that re-states this contract for every feature.

## What this implies for contributors

If you add a label, decide which tier it belongs to and include the source. If you don't have a source, the right tier is `AUTO-LABEL`, and the source field should say so honestly (e.g., "auto-interp via gpt-4o on 2024-12-01").

If the label is speculation you came up with, do not ship it.

— Betty Guo
