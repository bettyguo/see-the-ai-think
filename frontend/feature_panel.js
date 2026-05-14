// Right-side panel: clicked feature → triggering tokens, label with tier badge,
// and an honesty note. Fetches /api/feature/{model}/{layer}/{feature}; falls
// back to inline bundled labels if the API is unavailable.

const BUNDLED = {
  "gpt2-small:6:12": {
    label: { text: "activates on the first token of a word that follows a sentence-ending period",
             tier: "AUTO-LABEL", source: "Neuronpedia auto-interp (gpt2-small-res-jb)" },
    top_corpus_examples: [
      { text: "The day was bright. Sunlight streamed through the window.", activating_index: 5, activation: 3.7 },
      { text: "She paused. Then she began to write.", activating_index: 3, activation: 3.4 },
    ],
  },
  "gpt2-small:8:1024": {
    label: { text: "activates on tokens that begin proper nouns (people, places)",
             tier: "AUTO-LABEL", source: "Neuronpedia auto-interp (gpt2-small-res-jb)" },
    top_corpus_examples: [
      { text: "Marie Curie won the Nobel Prize twice.", activating_index: 0, activation: 4.2 },
      { text: "The Eiffel Tower stands in Paris.", activating_index: 1, activation: 3.9 },
    ],
  },
};

const HONESTY = "Activation magnitudes are real. Any text label is auto-generated or community-sourced (see the tier badge) and may be wrong.";

export class FeaturePanel {
  constructor(panelEl, layoutEl, opts) {
    this.el = panelEl;
    this.layoutEl = layoutEl;
    this.model = opts.model || "gpt2-small";
    this.labelsCache = new Map();   // key -> {text, tier, source}
    this.onClose = opts.onClose || (() => {});
    this.currentSessionTokens = [];
  }

  setSessionTokens(tokens) { this.currentSessionTokens = tokens; }
  setModel(name) { this.model = name; }

  hide() {
    this.el.classList.add("hidden");
    this.layoutEl.classList.remove("panel-open");
    this.onClose();
  }

  async open(layer, feature) {
    const key = `${this.model}:${layer}:${feature}`;
    const detail = await this._fetchDetail(layer, feature, key);
    this._render(layer, feature, detail);
    if (detail.label) this.labelsCache.set(`L${layer}/F${feature}`, detail.label);
    this.el.classList.remove("hidden");
    this.layoutEl.classList.add("panel-open");
  }

  async _fetchDetail(layer, feature, key) {
    try {
      const r = await fetch(`/api/feature/${this.model}/${layer}/${feature}`);
      if (r.ok) return await r.json();
    } catch (_) { /* fall through to bundled */ }
    const bundled = BUNDLED[key];
    if (bundled) return { ...bundled, honesty_note: HONESTY };
    return { label: null, top_corpus_examples: [], honesty_note: HONESTY };
  }

  _render(layer, feature, detail) {
    const sessionHits = this._sessionHitsFor(layer, feature);
    const label = detail.label;
    const tierClass = label ? this._tierClass(label.tier) : null;
    const tierLabel = label ? label.tier : null;

    this.el.innerHTML = `
      <button class="panel-close" aria-label="close">×</button>
      <h3>Feature inspector</h3>
      <div class="feature-id">L${layer} · feature ${feature}</div>
      ${label ? `
        <div>
          <span class="pill ${tierClass}">${escape(tierLabel)}</span>
          <span class="label-text">${escape(label.text)}</span>
        </div>
        <div class="source">source: ${escape(label.source)}${label.tier === "AUTO-LABEL" ? " · auto-generated, may be wrong" : ""}</div>
      ` : `<div class="label-text"><em>no human or auto-label available for this feature yet</em></div>`}

      <h4>Activations in this prompt</h4>
      <ul>
        ${sessionHits.length ? sessionHits.map(h => `
          <li>
            <span>${escape(this._displayToken(h.text))}</span>
            <span style="float:right;color:#3a86ff">${h.act.toFixed(2)}</span>
            <span class="token-bar" style="width:${Math.min(120, h.act * 22)}px"></span>
          </li>`).join("") : `<li><em>no positive activations in the current prompt</em></li>`}
      </ul>

      <h4>Triggering examples (corpus)</h4>
      <ul>
        ${(detail.top_corpus_examples || []).length ? (detail.top_corpus_examples || []).map(ex => `
          <li><span class="example-text">${this._renderExample(ex)}</span></li>
        `).join("") : `<li><em>no bundled corpus examples for this feature</em></li>`}
      </ul>

      <h4>What this means</h4>
      <p style="font-size:11px;color:#8d94a7;line-height:1.5;margin:0">${escape(detail.honesty_note || HONESTY)}</p>
    `;
    this.el.querySelector(".panel-close").addEventListener("click", () => this.hide());
  }

  _sessionHitsFor(layer, feature) {
    const hits = [];
    for (const tok of this.currentSessionTokens) {
      for (const f of tok.top_features || []) {
        if (f.layer === layer && f.feature === feature) {
          hits.push({ text: tok.text, act: f.act });
        }
      }
    }
    hits.sort((a, b) => b.act - a.act);
    return hits.slice(0, 10);
  }

  _renderExample(ex) {
    // Highlight the activating index inside the example. Naively split on
    // whitespace; the bundled examples are short enough for that.
    const text = String(ex.text || "");
    const parts = text.split(/(\s+)/);  // keep separators
    let wordIdx = -1;
    const out = parts.map(p => {
      if (/^\s+$/.test(p)) return p;
      wordIdx++;
      if (wordIdx === ex.activating_index) return `<mark>${escape(p)}</mark>`;
      return escape(p);
    }).join("");
    return out;
  }

  _tierClass(tier) {
    if (tier === "MEASURED") return "measured";
    if (tier === "SOURCED") return "sourced";
    return "auto";
  }

  _displayToken(t) {
    return (t || "").replace(/^ /, "·").replace(/\n/g, "⏎");
  }
}

function escape(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
