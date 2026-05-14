// Canvas heatmap. The hero visual.
//
// Rows = top-N most-active features across the prompt so far,
// columns = tokens. Cell color = activation magnitude on a perceptual ramp.
// As new tokens stream in, columns slide; as new features enter the top-N,
// rows re-rank with a brief animation.

const MAX_ROWS = 24;          // shown rows; the data may carry more
const CELL_H = 16;            // px
const CELL_GAP = 1;
const MIN_CELL_W = 14;
const PALETTE = [
  // dark navy -> blue -> cyan -> amber -> hot pink
  [11, 13, 20],
  [26, 38, 70],
  [58, 134, 255],
  [120, 220, 240],
  [255, 209, 102],
  [239, 71, 111],
];

function rampColor(t) {
  // t in [0,1] — piecewise-linear across PALETTE
  if (!Number.isFinite(t) || t <= 0) return PALETTE[0];
  if (t >= 1) return PALETTE[PALETTE.length - 1];
  const idxF = t * (PALETTE.length - 1);
  const i = Math.floor(idxF);
  const f = idxF - i;
  const a = PALETTE[i];
  const b = PALETTE[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export class Heatmap {
  constructor(canvas, legendEl, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.legend = legendEl;
    this.opts = opts || {};
    this.tokens = [];              // [{position, text, top_features:[{layer,feature,act}], ...}]
    this.featureRows = [];         // current row order: array of "L{layer}/F{fid}"
    this.featureMax = new Map();   // key -> max activation seen
    this.activeKey = null;         // selected feature key
    this.onFeatureClick = null;
    this.tooltipEl = null;
    this._installTooltip();
    this._installInteractions();
    this._dpr = window.devicePixelRatio || 1;
  }

  setOnFeatureClick(fn) { this.onFeatureClick = fn; }

  reset() {
    this.tokens = [];
    this.featureRows = [];
    this.featureMax.clear();
    this.activeKey = null;
    this._render();
  }

  pushToken(tok) {
    this.tokens.push(tok);
    for (const f of tok.top_features || []) {
      const k = this._key(f.layer, f.feature);
      const prev = this.featureMax.get(k) || 0;
      if (f.act > prev) this.featureMax.set(k, f.act);
    }
    this._reRank();
    this._render();
  }

  pushTokens(arr) {
    for (const t of arr) this.pushToken(t);
  }

  setActiveFeature(layer, feature) {
    this.activeKey = (layer == null) ? null : this._key(layer, feature);
    this._render();
  }

  // ------------------------------------------------------------------

  _key(layer, feature) { return `L${layer}/F${feature}`; }

  _reRank() {
    // Pick top MAX_ROWS feature keys by total activation across tokens so far.
    const score = new Map();
    for (const tok of this.tokens) {
      for (const f of tok.top_features || []) {
        const k = this._key(f.layer, f.feature);
        score.set(k, (score.get(k) || 0) + f.act);
      }
    }
    const sorted = [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_ROWS);
    this.featureRows = sorted.map(([k]) => k);
  }

  _activationAt(tokenIdx, key) {
    const tok = this.tokens[tokenIdx];
    if (!tok) return 0;
    for (const f of tok.top_features || []) {
      if (this._key(f.layer, f.feature) === key) return f.act;
    }
    return 0;
  }

  _globalMaxAct() {
    let m = 0;
    for (const v of this.featureMax.values()) if (v > m) m = v;
    return m || 1;
  }

  _render() {
    const T = this.tokens.length;
    const rows = this.featureRows.length;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== w * this._dpr || this.canvas.height !== h * this._dpr) {
      this.canvas.width = Math.max(1, w * this._dpr);
      this.canvas.height = Math.max(1, h * this._dpr);
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.scale(this._dpr, this._dpr);
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "#0d101a";
    ctx.fillRect(0, 0, w, h);

    if (!T || !rows) {
      ctx.restore();
      this._renderLegend();
      return;
    }

    const cellW = Math.max(MIN_CELL_W, Math.floor((w - (T - 1) * CELL_GAP) / T));
    const totalH = rows * (CELL_H + CELL_GAP);
    const yOffset = Math.max(0, Math.min(h - totalH, 0));
    const gMax = this._globalMaxAct();

    for (let r = 0; r < rows; r++) {
      const key = this.featureRows[r];
      const isActive = key === this.activeKey;
      for (let c = 0; c < T; c++) {
        const a = this._activationAt(c, key);
        const t = a > 0 ? Math.min(1, a / gMax) : 0;
        const [R, G, B] = rampColor(t);
        const x = c * (cellW + CELL_GAP);
        const y = yOffset + r * (CELL_H + CELL_GAP);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(x, y, cellW, CELL_H);
        if (isActive && a > 0) {
          ctx.strokeStyle = "rgba(255,209,102,0.9)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, cellW - 1, CELL_H - 1);
        }
      }
    }
    ctx.restore();
    this._renderLegend();
  }

  _renderLegend() {
    if (!this.legend) return;
    const labels = this.opts.featureLabels || new Map();  // optional cache
    this.legend.innerHTML = "";
    for (const key of this.featureRows) {
      const li = document.createElement("li");
      li.dataset.key = key;
      if (key === this.activeKey) li.classList.add("active");
      const [lpart, fpart] = key.split("/");
      const lf = document.createElement("span");
      lf.className = "lf";
      lf.textContent = `${lpart}/${fpart}`;
      const lbl = document.createElement("span");
      lbl.className = "lbl";
      const cached = labels.get(key);
      lbl.textContent = cached ? cached.text : "";
      lbl.title = cached ? `${cached.tier} · ${cached.source}` : "click for details";
      li.append(lf, lbl);
      li.addEventListener("click", () => {
        const [layer, feature] = this._parseKey(key);
        if (this.onFeatureClick) this.onFeatureClick(layer, feature);
      });
      this.legend.appendChild(li);
    }
  }

  _parseKey(key) {
    const m = key.match(/^L(\d+)\/F(\d+)$/);
    if (!m) return [null, null];
    return [parseInt(m[1], 10), parseInt(m[2], 10)];
  }

  // ------------------------------------------------------------------

  _installInteractions() {
    this.canvas.addEventListener("click", (ev) => {
      const hit = this._hitTest(ev);
      if (!hit) return;
      if (this.onFeatureClick) this.onFeatureClick(hit.layer, hit.feature);
    });
    this.canvas.addEventListener("mousemove", (ev) => {
      const hit = this._hitTest(ev);
      if (!hit) { this._hideTooltip(); return; }
      const tok = this.tokens[hit.col];
      const text = tok ? tok.text : "?";
      this._showTooltip(ev.clientX, ev.clientY,
        `${text}\nL${hit.layer}/F${hit.feature} · act ${hit.act.toFixed(2)}`);
    });
    this.canvas.addEventListener("mouseleave", () => this._hideTooltip());
    window.addEventListener("resize", () => this._render());
  }

  _hitTest(ev) {
    const rect = this.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const T = this.tokens.length;
    const rows = this.featureRows.length;
    if (!T || !rows) return null;
    const w = this.canvas.clientWidth;
    const cellW = Math.max(MIN_CELL_W, Math.floor((w - (T - 1) * CELL_GAP) / T));
    const col = Math.floor(x / (cellW + CELL_GAP));
    const row = Math.floor(y / (CELL_H + CELL_GAP));
    if (col < 0 || col >= T || row < 0 || row >= rows) return null;
    const key = this.featureRows[row];
    const [layer, feature] = this._parseKey(key);
    const a = this._activationAt(col, key);
    return { col, row, layer, feature, act: a };
  }

  _installTooltip() {
    this.tooltipEl = document.createElement("div");
    this.tooltipEl.className = "tooltip";
    this.tooltipEl.style.display = "none";
    document.body.appendChild(this.tooltipEl);
  }
  _showTooltip(x, y, text) {
    if (!this.tooltipEl) return;
    this.tooltipEl.textContent = text;
    this.tooltipEl.style.left = `${x + 12}px`;
    this.tooltipEl.style.top = `${y + 12}px`;
    this.tooltipEl.style.display = "block";
  }
  _hideTooltip() {
    if (this.tooltipEl) this.tooltipEl.style.display = "none";
  }
}
