// Attention arcs — for the selected token, draw arcs above the token row
// pointing at the source positions it attended to most, colored by layer.
//
// Data per token: `attn_top_per_layer[L]` = [{head, src, weight}, ...]
// We render the top-K (= 3) src per layer; layer index controls color.

const LAYER_PALETTE = [
  "#1f2433", "#26345a", "#2c4882", "#345ab8", "#3a86ff",
  "#5093ff", "#67a3ff", "#83b5fe", "#a3c8fc", "#ffd166",
  "#ffaa5d", "#ef476f",
];

export class AttentionArcs {
  constructor(svg) {
    this.svg = svg;
    this.tokens = [];
    this.selectedIdx = null;
    this._installResize();
  }

  reset() {
    this.tokens = [];
    this.selectedIdx = null;
    this._render();
  }

  pushToken(tok) {
    this.tokens.push(tok);
    if (this.selectedIdx == null) this.selectedIdx = this.tokens.length - 1;
    this._render();
  }

  pushTokens(arr) {
    for (const t of arr) this.pushToken(t);
  }

  selectToken(idx) {
    if (idx < 0 || idx >= this.tokens.length) return;
    this.selectedIdx = idx;
    this._render();
  }

  _installResize() {
    window.addEventListener("resize", () => this._render());
  }

  _render() {
    const svg = this.svg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const T = this.tokens.length;
    if (!T) { this._placeholder(); return; }

    const W = 800;
    const H = 180;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const padX = 20;
    const baselineY = H - 22;
    const cellW = (W - padX * 2) / T;

    // Token labels along the baseline.
    for (let i = 0; i < T; i++) {
      const x = padX + cellW * (i + 0.5);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x);
      t.setAttribute("y", baselineY + 14);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", i === this.selectedIdx ? "#ffd166" : "#8d94a7");
      t.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
      t.setAttribute("font-size", "11");
      const txt = (this.tokens[i].text || "·").replace(/^ /, "·").replace(/\n/g, "⏎");
      t.textContent = txt.length > 8 ? txt.slice(0, 7) + "…" : txt;
      t.style.cursor = "pointer";
      t.addEventListener("click", () => this.selectToken(i));
      svg.appendChild(t);

      // baseline tick
      const tick = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      tick.setAttribute("cx", x);
      tick.setAttribute("cy", baselineY);
      tick.setAttribute("r", i === this.selectedIdx ? 4 : 2.5);
      tick.setAttribute("fill", i === this.selectedIdx ? "#ffd166" : "#5a607a");
      tick.style.cursor = "pointer";
      tick.addEventListener("click", () => this.selectToken(i));
      svg.appendChild(tick);
    }

    const tok = this.tokens[this.selectedIdx];
    if (!tok || !tok.attn_top_per_layer) { return; }

    // For each layer, draw an arc from the selected position to each top src.
    const layers = tok.attn_top_per_layer;
    const fromX = padX + cellW * (this.selectedIdx + 0.5);
    for (let L = 0; L < layers.length; L++) {
      const color = LAYER_PALETTE[Math.min(LAYER_PALETTE.length - 1, L)];
      for (const entry of layers[L] || []) {
        const src = entry.src ?? entry[1];
        const w = entry.weight ?? entry[2] ?? 0;
        if (src == null || src >= T) continue;
        const toX = padX + cellW * (src + 0.5);
        const dx = Math.abs(fromX - toX);
        const arcHeight = Math.min(baselineY - 16, 18 + dx * 0.35 + L * 4);
        const midX = (fromX + toX) / 2;
        const midY = baselineY - arcHeight;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${fromX} ${baselineY} Q ${midX} ${midY} ${toX} ${baselineY}`);
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", Math.max(0.5, w * 2.5));
        path.setAttribute("stroke-opacity", Math.max(0.18, Math.min(0.9, w)));
        path.setAttribute("fill", "none");
        path.setAttribute("stroke-linecap", "round");
        path.appendChild(_title(`L${L} · src ${src} · weight ${w.toFixed(2)}`));
        svg.appendChild(path);
      }
    }

    // Legend (top-left).
    const legend = document.createElementNS("http://www.w3.org/2000/svg", "text");
    legend.setAttribute("x", 12);
    legend.setAttribute("y", 16);
    legend.setAttribute("fill", "#5a607a");
    legend.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
    legend.setAttribute("font-size", "10");
    legend.textContent = `attending from "${(tok.text || '?').replace(/^ /, '·')}" · arc thickness = attention weight · color = layer`;
    svg.appendChild(legend);
  }

  _placeholder() {
    const svg = this.svg;
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 12);
    t.setAttribute("y", 22);
    t.setAttribute("fill", "#5a607a");
    t.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
    t.setAttribute("font-size", "12");
    t.textContent = "no captures yet — press ▶ run.";
    svg.appendChild(t);
  }
}

function _title(text) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
  t.textContent = text;
  return t;
}
