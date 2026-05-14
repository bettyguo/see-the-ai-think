// Logit-lens scrubber — for the latest token, show the top-5 next-token
// predictions from each layer's residual. Slider picks the layer; bars
// transition smoothly.

const BAR_GAP = 8;
const BAR_W = 120;
const LABEL_W = 70;
const PROB_W = 50;
const ROW_H = 32;

export class LogitLens {
  constructor(svg, slider, readout) {
    this.svg = svg;
    this.slider = slider;
    this.readout = readout;
    this.tokens = [];
    this.currentLayer = null;
    this.maxLayer = 12;
    this._init();
  }

  reset() {
    this.tokens = [];
    this._render();
  }

  pushToken(tok) {
    this.tokens.push(tok);
    if (this.tokens.length === 1) {
      const n = (tok.logits_per_layer || []).length;
      this.maxLayer = Math.max(0, n - 1);
      this.slider.min = 0;
      this.slider.max = this.maxLayer;
      this.slider.value = this.maxLayer;
      this.currentLayer = this.maxLayer;
      this.readout.textContent = String(this.maxLayer);
    }
    this._render();
  }

  pushTokens(arr) {
    for (const t of arr) this.pushToken(t);
  }

  _init() {
    this.slider.addEventListener("input", (ev) => {
      this.currentLayer = parseInt(ev.target.value, 10);
      this.readout.textContent = String(this.currentLayer);
      this._render();
    });
  }

  _render() {
    const svg = this.svg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!this.tokens.length || this.currentLayer == null) {
      this._placeholder();
      return;
    }
    const last = this.tokens[this.tokens.length - 1];
    const layers = last.logits_per_layer || [];
    const layer = layers[this.currentLayer];
    if (!layer) { this._placeholder(); return; }

    const W = 800;
    const H = 220;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    // Title row: which token are we predicting after?
    const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
    title.setAttribute("x", 12);
    title.setAttribute("y", 22);
    title.setAttribute("fill", "#8d94a7");
    title.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
    title.setAttribute("font-size", "12");
    title.textContent = `after token: "${last.text}"  ·  layer ${this.currentLayer} / ${this.maxLayer}`;
    svg.appendChild(title);

    const k = Math.min(5, (layer.tokens || []).length);
    const rowsTop = 50;
    const maxProb = Math.max(0.001, ...(layer.probs || []));

    for (let i = 0; i < k; i++) {
      const tokText = layer.tokens[i] ?? "?";
      const prob = layer.probs[i] ?? 0;
      const y = rowsTop + i * (ROW_H + 4);

      // token label
      const lab = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lab.setAttribute("x", 12);
      lab.setAttribute("y", y + ROW_H / 2 + 4);
      lab.setAttribute("fill", "#e6e8ef");
      lab.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
      lab.setAttribute("font-size", "13");
      lab.textContent = this._displayToken(tokText);
      svg.appendChild(lab);

      // bar background
      const bgw = W - 12 - LABEL_W - PROB_W - 24;
      const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bg.setAttribute("x", LABEL_W + 12);
      bg.setAttribute("y", y + 6);
      bg.setAttribute("width", bgw);
      bg.setAttribute("height", ROW_H - 12);
      bg.setAttribute("rx", 4);
      bg.setAttribute("fill", "#11141e");
      svg.appendChild(bg);

      // bar
      const ratio = Math.min(1, prob / maxProb);
      const barW = Math.max(2, ratio * bgw);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", LABEL_W + 12);
      rect.setAttribute("y", y + 6);
      rect.setAttribute("width", barW);
      rect.setAttribute("height", ROW_H - 12);
      rect.setAttribute("rx", 4);
      const hue = 215 - i * 15;
      const sat = 80 - i * 10;
      const light = 55 - i * 4;
      rect.setAttribute("fill", `hsl(${hue}, ${sat}%, ${light}%)`);
      rect.setAttribute("opacity", i === 0 ? 1.0 : 0.85);
      rect.style.transition = "width 120ms ease";
      svg.appendChild(rect);

      // probability readout
      const pr = document.createElementNS("http://www.w3.org/2000/svg", "text");
      pr.setAttribute("x", W - 12);
      pr.setAttribute("y", y + ROW_H / 2 + 4);
      pr.setAttribute("text-anchor", "end");
      pr.setAttribute("fill", "#8d94a7");
      pr.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
      pr.setAttribute("font-size", "12");
      pr.textContent = (prob * 100).toFixed(1) + "%";
      svg.appendChild(pr);
    }

    // small footer reminder
    const f = document.createElementNS("http://www.w3.org/2000/svg", "text");
    f.setAttribute("x", 12);
    f.setAttribute("y", H - 8);
    f.setAttribute("fill", "#5a607a");
    f.setAttribute("font-family", "ui-monospace, SFMono-Regular, monospace");
    f.setAttribute("font-size", "10");
    f.textContent = "logit lens: unembed(LN(residual)) → softmax. Probabilities at this layer; the final layer matches the model's actual prediction.";
    svg.appendChild(f);
  }

  _displayToken(t) {
    // BPE pieces include leading spaces — render visibly so the user can see whitespace.
    return t.replace(/^ /, "·").replace(/\n/g, "⏎");
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
