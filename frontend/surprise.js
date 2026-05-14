// Surprise ribbon — color-coded entropy of the final-layer next-token
// distribution per position. Low entropy = certain = green; high entropy =
// surprised = warm/red. This is a cheap visual hint about where the model
// "knew what was coming" vs "had to think about it."

export class SurpriseRibbon {
  constructor(el) {
    this.el = el;
    this.cells = [];
  }

  reset() {
    this.cells = [];
    this.el.innerHTML = "";
  }

  pushToken(tok) {
    const ent = entropyFromTopK(this._finalLayer(tok));
    const cell = document.createElement("span");
    cell.className = "surprise-cell";
    cell.style.background = entropyColor(ent);
    cell.title = `entropy ${ent.toFixed(2)} (top-5 only — true entropy is higher)`;
    this.el.appendChild(cell);
    this.cells.push(cell);
  }

  pushTokens(arr) {
    for (const t of arr) this.pushToken(t);
  }

  _finalLayer(tok) {
    const ll = tok.logits_per_layer || [];
    return ll[ll.length - 1] || { probs: [] };
  }
}

function entropyFromTopK(layer) {
  // Approximation: entropy over the top-K we have (so it under-counts true
  // entropy, but the *trend* across tokens is what we want to show).
  const probs = layer.probs || [];
  if (!probs.length) return 0;
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

function entropyColor(h) {
  // h ranges ~0 (certain) .. ~1.6 (uniform over 5 tokens = ln(5))
  const ln5 = Math.log(5);
  const t = Math.max(0, Math.min(1, h / ln5));
  // green (certain) -> blue (medium) -> amber (high) -> hot pink (very high)
  const stops = [
    [6, 214, 160],
    [58, 134, 255],
    [255, 209, 102],
    [239, 71, 111],
  ];
  const idxF = t * (stops.length - 1);
  const i = Math.floor(idxF);
  const f = idxF - i;
  const a = stops[i];
  const b = stops[Math.min(stops.length - 1, i + 1)];
  const R = Math.round(a[0] + (b[0] - a[0]) * f);
  const G = Math.round(a[1] + (b[1] - a[1]) * f);
  const B = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${R},${G},${B})`;
}
