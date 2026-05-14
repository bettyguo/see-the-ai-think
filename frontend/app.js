// see-the-ai-think — top-level page controller.
//
// Owns the runtime state (current capture, status), wires examples, runs
// either /api/generate (live SSE) or the bundled demo capture, and routes
// data to the three visualization modules.

import { Heatmap } from "/static/heatmap.js";
import { LogitLens } from "/static/logit_lens.js";
import { FeaturePanel } from "/static/feature_panel.js";

const els = {
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  prompt: document.getElementById("prompt"),
  example: document.getElementById("example-select"),
  maxNew: document.getElementById("max-new-tokens"),
  temp: document.getElementById("temperature"),
  runBtn: document.getElementById("run-btn"),
  demoBtn: document.getElementById("demo-btn"),
  banner: document.getElementById("banner"),
  tokenRow: document.getElementById("token-row"),
  heatmapCanvas: document.getElementById("heatmap"),
  legend: document.getElementById("feature-legend"),
  layerSlider: document.getElementById("layer-slider"),
  layerReadout: document.getElementById("layer-readout"),
  logitLens: document.getElementById("logit-lens"),
  featurePanel: document.getElementById("feature-panel"),
  layout: document.querySelector(".layout"),
};

const state = {
  model: "gpt2-small",
  promptLen: 0,
  tokens: [],
  running: false,
};

const heatmap = new Heatmap(els.heatmapCanvas, els.legend, { featureLabels: new Map() });
const logitLens = new LogitLens(els.logitLens, els.layerSlider, els.layerReadout);
const panel = new FeaturePanel(els.featurePanel, els.layout, {
  model: state.model,
  onClose: () => heatmap.setActiveFeature(null, null),
});

heatmap.setOnFeatureClick((layer, feature) => {
  heatmap.setActiveFeature(layer, feature);
  panel.setSessionTokens(state.tokens);
  panel.open(layer, feature);
});

// ------------------------------------------------------------------
// status

function setStatus(kind, text) {
  els.status.className = "status " + kind;
  els.statusText.textContent = text;
}

function showBanner(text, isError = false) {
  els.banner.textContent = text;
  els.banner.classList.remove("hidden");
  els.banner.classList.toggle("error", isError);
}
function hideBanner() { els.banner.classList.add("hidden"); }

// ------------------------------------------------------------------
// examples

async function loadExamples() {
  try {
    const r = await fetch("/api/examples");
    if (!r.ok) throw new Error("no examples");
    const items = await r.json();
    populateExamples(items);
  } catch {
    populateExamples([{ id: "fox", title: "The famous one", prompt: "The quick brown fox jumps over the lazy", why: "" }]);
  }
}

function populateExamples(items) {
  els.example.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "— pick an example —";
  els.example.appendChild(blank);
  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = it.id;
    opt.textContent = it.title || it.id;
    opt.dataset.prompt = it.prompt;
    opt.dataset.why = it.why || "";
    els.example.appendChild(opt);
  }
  els.example.addEventListener("change", () => {
    const opt = els.example.selectedOptions[0];
    if (opt && opt.dataset.prompt) {
      els.prompt.value = opt.dataset.prompt;
      if (opt.dataset.why) showBanner(opt.dataset.why);
      else hideBanner();
    }
  });
}

// ------------------------------------------------------------------
// token row

function renderTokenRow() {
  els.tokenRow.innerHTML = "";
  state.tokens.forEach((tok, i) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    if (i >= state.promptLen) chip.classList.add("generated");
    if (i === state.tokens.length - 1) chip.classList.add("latest");
    chip.textContent = tok.text.replace(/^ /, "·").replace(/\n/g, "⏎") || "·";
    chip.title = `position ${tok.position} · token id ${tok.token_id}`;
    els.tokenRow.appendChild(chip);
  });
  els.tokenRow.scrollLeft = els.tokenRow.scrollWidth;
}

// ------------------------------------------------------------------
// run modes

function resetSession() {
  state.tokens = [];
  state.promptLen = 0;
  heatmap.reset();
  logitLens.reset();
  panel.hide();
  els.tokenRow.innerHTML = "";
}

function ingestToken(tok) {
  state.tokens.push(tok);
  heatmap.pushToken(tok);
  logitLens.pushToken(tok);
  renderTokenRow();
}

async function runLive() {
  if (state.running) return;
  state.running = true;
  resetSession();
  hideBanner();
  setStatus("live", "running…");

  const body = {
    prompt: els.prompt.value,
    max_new_tokens: clampInt(els.maxNew.value, 1, 64, 16),
    temperature: clampFloat(els.temp.value, 0, 2, 0.7),
    top_k_features: 32,
    top_k_logits: 5,
  };

  let response;
  try {
    response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    state.running = false;
    setStatus("error", "network error");
    showBanner("network error — falling back to demo mode.", true);
    await runDemo();
    return;
  }
  if (response.status === 503 || !response.ok) {
    state.running = false;
    setStatus("demo", "demo mode");
    showBanner("model not loaded on the server — animating bundled demo capture. (Run `make run` locally to use a real model.)");
    await runDemo();
    return;
  }
  await consumeSSE(response);
  state.running = false;
  setStatus("connected", "done");
}

async function consumeSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const parsed = parseSSEFrame(frame);
      if (!parsed) continue;
      onSSEEvent(parsed.event, parsed.data);
    }
  }
}

function parseSSEFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

function onSSEEvent(event, data) {
  if (event === "meta") {
    state.promptLen = 0; // server sends prompt + generated; we mark "generated" later via position
    if (!data.sae_loaded) {
      showBanner((data.notes && data.notes[0]) || "SAE unavailable — showing raw neuron activations.");
    }
    state.model = data.model;
    panel.setModel(state.model);
    return;
  }
  if (event === "token") {
    ingestToken(data);
    return;
  }
  if (event === "done") {
    return;
  }
}

async function runDemo() {
  resetSession();
  setStatus("demo", "demo mode");
  let demo;
  try {
    const r = await fetch("/static/demo_capture.json");
    if (!r.ok) throw new Error();
    demo = await r.json();
  } catch {
    setStatus("error", "demo capture missing");
    return;
  }
  state.model = demo.meta.model;
  panel.setModel(state.model);
  state.promptLen = demo.prompt_length || 0;
  if (demo.meta.notes && demo.meta.notes[0]) showBanner(demo.meta.notes[0]);
  // animate token-by-token at ~7 Hz so it looks alive
  for (const tok of demo.tokens) {
    ingestToken(tok);
    await sleep(140);
  }
  setStatus("connected", "demo done");
}

// ------------------------------------------------------------------
// util

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function clampInt(v, lo, hi, def) {
  const n = parseInt(v, 10);
  if (!Number.isFinite(n)) return def;
  return Math.max(lo, Math.min(hi, n));
}
function clampFloat(v, lo, hi, def) {
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return def;
  return Math.max(lo, Math.min(hi, n));
}

// ------------------------------------------------------------------
// boot

async function boot() {
  setStatus("connecting", "connecting…");
  await loadExamples();
  try {
    const r = await fetch("/api/health");
    if (r.ok) {
      const j = await r.json();
      if (j.ok) {
        setStatus("connected", j.sae_loaded ? "model + SAE ready" : "model ready · no SAE");
      } else {
        setStatus("demo", "demo mode");
      }
    }
  } catch {
    setStatus("demo", "static / demo mode");
  }
  // Show the bundled demo immediately so a visitor sees something within ~50ms.
  await runDemo();
}

els.runBtn.addEventListener("click", runLive);
els.demoBtn.addEventListener("click", () => { state.running = false; runDemo(); });
els.prompt.addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") runLive();
});

boot();
