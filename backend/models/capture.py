"""Activation capture: forward hooks → per-layer residual + attention.

This module is the core of the backend. It runs a forward pass through an HF
model with `register_forward_hook` on each transformer block, collecting the
residual stream and attention pattern at every layer. The SAE encoding and
logit-lens projection live in `backend.features` and consume what this module
returns.

The CaptureEngine is intended for short prompts (~64 tokens). It supports two
modes:

  * analyze(prompt)              — one forward pass, all layers captured.
  * generate_stream(prompt, n)   — autoregressive, yields per-token captures.

Lazy torch import: the dataclasses at the bottom of this file are usable
without torch, so the server schemas and tests of pure data flow can run on a
torch-less machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.config import ModelSpec

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class FeatureHit:
    layer: int
    feature_id: int
    activation: float


@dataclass(frozen=True)
class LogitLensTop:
    """Top-K next-token predictions read out of one layer's residual stream."""

    layer: int
    tokens: list[str]
    probs: list[float]


@dataclass
class TokenCapture:
    """Everything we know about one position in the sequence."""

    position: int
    token_id: int
    text: str
    top_features: list[FeatureHit] = field(default_factory=list)
    logits_per_layer: list[LogitLensTop] = field(default_factory=list)
    attn_top_per_layer: list[list[tuple[int, int, float]]] = field(default_factory=list)


@dataclass
class CaptureResult:
    model_name: str
    prompt: str
    tokens: list[TokenCapture]
    n_layers: int
    sae_loaded: bool
    notes: list[str] = field(default_factory=list)


class CaptureEngine:
    """Owns a loaded model + tokenizer + hook plumbing + (optional) SAE bundle.

    Construction is cheap; `attach()` is what actually loads the model and
    installs hooks. This split keeps imports light for tests.
    """

    def __init__(self, spec_name: str = "gpt2-small", device: str = "cpu", enable_sae: bool = True):
        from backend.config import get_model_spec

        self.spec: ModelSpec = get_model_spec(spec_name)
        self.device = device
        self.enable_sae = enable_sae
        self._model: Any = None
        self._tokenizer: Any = None
        self._sae_bundle: Any = None
        self._handles: list[Any] = []
        self._residuals: list[Any] = []
        self._attentions: list[Any] = []

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Load model + tokenizer (+ SAE if enabled) and install hooks."""
        from backend.features.sae import SAEBundle, try_load_saes
        from backend.models.hf_loader import load_model_and_tokenizer

        self._model, self._tokenizer, self.spec = load_model_and_tokenizer(
            self.spec.name, device=self.device
        )
        self._install_hooks()
        if self.enable_sae:
            self._sae_bundle = try_load_saes(self.spec, device=self.device)
        else:
            self._sae_bundle = SAEBundle.empty()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._model = None
        self._tokenizer = None
        self._sae_bundle = None

    @property
    def sae_loaded(self) -> bool:
        return bool(self._sae_bundle and self._sae_bundle.is_loaded)

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("CaptureEngine not attached")
        return self._tokenizer

    # ------------------------------------------------------------------
    # hook plumbing
    # ------------------------------------------------------------------

    def _block_modules(self) -> list[Any]:
        """Return the transformer blocks in order. GPT-2 family only for now."""
        m = self._model
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return list(m.transformer.h)
        raise NotImplementedError(
            f"don't know how to find blocks on {type(m).__name__}; "
            "extend backend.models.capture._block_modules() to add support."
        )

    def _install_hooks(self) -> None:
        blocks = self._block_modules()
        self._residuals = [None] * (len(blocks) + 1)
        self._attentions = [None] * len(blocks)

        # Capture the embedding-output ("layer 0" residual) via the first block's pre-hook.
        def make_pre(idx: int):
            def pre(_module, inputs):
                hidden = inputs[0]
                self._residuals[idx] = hidden.detach()
            return pre

        # Capture the post-block residual and attention pattern.
        def make_post(idx: int):
            def post(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self._residuals[idx + 1] = hidden.detach()
                # GPT-2 returns attention weights when output_attentions=True
                # We capture them via a second hook on the inner attention module below.
            return post

        for i, block in enumerate(blocks):
            self._handles.append(block.register_forward_pre_hook(make_pre(i)))
            self._handles.append(block.register_forward_hook(make_post(i)))
        # Attention weights are surfaced via output_attentions=True in _forward_capture,
        # which is the cheapest portable path across HF GPT-2-family models.

    # ------------------------------------------------------------------
    # forward / capture
    # ------------------------------------------------------------------

    def _forward_capture(self, input_ids: Any, past_key_values: Any = None) -> dict[str, Any]:
        """One forward pass; returns residuals (list of [T,D]), attentions, logits, kv."""
        import torch

        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        residuals = [r[0] if r is not None else None for r in self._residuals]
        # HF returns attentions as a tuple of (B, H, T_q, T_k)
        attentions = [a[0].detach() for a in out.attentions]
        return {
            "residuals": residuals,
            "attentions": attentions,
            "logits": out.logits[0].detach(),
            "past_key_values": out.past_key_values,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def analyze(self, prompt: str, top_k_features: int = 32, top_k_logits: int = 5) -> CaptureResult:
        """One-shot analysis of `prompt` (no generation)."""
        if self._model is None:
            raise RuntimeError("CaptureEngine not attached; call attach() first")

        enc = self._tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        cap = self._forward_capture(input_ids)

        tokens = self._token_captures(input_ids[0], cap, top_k_features, top_k_logits)

        notes: list[str] = []
        if not self.sae_loaded:
            notes.append("SAE weights unavailable for this model — feature view falls back to raw neuron activations.")

        return CaptureResult(
            model_name=self.spec.name,
            prompt=prompt,
            tokens=tokens,
            n_layers=self.spec.n_layers,
            sae_loaded=self.sae_loaded,
            notes=notes,
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        top_k_features: int = 32,
        top_k_logits: int = 5,
        temperature: float = 0.8,
    ) -> Iterator[TokenCapture]:
        """Yield per-token TokenCapture as generation proceeds."""
        if self._model is None:
            raise RuntimeError("CaptureEngine not attached")

        enc = self._tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        cap = self._forward_capture(input_ids)
        # Emit captures for every prompt position so the UI immediately has the heatmap.
        for tc in self._token_captures(input_ids[0], cap, top_k_features, top_k_logits):
            yield tc

        past = cap["past_key_values"]
        last_logits = cap["logits"][-1]
        position = int(input_ids.shape[1])

        for _ in range(max_new_tokens):
            next_id = self._sample(last_logits, temperature)
            new_ids = next_id.view(1, 1)
            cap = self._forward_capture(new_ids, past_key_values=past)
            past = cap["past_key_values"]
            last_logits = cap["logits"][-1]

            # For a single-token step, residuals are length-1 along T.
            text = self._tokenizer.decode([int(next_id.item())])
            tc = self._single_position_capture(
                position=position,
                token_id=int(next_id.item()),
                text=text,
                cap=cap,
                top_k_features=top_k_features,
                top_k_logits=top_k_logits,
            )
            yield tc
            position += 1

    # ------------------------------------------------------------------
    # internals: token capture assembly
    # ------------------------------------------------------------------

    def _sample(self, logits: Any, temperature: float) -> Any:
        import torch

        if temperature <= 1e-6:
            return torch.argmax(logits)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _token_captures(
        self,
        token_ids: Any,
        cap: dict[str, Any],
        top_k_features: int,
        top_k_logits: int,
    ) -> list[TokenCapture]:
        from backend.features.logit_lens import logit_lens_topk
        from backend.features.sae import encode_topk

        out: list[TokenCapture] = []
        T = int(token_ids.shape[0])

        # Pre-compute SAE encodings per layer for all positions.
        # Bloom's gpt2-small-res-jb SAEs are trained on `hook_resid_pre`, i.e.
        # the residual stream ENTERING each block. `_residuals[layer_idx]` is
        # exactly that (our pre-hook captures the block's input). Earlier
        # versions used the post-block residual and got systematically wrong
        # activations.
        per_layer_features: list[Any] = []
        if self.sae_loaded:
            for layer_idx in range(self.spec.n_layers):
                resid = cap["residuals"][layer_idx]
                if resid is None:
                    per_layer_features.append(None)
                    continue
                per_layer_features.append(
                    encode_topk(self._sae_bundle, layer_idx, resid, k=top_k_features)
                )
        else:
            for layer_idx in range(self.spec.n_layers):
                resid = cap["residuals"][layer_idx]
                per_layer_features.append(
                    self._raw_neuron_topk(resid, k=top_k_features) if resid is not None else None
                )

        # Pre-compute logit lens for every layer × position.
        logit_topks = logit_lens_topk(
            self._model,
            cap["residuals"],
            k=top_k_logits,
            tokenizer=self._tokenizer,
        )

        for pos in range(T):
            tid = int(token_ids[pos].item())
            text = self._tokenizer.decode([tid])
            top_features = self._merge_top_features(per_layer_features, pos, top_k_features)
            logits_per_layer = [lt[pos] for lt in logit_topks]
            attn_top = self._attn_top_per_layer(cap["attentions"], pos)
            out.append(
                TokenCapture(
                    position=pos,
                    token_id=tid,
                    text=text,
                    top_features=top_features,
                    logits_per_layer=logits_per_layer,
                    attn_top_per_layer=attn_top,
                )
            )
        return out

    def _single_position_capture(
        self,
        position: int,
        token_id: int,
        text: str,
        cap: dict[str, Any],
        top_k_features: int,
        top_k_logits: int,
    ) -> TokenCapture:
        """Build a TokenCapture from a single-token forward step (T=1 over residuals)."""
        from backend.features.logit_lens import logit_lens_topk
        from backend.features.sae import encode_topk

        # Same residual-tap rule as in _token_captures: hook_resid_pre (input
        # to each block) — see comment there.
        per_layer_features: list[Any] = []
        if self.sae_loaded:
            for layer_idx in range(self.spec.n_layers):
                resid = cap["residuals"][layer_idx]
                if resid is None:
                    per_layer_features.append(None)
                    continue
                per_layer_features.append(
                    encode_topk(self._sae_bundle, layer_idx, resid, k=top_k_features)
                )
        else:
            for layer_idx in range(self.spec.n_layers):
                resid = cap["residuals"][layer_idx]
                per_layer_features.append(
                    self._raw_neuron_topk(resid, k=top_k_features) if resid is not None else None
                )

        logit_topks = logit_lens_topk(self._model, cap["residuals"], k=top_k_logits, tokenizer=self._tokenizer)

        top_features = self._merge_top_features(per_layer_features, 0, top_k_features)
        logits_per_layer = [lt[0] for lt in logit_topks]
        attn_top = self._attn_top_per_layer(cap["attentions"], 0)

        return TokenCapture(
            position=position,
            token_id=token_id,
            text=text,
            top_features=top_features,
            logits_per_layer=logits_per_layer,
            attn_top_per_layer=attn_top,
        )

    def _merge_top_features(
        self,
        per_layer: list[Any],
        pos: int,
        k: int,
    ) -> list[FeatureHit]:
        hits: list[FeatureHit] = []
        for layer_idx, table in enumerate(per_layer):
            if table is None:
                continue
            ids, acts = table  # both [T, k]
            for j in range(ids.shape[1]):
                fid = int(ids[pos, j].item())
                act = float(acts[pos, j].item())
                if act <= 0:
                    continue
                hits.append(FeatureHit(layer=layer_idx, feature_id=fid, activation=act))
        hits.sort(key=lambda h: -h.activation)
        return hits[:k]

    def _raw_neuron_topk(self, resid: Any, k: int) -> tuple[Any, Any]:
        """Fallback when SAE is unavailable: top-k absolute residual coords."""
        import torch

        abs_vals = resid.abs()
        topk = torch.topk(abs_vals, k=min(k, abs_vals.shape[-1]), dim=-1)
        return topk.indices, topk.values

    def _attn_top_per_layer(self, attentions: list[Any], pos: int) -> list[list[tuple[int, int, float]]]:
        """Per layer: top-3 (head, src_token, weight) attending to `pos`."""
        import torch

        out: list[list[tuple[int, int, float]]] = []
        for attn in attentions:
            # attn: [H, T_q, T_k]
            row = attn[:, pos, :]  # [H, T_k]
            flat = row.flatten()
            k = min(3, flat.numel())
            topk = torch.topk(flat, k=k)
            entries: list[tuple[int, int, float]] = []
            T_k = row.shape[1]
            for idx, val in zip(topk.indices.tolist(), topk.values.tolist(), strict=False):
                head = idx // T_k
                src = idx % T_k
                entries.append((int(head), int(src), float(val)))
            out.append(entries)
        return out
