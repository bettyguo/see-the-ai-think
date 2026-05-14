"""Thin HuggingFace loader. Lazy imports so the package imports without torch installed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.config import ModelSpec, get_model_spec

if TYPE_CHECKING:
    import torch  # noqa: F401


def load_model_and_tokenizer(spec_name: str, device: str = "cpu") -> tuple[Any, Any, ModelSpec]:
    """Load a registered model + tokenizer on the requested device.

    Returns (model, tokenizer, spec). Lazy-imports transformers / torch so the
    rest of the package (config, schemas, registry) is importable on a machine
    without torch — useful for tests of pure data structures.
    """
    spec = get_model_spec(spec_name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    return model, tokenizer, spec
