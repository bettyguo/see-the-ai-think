"""First-run cache warm: download the model, SAEs, and triggering corpus.

Idempotent — re-running is a fast no-op once the cache is populated.

Run via the Makefile (`make run` triggers `make warm`), or directly:
    python -m backend.warm
    python -m backend.warm --no-sae
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.config import CACHE_ROOT, DEFAULT_MODEL, get_model_spec

log = logging.getLogger("see-the-ai-think.warm")


def warm(spec_name: str = DEFAULT_MODEL, enable_sae: bool = True) -> int:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    spec = get_model_spec(spec_name)

    log.info("warming model: %s (%s)", spec.name, spec.hf_id)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoTokenizer.from_pretrained(spec.hf_id)
        AutoModelForCausalLM.from_pretrained(spec.hf_id)
        log.info("model %s ready", spec.hf_id)
    except Exception as e:
        log.warning("model download failed (%s); continuing — server will retry on startup", e)

    if enable_sae and spec.sae_release is not None:
        log.info("warming SAEs: %s", spec.sae_release)
        try:
            from sae_lens import SAE

            for layer_idx in spec.sae_layer_ids:
                sae_id = f"blocks.{layer_idx}.hook_resid_pre"
                try:
                    SAE.from_pretrained(release=spec.sae_release, sae_id=sae_id, device="cpu")
                    log.info("  layer %02d ok", layer_idx)
                except Exception as e:
                    log.warning("  layer %02d failed: %s", layer_idx, e)
        except ImportError:
            log.warning("sae-lens not installed; SAE features will be unavailable. "
                        "install with: pip install '.[sae]'")
    else:
        log.info("skipping SAE warm (disabled or no release configured)")

    log.info("warm complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="warm")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-sae", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    return warm(args.model, enable_sae=not args.no_sae)


if __name__ == "__main__":
    sys.exit(main())
