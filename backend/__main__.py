"""Entry point: `python -m backend`."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="see-the-ai-think")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-sae", action="store_true", help="skip SAE encoding (degraded view)")
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="skip model load (server runs with /generate disabled — for smoke tests and the static demo)",
    )
    parser.add_argument("--reload", action="store_true", help="dev autoreload")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    import uvicorn

    from backend.config import RuntimeConfig
    from backend.server.app import create_app

    cfg = RuntimeConfig(
        model_name=args.model,
        device=args.device,
        sae_enabled=not args.no_sae,
    )

    # We can't pass `config` through uvicorn's import-string path with
    # --reload; in --reload mode the module-level app uses defaults.
    if args.reload:
        uvicorn.run(
            "backend.server.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        app = create_app(cfg, lazy=args.lazy)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
