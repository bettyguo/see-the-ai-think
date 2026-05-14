"""FastAPI application factory.

Layout:
  /            → frontend/index.html
  /static/...  → frontend assets
  /api/...     → all JSON + SSE endpoints (mounted from routes.make_router)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR, RuntimeConfig
from backend.server.routes import make_router

log = logging.getLogger("see-the-ai-think")


def create_app(config: RuntimeConfig | None = None, lazy: bool = False) -> FastAPI:
    """Build the FastAPI app.

    Args:
        config: runtime configuration. If None, defaults are used.
        lazy: if True, the model is NOT loaded at startup — useful for tests
              that exercise the routes without pulling torch weights.
    """
    cfg = config or RuntimeConfig()
    state: dict[str, Any] = {
        "model_name": cfg.model_name,
        "config": cfg,
        "engine": None,
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if lazy:
            log.info("lazy=True; skipping model load")
        else:
            log.info("loading model %s on %s", cfg.model_name, cfg.device)
            from backend.models.capture import CaptureEngine

            engine = CaptureEngine(
                spec_name=cfg.model_name,
                device=cfg.device,
                enable_sae=cfg.sae_enabled,
            )
            engine.attach()
            state["engine"] = engine
            log.info("model loaded. sae_loaded=%s", engine.sae_loaded)

        yield

        engine = state.get("engine")
        if engine is not None:
            engine.detach()

    app = FastAPI(
        title="see-the-ai-think",
        description="Watch an LLM think. Interactive interpretability for small open LLMs.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    api = make_router(state)
    app.include_router(api, prefix="/api")

    # Frontend mounting — directory may not exist in some test scenarios.
    if FRONTEND_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(FRONTEND_DIR)),
            name="static",
        )

        @app.get("/", response_class=HTMLResponse)
        async def index() -> FileResponse:
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


# Module-level app for `uvicorn backend.server.app:app`
app = create_app()
