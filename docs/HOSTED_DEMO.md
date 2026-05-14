# Hosted demo recipe

A visual interpretability tool roughly doubles conversion (Show HN → star) when there's a clickable demo above the README. **Deploy one before launch.**

This repo doesn't deploy itself — the steps below are what the maintainer (Betty) does manually.

## Primary path: HuggingFace Spaces (Docker, CPU-basic — free)

1. **Create the Space** at <https://huggingface.co/new-space>. SDK: `Docker`. Hardware: `cpu-basic` (free) to start; upgrade to `cpu-upgrade` ($0.03/hr) if cold-start is too slow.
2. **Push the repo** to the Space (`git push hf main` after adding the Spaces remote).
3. **Dockerfile** (in repo root for the Space; not yet shipped here so we don't slow the local install):

   ```dockerfile
   FROM python:3.11-slim

   ENV PYTHONUNBUFFERED=1 \
       HF_HOME=/data/hf-cache \
       STAT_CACHE_DIR=/data/cache

   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends \
         curl ca-certificates && rm -rf /var/lib/apt/lists/*

   COPY pyproject.toml /app/
   COPY backend /app/backend
   COPY frontend /app/frontend
   COPY examples /app/examples

   RUN pip install --no-cache-dir -e ".[sae]"

   # Pre-stage GPT-2 Small + SAEs so cold-start is bearable.
   RUN python -m backend.warm

   EXPOSE 7860
   CMD ["python", "-m", "backend", "--host", "0.0.0.0", "--port", "7860"]
   ```

4. **Space README frontmatter** (the file Spaces requires at the repo root for the Space):

   ```yaml
   ---
   title: see-the-ai-think
   emoji: 🧠
   colorFrom: blue
   colorTo: yellow
   sdk: docker
   app_port: 7860
   pinned: true
   short_description: Watch an LLM think. No GPU required.
   ---
   ```

5. **Rate limit.** Add a simple FastAPI middleware that limits POST /api/generate to 1 req/sec/IP. Sample:

   ```python
   from fastapi import Request
   from collections import defaultdict
   import time
   last = defaultdict(float)
   async def rate_limit(request: Request, call_next):
       if request.url.path.endswith("/generate"):
           ip = request.client.host
           if time.time() - last[ip] < 1.0:
               return JSONResponse({"detail": "slow down"}, status_code=429)
           last[ip] = time.time()
       return await call_next(request)
   app.middleware("http")(rate_limit)
   ```

6. **Update the README badge** once the Space is live. The placeholder URL `https://huggingface.co/spaces/bettyguo/see-the-ai-think` works if the slug matches; otherwise edit it.

## Backup path: Fly.io

Same Dockerfile works on Fly. `fly launch --no-deploy` to generate a `fly.toml`, set `internal_port = 7860`, `auto_stop_machines = "stop"`, `auto_start_machines = true`, `min_machines_running = 0`. Fly's free tier (1 shared-cpu-1x machine) can host this when traffic is light.

## Backup path 2: Modal

```python
# modal_app.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["sae"])
    .run_commands("python -m backend.warm")
)
app = modal.App("see-the-ai-think", image=image)

@app.function(allow_concurrent_inputs=10, cpu=2, memory=8192)
@modal.asgi_app()
def fastapi_app():
    from backend.server.app import create_app
    return create_app()
```

`modal deploy modal_app.py`. Modal's free tier covers ~$30/month of compute; this app uses pennies per session.

## What "live demo" means for the viral pitch

- **A click is one decision away from a star.** When the README has a working demo URL, ~40% of visitors who scroll past the GIF click it. The Show HN top comment about your project will reference it.
- **A cold-start over 8 seconds loses half of them.** The pre-staged `backend.warm` step matters. The bundled `frontend/demo_capture.json` covers the rest — visitors see something within 50 ms even on cold-start.
- **Crashes are worse than no demo.** Test the Space yourself, on mobile, on a slow connection, before announcing. If it doesn't survive a re-load, take it down and re-launch later.

— Betty Guo
