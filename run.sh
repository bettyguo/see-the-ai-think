#!/usr/bin/env bash
# see-the-ai-think — one-command start for Linux/macOS.
# This is a thin wrapper around the Makefile so users without `make`
# (rare on Linux/macOS but possible on stripped-down systems) can still go.
set -euo pipefail
cd "$(dirname "$0")"

if command -v make >/dev/null 2>&1; then
  exec make run "$@"
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: python3 not found. install Python 3.11+ and re-run." >&2
  exit 1
fi

# Fallback path — manual replication of the Makefile target.
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip wheel
fi
.venv/bin/pip install -e ".[sae]"
.venv/bin/python -m backend.warm
( sleep 1 && .venv/bin/python -m webbrowser "http://127.0.0.1:8000" ) &
exec .venv/bin/python -m backend --host 127.0.0.1 --port 8000
