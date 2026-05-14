# see-the-ai-think — one-command start
#
# Usage:
#   make run         install deps + start server + open browser
#   make run-fast    same, but skip SAE download (degraded but works on slow networks)
#   make dev         install dev deps + run server with --reload
#   make test        run pytest (skips network-required integration tests)
#   make lint        ruff check
#   make clean       remove .venv, caches, __pycache__
#
# This Makefile is the canonical one-command entry point. run.sh and run.ps1
# call it (or replicate it) so users on any platform get the same behavior.

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: run run-fast dev test lint clean venv install warm open help

help:
	@echo "see-the-ai-think targets:"
	@echo "  make run         install + start + open browser"
	@echo "  make run-fast    same, but skip SAE download"
	@echo "  make dev         install dev deps + reload server"
	@echo "  make test        run pytest"
	@echo "  make lint        ruff check"
	@echo "  make clean       remove venv and caches"

$(VENV_PY):
	$(PY) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip wheel

venv: $(VENV_PY)

install: venv
	$(VENV_PIP) install -e ".[sae]"

install-fast: venv
	$(VENV_PIP) install -e .

warm: install
	$(VENV_PY) -m backend.warm

warm-fast: install-fast
	$(VENV_PY) -m backend.warm --no-sae

run: warm
	@echo "starting see-the-ai-think on http://$(HOST):$(PORT)"
	@( sleep 1 && $(VENV_PY) -m webbrowser "http://$(HOST):$(PORT)" ) &
	$(VENV_PY) -m backend --host $(HOST) --port $(PORT)

run-fast: warm-fast
	@echo "starting see-the-ai-think (no-SAE mode) on http://$(HOST):$(PORT)"
	@( sleep 1 && $(VENV_PY) -m webbrowser "http://$(HOST):$(PORT)" ) &
	$(VENV_PY) -m backend --host $(HOST) --port $(PORT) --no-sae

dev: venv
	$(VENV_PIP) install -e ".[sae,dev]"
	$(VENV_PY) -m backend --host $(HOST) --port $(PORT) --reload

test: venv
	$(VENV_PIP) install -e ".[dev]"
	$(VENV_PY) -m pytest

lint: venv
	$(VENV_PIP) install -e ".[dev]"
	$(VENV_PY) -m ruff check backend tests

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
