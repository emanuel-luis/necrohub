SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help install install-ml install-gpu gpu-check lint type test test-ml test-fast leak-check demo generate ingest preprocess build-dataset train evaluate index index-load query serve ui up down smoke format clean d2-install diagrams diagrams-check diagrams-watch

help:           ## list the targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:        ## sync the environment from uv.lock
	uv sync --locked

install-ml:     ## sync with every extra: ml, cpu (CPU torch wheels), ollama, genai, pipeline (dvc), db, serve, ui; needed by `make type` and `make test-ml`
	uv sync --locked --extra ml --extra cpu --extra ollama --extra genai --extra pipeline --extra db --extra serve --extra ui

install-gpu:    ## CUDA 12.6 torch + ml + pipeline extras: a multi-GB download, for local GPU training only
	uv sync --locked --extra ml --extra cu126 --extra pipeline

gpu-check:      ## report the torch CUDA build and fail unless it has kernels for the installed GPU
	uv run necrohub gpu-check

lint:           ## what CI's lint job runs
	uv run ruff check
	uv run ruff format --check
	uv run necrohub check-identifiers

type:           ## strict mypy on src (run `make install-ml` first: src imports torch, transformers and the LLM SDKs)
	uv run mypy

test:           ## offline unit tests (ml, db, external and slow are deselected)
	uv run pytest

test-ml:        ## tiny-model tests, offline (run `make install-ml` first)
	HF_HUB_OFFLINE=1 uv run pytest -m ml

test-fast:      ## same tests, quiet
	uv run pytest -q

leak-check:     ## fail on identifiers that are not in the fake pool
	uv run necrohub check-identifiers

demo:           ## the whole pipeline with DVC (generate -> index, tiny-scratch on the CPU), then a question; needs `make install-ml`
	uv run --locked --extra pipeline dvc repro
	uv run necrohub query "Qual a causa da morte no caso c7-00003?"

generate:       ## synthetic reports with gold labels into data/raw (params.yaml)
	uv run necrohub generate --params params.yaml --out data/raw

ingest:         ## raw reports (PDF text when a case has one) into data/interim
	uv run necrohub ingest --params params.yaml --in data/raw --out data/interim

preprocess:     ## normalize, split sections and de-identify data/interim into data/processed
	uv run necrohub preprocess --params params.yaml --in data/interim --out data/processed

build-dataset:  ## per-task train/validation/test datasets and the dataset card into data/datasets
	uv run necrohub build-dataset --params params.yaml --processed data/processed --raw data/raw --out data/datasets

train:          ## fine-tune the encoders (train.base_model) on data/datasets into models/encoder-<task>
	uv run necrohub train --task ner --params params.yaml --data data/datasets --out models/encoder-ner
	uv run necrohub train --task sections --params params.yaml --data data/datasets --out models/encoder-sections

evaluate:       ## score the encoders and the regex baseline on the test split into metrics/eval.json
	uv run necrohub evaluate --params params.yaml --models models --data data/datasets --raw data/raw --out metrics/eval.json

index:          ## chunk and embed data/processed into data/serving/index (params.yaml: embed, index)
	uv run necrohub index build --params params.yaml --in data/processed --gold data/raw/gold --out data/serving/index

index-load:     ## upsert data/serving/index into pgvector (needs DATABASE_URL and the db extra)
	uv run necrohub index load --in data/serving/index

query:          ## ask a question: make query Q="Qual a causa da morte no caso c7-00003?"
	uv run necrohub query "$(Q)"

serve:          ## the API on 127.0.0.1:8000 (needs the serve extra)
	uv run necrohub serve

ui:             ## the Streamlit UI on 127.0.0.1:8501 against NECROHUB_API_URL (needs the ui extra, no docker)
	uv run necrohub ui

up:             ## build and start db, loader, api and ui (docker compose; run `make demo` first)
	@test -f data/serving/index/index.json || { echo "data/serving/index is missing: run 'make demo' first" >&2; exit 1; }
	@test -f .env || { echo ".env is missing: copy .env.example to .env" >&2; exit 1; }
	docker compose up -d --build --wait db loader api ui

down:           ## stop the stack and delete its volumes
	docker compose down -v

smoke:          ## the CI compose checks against the running stack (`make up` first; needs curl and jq)
	bash scripts/compose_smoke.sh

format:         ## apply ruff fixes and formatting
	uv run ruff check --fix
	uv run ruff format

clean:          ## remove caches and build output
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .hypothesis dist

# Diagrams: docs/diagrams/<name>.d2 rendered to <name>.svg (committed). One pinned D2 everywhere.
# Layout engine, themes and padding live in docs/diagrams/_style.d2 (d2-config), so d2 runs in a
# clean environment: D2_* and SCALE variables would override that config. --scale 1 is not a
# d2-config key; it writes the natural width and height, so GitHub never upscales a small diagram.
D2_VERSION := 0.9.0
# sha256 of d2-v$(D2_VERSION)-linux-amd64.tar.gz, from the release's SHA256SUMS
D2_SHA256 := 5669ddc46b99e942cc96078f4a4e36d5e62103348f4c05179ede27802fdd87a9
D2_HOME ?= $(HOME)/.local/share/d2/v$(D2_VERSION)
D2 ?= $(D2_HOME)/bin/d2
D2_RUN = env -i PATH="$$PATH" HOME="$$HOME" "$(D2)" --scale 1
DIAGRAMS := $(filter-out docs/diagrams/_%,$(wildcard docs/diagrams/*.d2))

d2-install:     ## install the pinned D2 (Linux x86_64) into D2_HOME, checksum verified
	@if [ -x "$(D2_HOME)/bin/d2" ]; then echo "D2 v$(D2_VERSION) already in $(D2_HOME)"; exit 0; fi; \
	[ "$$(uname -sm)" = "Linux x86_64" ] || { echo "d2-install covers Linux x86_64 only;" \
		"install D2 v$(D2_VERSION) yourself and pass D2=/path/to/d2" >&2; exit 1; }; \
	tmp="$$(mktemp -d)"; trap 'rm -rf "$$tmp"' EXIT; \
	curl -fsSLo "$$tmp/d2.tar.gz" \
		"https://github.com/terrastruct/d2/releases/download/v$(D2_VERSION)/d2-v$(D2_VERSION)-linux-amd64.tar.gz"; \
	echo "$(D2_SHA256)  $$tmp/d2.tar.gz" | sha256sum --check --strict; \
	tar -xzf "$$tmp/d2.tar.gz" -C "$$tmp"; \
	mkdir -p "$(D2_HOME)/bin"; \
	install -m 0755 "$$tmp/d2-v$(D2_VERSION)/bin/d2" "$(D2_HOME)/bin/d2"; \
	"$(D2_HOME)/bin/d2" --version

diagrams:       ## render docs/diagrams/*.d2 to SVG with the pinned D2 (run make d2-install once)
	@v="$$($(D2_RUN) --version 2>/dev/null || true)"; [ "$$v" = "v$(D2_VERSION)" ] || { \
		echo "need D2 v$(D2_VERSION) at $(D2), found '$${v:-nothing}': run make d2-install" >&2; exit 1; }
	@for f in $(DIAGRAMS); do $(D2_RUN) "$$f" "$${f%.d2}.svg"; done

diagrams-check: diagrams ## render, then fail if a committed SVG is stale, missing or has no source
	@shopt -s nullglob; for s in docs/diagrams/*.svg; do \
		[ -f "$${s%.svg}.d2" ] || { echo "$$s has no .d2 source" >&2; exit 1; }; done
	@changed="$$(git status --porcelain --untracked-files=all -- 'docs/diagrams/*.svg')"; \
	if [ -n "$$changed" ]; then echo "$$changed" >&2; \
		echo "rendered SVGs differ from the committed ones: run make diagrams and commit" >&2; exit 1; fi

diagrams-watch: ## live preview in the browser while editing: make diagrams-watch NAME=architecture
	@# NAME only from the command line: WSL exports NAME (the hostname) into every shell.
	@[ "$(origin NAME)" = "command line" ] && [ -f "docs/diagrams/$(NAME).d2" ] || { \
		echo "usage: make diagrams-watch NAME=<one of: $(basename $(notdir $(DIAGRAMS)))>" >&2; exit 1; }
	$(D2_RUN) --watch --browser "$${BROWSER:-0}" docs/diagrams/$(NAME).d2 docs/diagrams/$(NAME).svg
