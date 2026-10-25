.PHONY: help install install-ml lint type test test-ml test-fast leak-check demo generate ingest preprocess build-dataset format clean

help:           ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:        ## sync the environment from uv.lock
	uv sync --locked

install-ml:     ## sync with every extra: ml, cpu (CPU torch wheels), ollama, genai, pipeline (dvc); needed by `make type` and `make test-ml`
	uv sync --locked --extra ml --extra cpu --extra ollama --extra genai --extra pipeline

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

demo:           ## the whole pipeline with DVC (generate -> build_dataset); installs the pipeline extra if missing
	uv run --locked --extra pipeline dvc repro

generate:       ## synthetic reports with gold labels into data/raw (params.yaml)
	uv run necrohub generate --params params.yaml --out data/raw

ingest:         ## raw reports (PDF text when a case has one) into data/interim
	uv run necrohub ingest --params params.yaml --in data/raw --out data/interim

preprocess:     ## normalize, split sections and de-identify data/interim into data/processed
	uv run necrohub preprocess --params params.yaml --in data/interim --out data/processed

build-dataset:  ## per-task train/validation/test datasets and the dataset card into data/datasets
	uv run necrohub build-dataset --params params.yaml --processed data/processed --raw data/raw --out data/datasets

format:         ## apply ruff fixes and formatting
	uv run ruff check --fix
	uv run ruff format

clean:          ## remove caches and build output
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .hypothesis dist
