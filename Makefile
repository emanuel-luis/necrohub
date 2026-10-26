.PHONY: help install install-ml install-gpu gpu-check lint type test test-ml test-fast leak-check demo generate ingest preprocess build-dataset train evaluate format clean

help:           ## list the targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:        ## sync the environment from uv.lock
	uv sync --locked

install-ml:     ## sync with every extra: ml, cpu (CPU torch wheels), ollama, genai, pipeline (dvc); needed by `make type` and `make test-ml`
	uv sync --locked --extra ml --extra cpu --extra ollama --extra genai --extra pipeline

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

demo:           ## the whole pipeline with DVC (generate -> evaluate, tiny-scratch on the CPU); needs `make install-ml`
	uv run --locked --extra pipeline dvc repro

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

format:         ## apply ruff fixes and formatting
	uv run ruff check --fix
	uv run ruff format

clean:          ## remove caches and build output
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache .hypothesis dist
