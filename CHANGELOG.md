# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

The first release carries over, rewritten, the useful parts of three earlier private prototypes.
`docs/REFACTOR_PLAN.md` has the migration map.

## [Unreleased]

## [0.1.0] - 2026-10-26

### Added

- Project scaffold: uv project and lock, ruff, mypy strict, pytest with markers, CI and security
  workflows, Dependabot (7-day cooldown, including docker and docker-compose), and zizmor over
  the whole `.github` folder.
- Typed contracts (Pydantic v2) for spans, sections, documents, generator gold and dataset
  records, and run metadata shared by all stages: manifests, atomic output swaps, tree hashes.
- Seeded synthetic generator: ten scenarios, three template families, exact gold spans, and a
  deterministic PDF rendering of a seeded share of the reports. An optional LLM realizer (local
  Ollama, or Gemini) goes through the same labelling step and rejects drafts that leak PII.
- De-identification and checks: regex de-identifier that redacts the value and keeps its label, an
  optional Hugging Face NER detector, and `necrohub check-identifiers`, which fails on
  checksum-valid CPF and CNPJ numbers and on RG, CRM, phone and e-mail values outside the
  committed fake pool, PDFs included.
- Text normalization and section splitting for necropsy reports, and embeddings (a deterministic
  hashing embedder and a Hugging Face one).
- Pipeline stages with boundary checks: `ingest` (PDF text through pypdf), `preprocess`
  (normalize, sections, de-identification, gold span projection with difflib), and
  `build-dataset` (per-task splits by case, held-out templates, dataset card).
- DVC pipeline (`dvc.yaml`, committed `dvc.lock` and `metrics/`) and `make demo`; a CI demo job
  runs it on a clean runner and calls `necrohub check-repro`, which requires data stages to
  reproduce bit for bit and model stages to keep inputs, structure and data-determined fields
  (decision 17).
- `necrohub train` (BIO NER and section classification with the Hugging Face `Trainer`, a tiny
  from-scratch model for the demo, resumable checkpoints), `necrohub evaluate` (strict and partial
  span F1, PII character recall, random and held-out-template scopes, regex baseline) and
  `necrohub gpu-check` with the `cu126` extra (decision 15).
- Model card and the metrics of a local fine-tune of `pucpr/biobertpt-clin` on the 1500-case
  corpus (`docs/MODEL_CARD.md`, `docs/results/encoder-biobertpt-clin/`). The weights are not
  published.
- Serving: `necrohub index build`, in-memory and pgvector stores, an extractive or LLM answerer
  with citations (`necrohub query`), a FastAPI app (`necrohub serve`) and a Streamlit UI
  (`necrohub ui`; a `?q=` link opens it with the question answered).
- Container stack: a `Dockerfile` (multi-stage uv build, base and uv pinned by digest, non-root,
  healthcheck), `compose.yaml` (db, loader, api, ui, and an `ollama` profile), Makefile `up`,
  `down`, `ui` and `smoke`, and CI jobs `test-db` and `compose-smoke` with hadolint.
- Generative fine-tune: `necrohub export-sft`, `necrohub eval-extraction`,
  `notebooks/qlora_extraction.ipynb` (cloud T4), `deploy/ollama/Modelfile` and
  `docs/results/full-corpus.json` for the 1500-case reference corpus (decisions 22 and 23).
- Golden, property (Hypothesis) and idempotency tests, offline `ml` tests with tiny models, and
  `NECROHUB_DEVICE=auto|cpu|cuda`.

[Unreleased]: https://github.com/emanuel-luis/necrohub/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/emanuel-luis/necrohub/releases/tag/v0.1.0
