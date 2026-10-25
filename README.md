# necrohub

A reproducible pipeline for synthetic Brazilian Portuguese forensic necropsy reports: generate
labelled reports, de-identify them, fine-tune encoders on the generated gold labels, and answer
questions over the corpus with retrieval.

> **Synthetic data only. Not for clinical, medico-legal or forensic use.**
> Every report, name and identifier in this repository is generated or invented. A CI check fails
> the build when a text file holds a CPF or CNPJ with a valid check digit, or an RG, CRM, phone or
> e-mail that is not on the fake allowlist. It is a pattern check, not a guarantee: it cannot see
> names, addresses or identifiers written in forms it does not know.

## Status

Phase 1 of 8: scaffold, text normalization, section splitting, regex and Hugging Face NER
de-identification, hashing and Hugging Face embeddings, and the identifier leak check are in
place. They are library functions with tests; the only CLI command so far is the leak check. The
generator, ingestion, the DVC pipeline and its data layers, training, the retrieval API and the UI
come in later phases. The design and the phase plan are in
[docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md), the decisions in [docs/DECISIONS.md](docs/DECISIONS.md).

## What this demonstrates

In place now:

- Typed contracts: Pydantic v2 models with `extra="forbid"` for spans, sections and processed
  text; `Span.verify` checks a span against the text it came from.
- De-identification with a leak check in CI: context-anchored patterns that redact the value and
  keep the label ("CPF nº [CPF]"), an optional Hugging Face NER detector, and a check that fails on
  identifier-shaped values outside a committed fake pool.
- Reproducible, offline checks: a pinned `uv.lock`, golden tests on synthetic reports, property
  tests, and CI that runs tiny local models with no downloads.

Planned (see the phase plan):

- Layered data flow: raw, interim, processed and serving layers, each written by one stage, with
  run metadata and hash-checkable outputs under DVC.
- Seeded synthetic generation that emits structured gold labels before the report text.
- Fine-tuning on those gold labels: exact character spans train and evaluate the NER and section
  encoders.

## Architecture (target)

```mermaid
flowchart LR
  generate --> raw[(raw)]
  raw --> ingest --> interim[(interim)]
  interim --> preprocess --> processed[(processed)]
  processed --> datasets[(datasets)]
  datasets --> train --> models[(models)]
  datasets --> evaluate
  models --> evaluate
  processed --> index --> pgvector[(pgvector)]
  pgvector --> api[RAG API] --> ui[UI]
```

## Tech stack

Python 3.12, uv, Pydantic v2 and pydantic-settings, Typer, ftfy, NumPy; Hugging Face
Transformers and PyTorch as the optional `ml` extra (CPU wheels in CI). Planned: DVC, Postgres with
pgvector, FastAPI, Streamlit, Docker Compose. Quality: ruff, mypy strict, pytest, Hypothesis, pip-audit and zizmor in CI.

## Quickstart

```bash
uv sync --locked
make test                          # offline unit tests
make install-ml && make test-ml    # CPU torch + tiny-model tests, offline
uv run necrohub check-identifiers  # fail on identifiers outside the fake pool
uv run necrohub --version
```

`make help` lists the other targets. On Windows without `make`, run the `uv run` commands from the
Makefile directly.

## Project structure

```
src/necrohub/
  cli.py                 Typer app (check-identifiers; stage commands come later)
  settings.py            environment settings (NECROHUB_*)
  ids.py                 CPF/CNPJ check digits, identifier patterns, fake pool, leak check
  data/fake_identifiers.toml
  contracts/             Span, SectionSpan, ProcessedText
  preprocess/            normalize, sections, pipeline, deid/{base,regex,hf_ner}
  embed/                 Embedder protocol, token windows, hashing and Hugging Face embedders
tests/
  unit/                  offline tests
  ml/                    tiny local models (marker ml), offline
  external/              real model downloads (marker external, never in CI)
  fixtures/reports/      three synthetic reports
  golden/                expected outputs (pytest --update-golden rewrites them)
docs/                    refactor plan and decisions
```

## Origin

This repository merges the kept parts of three earlier prototypes, ragnosis, necropsy-hub and
necropsy-llm-hub, into one pipeline. Their history is not imported.

## License

MIT, see [LICENSE](LICENSE).
