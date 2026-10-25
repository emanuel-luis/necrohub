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

Phase 3 of 8, in progress: scaffold, text normalization, section splitting, regex and Hugging Face NER
de-identification, hashing and Hugging Face embeddings, the identifier leak check, a seeded
template generator with exact gold labels, an optional LLM realizer (local Ollama, optional
Gemini) behind the same labelling step, PDF rendering of part of the reports, ingestion (PDF
text extraction), the preprocess stage, the per-task dataset build and a DVC pipeline over all of
them (`make demo`) are in place. The CLI has five commands: `check-identifiers`, `generate`,
`ingest`, `preprocess` and `build-dataset`. Training, the retrieval API and the UI come in later
phases. The design and the phase plan are in
[docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md), the decisions in [docs/DECISIONS.md](docs/DECISIONS.md).

## What this demonstrates

In place now:

- Seeded synthetic generation that samples structured case facts first and writes the report
  second, so every name, document number, date, injury, abnormal organ finding and cause of death has an
  exact character span. Runs are byte-identical for the same seed, every run writes a manifest,
  and a failed boundary check keeps the previous output.
- Typed contracts: Pydantic v2 models with `extra="forbid"` for spans, sections and processed
  text; `Span.verify` checks a span against the text it came from.
- De-identification with a leak check in CI: context-anchored patterns that redact the value and
  keep the label ("CPF nº [CPF]"), an optional Hugging Face NER detector, and a check that fails on
  identifier-shaped values outside a committed fake pool.
- Reproducible, offline checks: a pinned `uv.lock`, golden tests on synthetic reports, property
  tests, and CI that runs tiny local models with no downloads.
- Layered data flow: raw, interim, processed and datasets layers, each written by one stage with
  a run manifest and boundary checks; rerunning the stages gives byte-identical layers.
- Gold spans carried across PDF extraction and normalization by difflib projection, with the
  spans it cannot map counted instead of guessed.
- One-command reproducibility: `make demo` runs the DVC pipeline, and CI fails when a clean run
  on Linux does not reproduce the committed `dvc.lock` and `metrics/` bit for bit.

Planned (see the phase plan):

- The serving layer.
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

Python 3.12, uv, Pydantic v2 and pydantic-settings, Typer, ftfy, NumPy, fpdf2, pypdf; Hugging Face
Transformers and PyTorch as the optional `ml` extra (CPU wheels in CI); DVC as the `pipeline` extra. Planned: Postgres with
pgvector, FastAPI, Streamlit, Docker Compose. Quality: ruff, mypy strict, pytest, Hypothesis, pip-audit and zizmor in CI.

## Quickstart

```bash
make demo   # = uv run --locked --extra pipeline dvc repro: generate -> ingest -> preprocess -> build_dataset
```

`make demo` writes `data/raw`, `data/interim`, `data/processed` and `data/datasets` (gitignored)
and `metrics/<stage>.json` (committed). A second run does nothing; `uv run dvc status` reports the
pipeline up to date. There is no DVC remote: the data regenerates from the seed in `params.yaml`.

```bash
uv sync --locked
make test                          # offline unit tests
make install-ml && make test-ml    # CPU torch + tiny-model tests, offline
uv run necrohub check-identifiers  # fail on identifiers outside the fake pool
uv run necrohub generate --params params.yaml --out data/raw   # or: make generate
uv run necrohub ingest --in data/raw --out data/interim        # or: make ingest
uv run necrohub preprocess --in data/interim --out data/processed   # or: make preprocess
uv run necrohub build-dataset --processed data/processed --raw data/raw --out data/datasets   # or: make build-dataset
uv run necrohub --version
```

### LLM realizer (optional, local)

The template realizer is the default and the only path in CI. For more varied text, a local
model writes the report body instead. It sees only non-identifying case facts and slot names and
writes placeholders such as `{{DECEDENT_NAME}}` and `[[CAUSE_OF_DEATH|...]]`; the same `fill()`
step then inserts the identifiers and labels the spans, so gold labels stay exact. Drafts that
drop a required slot, use an unknown one, leave a marker broken, write a PII value as literal text
(any CPF, RG, CRM, phone, e-mail, name, date, document number, address or CEP the regex
de-identifier finds outside a slot) or use characters outside latin-1 are rejected per case
into `rejects.jsonl`, retried with a new seed up to `generate.llm.max_attempts`, and the run fails
when rejects exceed `max_reject_fraction`. Typographic dashes, quotes and ellipses are
transliterated first and counted in `metrics/generate.json`.

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen3:4b-instruct-2507-q4_K_M   # about 2.5 GB, fits a 4 GB GPU
uv sync --locked --extra ollama
# set generate.realizer: llm in params.yaml (OLLAMA_HOST overrides http://localhost:11434)
uv run necrohub generate --params params.yaml --out data/raw
```

Gemini is an optional, paid backend: `uv sync --locked --extra genai`, export `GEMINI_API_KEY`
and set `generate.llm.backend: gemini` with a Gemini model name. LLM runs depend on the backend,
so only template runs are byte-reproducible.

### PDFs

A seeded `pdf_fraction` of the reports (default 0.25) is also rendered to
`reports/<case_id>.pdf` with fpdf2 core fonts and fixed metadata, so the bytes repeat across
runs, and `GoldRecord.files.pdf` points at it. The txt stays the source of the spans.
`check-identifiers` scans the text pypdf extracts from a PDF, and lists a PDF it cannot parse as
skipped.

### Layers

| Layer | Written by | Files |
| --- | --- | --- |
| raw | `generate` | `reports/<case_id>.txt` (and `.pdf` for some cases), `gold/<case_id>.json`, `rejects.jsonl` |
| interim | `ingest` | `ingested.jsonl` (one document per case), `rejects.jsonl` |
| processed | `preprocess` | `documents.jsonl` (normalized text, sections, PII spans, redacted text), `rejects.jsonl` |
| datasets | `build-dataset` | `{ner,sections,extraction,qa}/{train,validation,test}.jsonl`, `README.md` (dataset card) |

`ingest` reads a case from its PDF when it has one (pypdf text extraction; decision 13) and from
its txt otherwise. `preprocess` normalizes, splits sections and de-identifies each document. Each
stage writes a manifest to `runs/<stage>/` and metrics to `metrics/<stage>.json`, and a failed
boundary check leaves the previous layer in place. `metrics/preprocess.json` also projects the
gold spans from each raw txt onto the processed text and counts exact projections, projections
with whitespace drift and rejects. PDF extraction drops blank lines and re-wraps long lines; a
span cut by a line break still projects (its text becomes the processed slice, line break
included), and only a span whose visible characters changed is rejected. At seed 7 every gold
span projects: 765 exactly, 8 across a PDF line wrap.

### Datasets

`build-dataset` writes `data/datasets/<task>/{train,validation,test}.jsonl` for four tasks:
`ner` (one record per section with gold character spans of every entity label), `sections`
(section bodies without their header, labelled), `extraction` (redacted report text in, the
non-PII case facts out) and `qa` (fixed pt-BR questions answered from the gold case facts).
Splits are by case, seeded, 80/10/10 by default, and cases written with a held-out template
(`generate.holdout_templates`) go to test only. The run fails when splits share a case, a held-out
case leaves test, more than `dataset.max_align_reject_fraction` (0.05) of the gold spans cannot be
projected onto the processed text, a gold PII value shows up in an extraction input, target or QA
pair, or an NER span does not match its record text. The sections of a document that lost a span
to a projection reject are left out of `ner`. The extraction input is redacted with the projected
gold PII spans plus the regex detections, not with the regex alone, whose recall gaps P4 measures;
a case with a gold PII span that did not project is left out of `extraction` (none at seed 7).
The PII check compares with whitespace removed, so a value split by a line wrap is still caught.

The stage also writes the dataset card, `data/datasets/README.md` (generated, not committed):
provenance, the synthetic-only notice, split sizes, label histograms, the held-out template and
the alignment reject rate. Its known limits: ETNIA, SEXO and standalone UF are not redacted in the
extraction inputs (decision 11), template text is repetitive so scores on it are optimistic, a
span whose visible characters change in extraction is rejected and its section (and, for PII, its
case's extraction record) is dropped, and Faker names may match real people by chance. The counts and the content sha256 of the layer are
in `metrics/build_dataset.json`.

`make help` lists the other targets. On Windows without `make`, run the `uv run` commands from the
Makefile directly.

## Project structure

```
src/necrohub/
  cli.py                 Typer app (check-identifiers, generate, ingest, preprocess, build-dataset)
  settings.py            environment settings (NECROHUB_*, OLLAMA_HOST, GEMINI_API_KEY)
  params.py              typed sections of params.yaml
  runs.py                run manifests, atomic output swaps, tree hashes, JSONL io
  ids.py                 CPF/CNPJ check digits, identifier patterns, fake pool, leak check
  data/fake_identifiers.toml
  contracts/             Span, SectionSpan, ProcessedText, IngestedDocument, ProcessedDocument,
                         CaseFacts, GoldRecord, dataset records (NerRecord, SectionRecord,
                         ExtractionRecord, QaRecord)
  llm/                   ChatLLM protocol, Ollama and Gemini clients (optional extras)
  generate/              vocab, sampler, templates/*.toml, realizers, llm_realizer, prompts/,
                         fill, checks, render_pdf, stage
  ingest/                readers (txt, pdf with pypdf), stage
  preprocess/            normalize, sections, align, pipeline, stage, deid/{base,regex,hf_ner}
  datasets/              split (seeded, by case), build (task records), card, stage
  embed/                 Embedder protocol, token windows, hashing and Hugging Face embedders
tests/
  unit/                  offline tests
  ml/                    tiny local models (marker ml), offline
  external/              real model downloads (marker external, never in CI)
  fixtures/reports/      three synthetic reports
  golden/                expected outputs (pytest --update-golden rewrites them)
docs/                    refactor plan and decisions
params.yaml              pipeline parameters (seed, generate, ingest, preprocess, dataset)
dvc.yaml, dvc.lock       DVC pipeline (`make demo`) and its committed lock
metrics/                 per-stage metrics, committed (DVC metrics, `cache: false`)
```

## Origin

This repository merges the kept parts of three earlier prototypes, ragnosis, necropsy-hub and
necropsy-llm-hub, into one pipeline. Their history is not imported.

## License

MIT, see [LICENSE](LICENSE).
