# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Serving stack (P7). `Dockerfile` (multi-stage uv build, base and uv pinned by digest, non-root,
  healthcheck), a Streamlit UI (`necrohub ui`, `ui` extra) that calls the API over HTTP, the
  compose stack (db, loader, api, ui; profile `ollama` with ollama and ollama-pull), Makefile
  `up`, `down`, `ui` and `smoke`, a CI `compose-smoke` job with hadolint, and Dependabot for the
  docker and docker-compose ecosystems.
- Generative fine-tune (P6). `necrohub export-sft` (DVC stage `export_sft`, `data/sft`,
  `metrics/export_sft.json`) writes chat-format SFT JSONL and the extraction JSON schema;
  `necrohub eval-extraction` scores raw model output (valid JSON rate, per-field accuracy, set
  F1 for injuries and organ findings). `notebooks/qlora_extraction.ipynb` runs QLoRA on a free
  cloud T4, `deploy/ollama/Modelfile` serves the GGUF, and `docs/results/full-corpus.json` pins
  the 1500-case reference corpus. Decisions 22 and 23.
- Serving layer (P5). `necrohub index build` (DVC stage `index`, `make index`) chunks the redacted
  text of each section, embeds it (`embed` params: hashing, or a Hugging Face model) and writes
  `data/serving/index/{chunks.jsonl,embeddings.npy,index.json}` byte for byte reproducibly, with a
  content-derived `index_version`. It fails on row, dimension, NaN, norm or regex-PII problems and
  reports `gold_pii_strings_in_chunks` in `metrics/index.json`.
- `InMemoryVectorStore` and `PgVectorStore` (psycopg 3 and pgvector, the `db` extra) behind one
  `VectorStore` protocol; `necrohub index load` migrates, upserts and checks the stored count.
- RAG answerer (`necrohub query`, `make query`): extractive by default, an Ollama or Gemini
  backend with `NECROHUB_LLM`; a case id in the question restricts retrieval to that case.
- FastAPI app (`necrohub serve`, the `serve` extra): `GET /healthz` and `POST /ask`; it refuses to
  start when the configured embedder differs from the index's.
- `compose.yaml` with the pgvector `db` service (image pinned by digest), the `test-db` CI job, and
  `make demo` now ends with `necrohub query`.
- `necrohub train` (`make train`): fine-tunes an encoder for NER (BIO labels over tokenizer
  offsets, windows with stride for long sections, window merge by distance from the edge) or
  section classification with the Hugging Face `Trainer`. `train.base_model` is `tiny-scratch`
  (a WordPiece tokenizer and a small BERT trained on the train split, offline; the demo) or a Hub
  id or local directory. Seeded and deterministic on one machine; resumable checkpoints under
  `models/.checkpoints/<task>/<fingerprint>`, removed after a good run. Out: `models/encoder-<task>`
  with `necrohub_train.json`; metrics in `metrics/train_<task>.json`. New `train` params section;
  `accelerate` joins the `ml` extra.
- `necrohub evaluate` (`make evaluate`): strict and partial span precision, recall and F1 per label,
  micro, macro and micro_pii, PII character recall and missed PII spans, for the regex
  de-identifier and the encoder, and section accuracy and macro-F1, on the test split, split into
  random and held-out-template scopes (`metrics/eval.json`).
- `necrohub gpu-check` and the `cu126` extra (`make install-gpu`, `make gpu-check`): checks that
  the CUDA torch build has kernels for the installed GPU. `cu126`, not `cu128`: there are no
  torch 2.14.1 cu128 wheels (decision 15).
- `necrohub check-repro`: the CI demo gate. Data stages must reproduce `dvc.lock` entries and
  metrics bit for bit; the model stages (train_ner, train_sections, evaluate) must keep command,
  params, inputs, metrics structure and data-determined fields; drift in model-determined values
  is a warning (an error with `--strict-models`; decision 17).
- DVC stages `train_ner`, `train_sections` and `evaluate`; the CI demo job now installs the `ml`
  and `cpu` extras, runs offline and calls `check-repro`.
- `necrohub build-dataset` (`make build-dataset`): processed layer and raw gold to
  `data/datasets/{ner,sections,extraction,qa}/{train,validation,test}.jsonl` plus a generated
  dataset card (`README.md`). Splits are seeded and by case; held-out templates go to test only.
  The run fails on a case in two splits, a held-out case outside test, gold span projection
  rejects over `dataset.max_align_reject_fraction` (0.05), a gold PII value in an extraction
  input, target or QA pair, or an NER span that does not match its text. Extraction inputs are
  redacted with the projected gold PII spans plus the regex detections; a case whose PII did not
  fully project is left out of `extraction` (none at seed 7). The PII check ignores whitespace,
  so a value split by a line wrap is caught. Contracts in
  `contracts/datasets.py`; params in the new `dataset` section of `params.yaml`.
- DVC pipeline (`dvc.yaml`: generate, ingest, preprocess, build_dataset) with a committed
  `dvc.lock` and `metrics/<stage>.json`; `dvc` is the new `pipeline` extra, there is no remote.
  Each stage depends on every necrohub source it imports and on `uv.lock` (a test checks the
  imports against `dvc.yaml`).
  `make demo` runs it; `make install-ml` now also installs the `pipeline` extra.
- CI `demo` job: `dvc repro` on a clean runner, then fails unless `dvc status` is up to date and
  `dvc.lock` and `metrics/` match the committed files; uploads `metrics/`.
- `necrohub ingest` (`make ingest`): raw layer to `data/interim/ingested.jsonl` and
  `rejects.jsonl`, one `IngestedDocument` per case. A case with a PDF is read from its text
  extracted with pypdf (new core dependency), otherwise from its txt. The run fails when a gold
  record has no document, a doc_id repeats, a text holds U+FFFD or a PDF's text length drifts
  more than `ingest.max_pdf_length_drift` (0.15) from its txt.
- `necrohub preprocess` (`make preprocess`): interim to `data/processed/documents.jsonl` and
  `rejects.jsonl`, one `ProcessedDocument` per document, with the normalize steps and
  de-identifiers from `params.yaml` (`hf_ner` only when a model is set). The run fails on
  unordered or overlapping sections, section coverage under 95 % of non-whitespace characters,
  spans out of bounds or regex PII left in the redacted text. `metrics/preprocess.json` counts
  entities, section labels, coverage and the gold spans projected from the raw txt onto the
  processed text or rejected.
- `preprocess.align.project`: maps spans through PDF extraction and normalization with difflib
  matching blocks. A span whose whitespace changed inside (a PDF line wrap) is projected with the
  target slice as its text; a span whose visible characters changed is rejected. The preprocess
  and build-dataset metrics count exact, whitespace-drift and rejected projections.
- `check-identifiers` scans the extracted text of PDF files; a PDF that does not parse is listed
  as skipped.

- Python package scaffold: `uv` project and lock, ruff, mypy strict, pytest with markers, CI and
  security workflows, Dependabot.
- Data contracts for entity and section spans and for processed text.
- Text normalization chain (`fix_text`, NFC, newlines, non-breaking spaces, trailing whitespace,
  blank lines) that keeps accents, case and newlines. It repeats until the text stops changing,
  so normalizing twice gives the same result.
- Section splitting for necropsy reports, accent-insensitive and tolerant of numbered headers.
- Regex de-identification for CPF, CNPJ, RG, CRM, phone, e-mail, postal code, document numbers,
  dates, person names and addresses. It redacts the value and keeps its label. Labels take a `:`
  or a `nº`/`Nº` separator. CRM accepts one of the 27 UF codes and a thousands dot, with the
  keyword and UF in any case; a labelled CPF may drop the dots and keep the hyphen; phones accept
  a space between the halves. "Corpo de Delito" and "Corpo de Bombeiros" are not read as names.
- Redaction covers the union of overlapping spans, also across detectors, so a partial overlap
  never leaves the uncovered tail readable; the record's `entities` keep the longest span.
- `necrohub check-identifiers`: fails on checksum-valid CPF and CNPJ numbers and on RG, CRM, phone
  and e-mail values that are not in the committed fake pool. It reads UTF-8, UTF-16/32 with or
  without a BOM, cp1252 and latin-1 text, and lists the binary files it skipped. A bare digit
  run inside a longer ASCII token, such as a hex hash, is not read as a CPF or CNPJ unless it
  follows its keyword.
- Golden tests on three synthetic reports.
- Embeddings: `Embedder` protocol, `cosine_matrix`, token `windows` (rejects `stride >= size`),
  a deterministic `HashingEmbedder` (blake2b feature hashing) and `HfEmbedder` (windows over ids
  without specials, every window wrapped in cls/sep, masked mean over tokens then over windows,
  models loaded once).
- `HfNerDeidentifier` (token-classification pipeline built once, label map to `EntityLabel`,
  threshold; requires a fast tokenizer so long reports are windowed, never truncated) and the pure
  `merge_subwords`.
- `NECROHUB_DEVICE=auto|cpu|cuda` through `resolve_device`.
- CI job `test-ml` (tiny offline models), `make install-ml` and `make test-ml`.
- Dependabot cooldown of 7 days; zizmor audits the whole `.github` folder.
- Run metadata shared by stages: run manifests, atomic output directory swaps, tree hashes.
- `params.yaml` with typed `seed` and `generate` sections; unknown sections are ignored.
- Generator contracts: `CaseFacts`, `Draft`, `Realized`, `Reject` and `GoldRecord`, whose
  `verify` checks the text hash, every span and that the sections tile the text.
- Seeded case sampler over ten scenarios (firearm homicide, sharp force, traffic, fall, hanging,
  drowning, cardiac and other natural deaths, poisoning, undetermined). Names come from Faker
  `pt_BR` seeded per case, CPFs are checksum-invalid, and RG, CRM and phone values come from the
  fake pool, now 30 RG, 30 CRM and 20 phone values.
- `fill()`: turns a draft with `{{SLOT}}`, `[[LABEL|surface]]` and `<<SEC:LABEL>>` markers into
  text with exact entity and section spans, or a typed reject.
- Template realizer with three report families (`t1`, `t2`, `t3`) and phrase variants; every
  family's sections equal `split_sections` and its text is latin-1 encodable.
- `necrohub generate` and `make generate`: writes `reports/`, `gold/` and `rejects.jsonl`,
  `metrics/generate.json` and a run manifest; boundary checks (spans, required slots, leftover
  markers, unlabelled identifier-shaped values, reject budget) fail the run and keep the previous
  output.
- `faker` as a core dependency.
- `ChatLLM` protocol with `ChatMessage` and `LlmError`; `OllamaLLM` (`ollama` extra) and
  `GeminiLLM` (`genai` extra, bounded retries of transient API errors only, key from
  `GEMINI_API_KEY` and never shown). SDKs are imported lazily.
- `LlmRealizer`: a local model writes the report draft from non-identifying case facts and slot
  names; `fill()` inserts identifiers and labels spans as for templates. Drafts with PII written as
  literal text (anything the regex de-identifier finds outside a gold PII span) or non-latin-1
  text are rejected per case, typographic characters are transliterated and counted, rejected
  cases retry with a derived seed up to `generate.llm.max_attempts`. Backend failures, including
  timeouts (`generate.llm.timeout_s`, default 120 s), are rejects, not crashes.
- `generate.realizer: template | llm`, `generate.llm` (`backend`, `model`, `temperature`,
  `max_attempts`) and `generate.pdf_fraction` params; the Ollama model is pinned to
  `qwen3:4b-instruct-2507-q4_K_M`.
- Deterministic PDF rendering (fpdf2 core fonts, fixed creation date and producer) for a seeded
  share of the reports; `GoldRecord.files.pdf` points at it. `fpdf2` as a core dependency.

### Origin

Carries over, rewritten, the useful parts of three earlier prototypes: ragnosis, necropsy-hub and
necropsy-llm-hub. See `docs/REFACTOR_PLAN.md` for the migration map.
