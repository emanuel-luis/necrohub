# necrohub

Pipeline for synthetic pt-BR forensic necropsy reports. Design: `docs/REFACTOR_PLAN.md`; decisions:
`docs/DECISIONS.md` (a decision there overrides the plan).

## Commands

- `uv sync --locked`: install (never sync a CUDA build; the torch wheels come from the `cpu` extra).
- `make install-ml`: `uv sync --locked --extra ml --extra cpu --extra ollama --extra genai --extra
  pipeline` (every extra; `pipeline` is dvc); needed before `make type` and `make test-ml`. Without the `ollama`/`genai` extras the client tests skip.
- `make lint`: `ruff check`, `ruff format --check`, `necrohub check-identifiers`.
- `make type`: strict mypy on `src` (after `make install-ml`).
- `make test-ml`: `ml` tests with tiny local models, `HF_HUB_OFFLINE=1`. `external` tests download real models and never run in CI.
- `make test`: offline unit tests. `uv run pytest --update-golden` rewrites `tests/golden`; read
  the diff before accepting it.
- `uv run necrohub check-identifiers [PATHS...]`: the identifier leak check.
- `make generate` / `uv run necrohub generate --params params.yaml --out data/raw`: synthetic
  reports and gold (`--runs-dir`, `--metrics-dir` override `runs/` and `metrics/`).
- `make ingest` / `uv run necrohub ingest --in data/raw --out data/interim`, then `make preprocess` /
  `uv run necrohub preprocess --in data/interim --out data/processed` (`--raw` for the gold
  alignment metric, default `raw` next to `--in`). Same `--params`, `--runs-dir`, `--metrics-dir`.
- `make build-dataset` / `uv run necrohub build-dataset --processed data/processed --raw data/raw
  --out data/datasets`: per-task splits and the dataset card.
- `make demo` = `uv run --locked --extra pipeline dvc repro` (all four stages; `dvc.yaml`). After a
  change that alters data or metrics, rerun it and commit `dvc.lock` and `metrics/`: CI's demo job
  fails when a clean Linux run does not reproduce them bit for bit. No DVC remote, no push/pull.
  A new import in a stage needs its source in that stage's `dvc.yaml` deps
  (`tests/unit/test_dvc_deps.py` fails otherwise).

## Repo map

- `src/necrohub/ids.py`: check digits, identifier patterns, fake pool loader, leak check (PDFs
  scanned through their pypdf text).
- `src/necrohub/data/fake_identifiers.toml`: the only identifier-shaped values allowed in the repo.
- `src/necrohub/contracts/`: Pydantic models for spans, documents, generator gold (`gold.py`) and
  dataset records (`datasets.py`).
- `src/necrohub/params.py` + `params.yaml`: typed params sections (`seed`, `generate` with `llm`,
  `ingest`, `preprocess`, `dataset`).
- `src/necrohub/settings.py`: env settings (`NECROHUB_*`, `OLLAMA_HOST`, `GEMINI_API_KEY`).
- `src/necrohub/llm/`: `base` (`ChatMessage`, `ChatLLM`, `LlmError`, `get_llm`), `ollama`, `gemini`
  (SDKs imported inside methods; never called in tests, which fake the SDK client).
- `src/necrohub/runs.py`: `stage_run`, `atomic_dir`, `sha256_tree`, JSONL io.
- `src/necrohub/generate/`: `vocab`, `sampler` (`sample_case(seed, i)`), `templates/*.toml`,
  `realizers` (`TextRealizer`, `TemplateRealizer`), `llm_realizer` (`LlmRealizer`, prompt in
  `prompts/llm_report.txt`, `template_id = llm:<backend>/<model>`), `fill`, `checks`,
  `render_pdf` (fpdf2 core fonts, fixed metadata), `stage` (`realize_case`: transliterate, fill,
  unlabeled-PII (regex de-identifier) and latin-1 rejects, retries). Template headers must be ones
  `split_sections` recognizes; template text stays latin-1. The txt is the span source; PDFs are
  renderings of it.
- `src/necrohub/ingest/`: `readers` (`txt_text`, `pdf_text` with pypdf, `extractor`), `stage`
  (`run_ingest`: one `IngestedDocument` per case, from the PDF when the case has one; decision 13).
- `src/necrohub/preprocess/`: `normalize`, `sections`, `pipeline`, `deid/{base,regex,hf_ner}`,
  `align` (`project`: difflib span projection, rejects instead of guessing), `stage`
  (`run_preprocess`: `ProcessedDocument`s, boundary checks, gold alignment metric).
  Any `DocReject` fails the ingest or preprocess run; a failed run keeps the previous out.
- `src/necrohub/datasets/`: `split` (`assign_splits`: seeded hash rank cut by fractions, held-out
  templates to test), `build` (pure task-record builders, `project_gold`, PII leak count), `card`
  (dataset card from the metrics), `stage` (`run_build_dataset`).
- `src/necrohub/embed/`: `base` (Embedder, cosine_matrix), `chunking`, `hashing`, `hf` (torch and transformers are imported inside functions and classes only).
- `tests/unit` (`test_idempotency.py`: every stage through build-dataset twice gives the
  same hashes), `tests/ml`, `tests/external`, `tests/fixtures/reports`, `tests/golden`, `tests/conftest.py` (`golden` fixture).

## Invariants

- Synthetic data only. No real person, document number, address or report, ever.
- Never write a real-looking identifier as a literal. Build it at runtime in the test (compute the
  check digits) and use `example.org` e-mail addresses.
- `data/` and `models/` never go into git.
- `necrohub check-identifiers` must pass before every commit and in CI.
- No network in the default tests and no model downloads. `ml`, `db`, `external` and `slow` tests
  are deselected by default.
- No work at import time: no file reads, model loads or environment reads outside function bodies
  and class definitions.
- No `__main__` blocks.

## Conventions

- Python 3.12, uv only (`uv run ...`), src layout, line length 100.
- Pydantic v2 models with `extra="forbid"`; top-level records carry `schema_version: Literal[1]`.
- Pure functions for text steps; detectors return spans and the pipeline merges and redacts once.
- Redaction keeps labels ("CPF nº") and replaces only the value with `[LABEL]`. ETNIA, SEXO and
  standalone UF are not redacted (decision 11).
- A broad `except` needs `# noqa: BLE001 - reason`.
- Code and docs in English. Commits follow the author's personal identity only.
