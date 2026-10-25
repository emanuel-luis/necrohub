# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
