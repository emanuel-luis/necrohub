# Pipeline details

The README gives the overview and the quickstart. This page holds the stage-level detail. Decisions are numbered in [DECISIONS.md](DECISIONS.md). The diagrams are in [diagrams/](diagrams/): D2 sources and the SVGs the README shows.

## Glossary

- Gold: the labels the generator writes together with each report. They are exact because the report text is built from them.
- Span: a character range in a text with a label, for example a CPF or a cause of death.
- BIO: a token labelling scheme (begin, inside, outside) that turns spans into one label per token, used for NER training.
- Held-out template: a report template family (`generate.holdout_templates`) that is used only in the test split, to measure how a model does on a layout it never saw in training.
- NER: named entity recognition, here for PII and case facts.

## Layers in detail

`ingest` reads a case from its PDF when it has one (pypdf text extraction; decision 13) and from its txt otherwise. `preprocess` normalizes, splits sections and de-identifies each document. A failed boundary check leaves the previous layer in place.

`metrics/preprocess.json` also projects the gold spans from each raw txt onto the processed text and counts exact projections, projections with whitespace drift and rejects. PDF extraction drops blank lines and re-wraps long lines. A span cut by a line break still projects (its text becomes the processed slice, line break included), and only a span whose visible characters changed is rejected. At seed 7 every gold span projects: 765 exactly, 8 across a PDF line wrap.

## LLM realizer (optional, local)

The template realizer is the default and the only path in CI. For more varied text, a local model writes the report body instead. It sees only non-identifying case facts and slot names and writes placeholders such as `{{DECEDENT_NAME}}` and `[[CAUSE_OF_DEATH|...]]`; the same `fill()` step then inserts the identifiers and labels the spans, so gold labels stay exact. Drafts that drop a required slot, use an unknown one, leave a marker broken, write a PII value as literal text (any CPF, RG, CRM, phone, e-mail, name, date, document number, address or CEP the regex de-identifier finds outside a slot) or use characters outside latin-1 are rejected per case into `rejects.jsonl` and retried with a new seed up to `generate.llm.max_attempts`. The run fails when rejects exceed `max_reject_fraction`. Typographic dashes, quotes and ellipses are transliterated first and counted in `metrics/generate.json`.

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen3:4b-instruct-2507-q4_K_M   # about 2.5 GB, fits a 4 GB GPU
uv sync --locked --extra ollama
# set generate.realizer: llm in params.yaml (OLLAMA_HOST overrides http://localhost:11434)
uv run necrohub generate --params params.yaml --out data/raw
```

Gemini is an optional, paid backend: `uv sync --locked --extra genai`, export `GEMINI_API_KEY` and set `generate.llm.backend: gemini` with a Gemini model name. LLM runs depend on the backend, so only template runs are byte-reproducible.

## PDFs

A seeded `pdf_fraction` of the reports (default 0.25) is also rendered to `reports/<case_id>.pdf` with fpdf2 core fonts and fixed metadata, so the bytes repeat across runs, and `GoldRecord.files.pdf` points at it. The txt stays the source of the spans. `check-identifiers` scans the text pypdf extracts from a PDF, and lists a PDF it cannot parse as skipped.

## Datasets

`build-dataset` writes `data/datasets/<task>/{train,validation,test}.jsonl` for four tasks:

- `ner`: one record per section, with gold character spans of every entity label.
- `sections`: section bodies without their header, labelled.
- `extraction`: redacted report text in, the non-PII case facts out.
- `qa`: fixed pt-BR questions answered from the gold case facts.

Splits are by case, seeded, 80/10/10 by default, and cases written with a held-out template go to test only. The run fails when splits share a case, a held-out case leaves test, more than `dataset.max_align_reject_fraction` (0.05) of the gold spans cannot be projected onto the processed text, a gold PII value shows up in an extraction input, target or QA pair, or an NER span does not match its record text. The sections of a document that lost a span to a projection reject are left out of `ner`. The extraction input is redacted with the projected gold PII spans plus the regex detections, not with the regex alone, whose recall gaps `metrics/eval.json` measures. A case with a gold PII span that did not project is left out of `extraction` (none at seed 7). The PII check compares with whitespace removed, so a value split by a line wrap is still caught. The counts and the content sha256 of the layer are in `metrics/build_dataset.json`.

The generated dataset card (`data/datasets/README.md`, not committed) lists provenance, the synthetic-only notice, split sizes, label histograms, the held-out template and the alignment reject rate. Known limits: ETNIA, SEXO and standalone UF are not redacted in the extraction inputs (decision 11), template text is repetitive so scores on it are optimistic, a span whose visible characters change in extraction is rejected and its section is dropped, and Faker names may match real people by chance.

## Serving

The index holds only redacted text. The default answerer is extractive and deterministic: it returns the sentence closest to the question and cites its chunk. `NECROHUB_LLM=ollama` or `gemini` answers from the retrieved context instead. A case id in the question limits the search to that case. `metrics/index.json` reports `gold_pii_strings_in_chunks`, the number of chunks holding a gold PII string of their own case. The regex de-identifier still misses some names, and this counts where that reaches the index (72 chunks of 194 in the committed demo run).

## Cloud QLoRA notebook (optional, free T4)

`necrohub export-sft` (DVC stage `export_sft`, `data/sft`) turns the `extraction` and `qa` records into chat-format JSONL (`{"messages": [...], "task", "id", "doc_id"}` per split, system prompts in pt-BR) plus `extraction.schema.json`, the JSON schema of the extraction target. It fails when a test case reaches train or validation, an assistant answer is not a valid `CaseExtraction`, or the regex de-identifier finds PII in a message.

`necrohub eval-extraction --data data/datasets --split test --predictions preds.jsonl --out results.json` scores raw model output (`{"id", "text"}` per line): valid JSON rate, exact accuracy per scalar field, set precision, recall and F1 for injuries and organ findings, and the rate of cases with every field right.

`notebooks/qlora_extraction.ipynb` (committed without outputs, never run in CI) does the following on a free T4:

1. Regenerates the 1500-case corpus from the seed and asserts its sha256 against `docs/results/full-corpus.json`.
2. Scores the base model, then QLoRA-tunes `Qwen/Qwen3-4B-Instruct-2507` (decisions 22 and 23) and scores it again with the same scorer.
3. Exports a Q4_K_M GGUF and writes `results/qlora_extraction.json` with scores, versions and wall times.

The author has not run it yet, so there are no fine-tuned numbers. After a run, that file is copied to `docs/results/`. `deploy/ollama/Modelfile` serves the GGUF with `ollama create necrohub-extract -f deploy/ollama/Modelfile`. The model is a demonstration, not for clinical or forensic use.

## Source tree

```
src/necrohub/
  cli.py                 Typer app (check-identifiers, generate, ingest, preprocess, build-dataset,
                         export-sft, eval-extraction, train, evaluate, gpu-check, check-repro,
                         index build/load, query, serve, ui)
  settings.py            environment settings (NECROHUB_*, DATABASE_URL, OLLAMA_HOST, GEMINI_API_KEY)
  params.py              typed sections of params.yaml
  runs.py                run manifests, atomic output swaps, tree hashes, JSONL io
  ids.py                 CPF/CNPJ check digits, identifier patterns, fake pool, leak check
  data/fake_identifiers.toml
  contracts/             Span, SectionSpan, ProcessedText, IngestedDocument, ProcessedDocument,
                         CaseFacts, GoldRecord, dataset records, serving records
  llm/                   ChatLLM protocol, Ollama and Gemini clients (optional extras)
  generate/              vocab, sampler, templates/*.toml, realizers, llm_realizer, prompts/,
                         fill, checks, render_pdf, stage
  ingest/                readers (txt, pdf with pypdf), stage
  preprocess/            normalize, sections, align, pipeline, stage, deid/{base,regex,hf_ner}
  datasets/              split (seeded, by case), build (task records), card, stage
  sft/                   chat-format export for the generative fine-tune
  embed/                 Embedder protocol, token windows, hashing and Hugging Face embedders
  index/                 index build and load, VectorStore protocol, in-memory and pgvector stores
  serve/                 RAG answerer, store runtime, FastAPI app, Streamlit UI, prompts/
  train/                 labels (BIO), tiny (from-scratch BERT), encoder (Trainer, predict), gpu, stage
  evaluate/              metrics (span P/R/F1, PII recall, sections), extraction, stage
  repro.py               check-repro
tests/                   unit (offline), ml (tiny models), external (downloads, never in CI),
                         fixtures/reports, golden (pytest --update-golden rewrites them)
```
