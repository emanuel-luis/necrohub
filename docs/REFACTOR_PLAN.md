# necrohub refactor plan

Target design and migration map for merging `ragnosis`, `necropsy-hub` and `necropsy-llm-hub` into
one reproducible pipeline for synthetic pt-BR forensic necropsy reports. Decisions are recorded in
`docs/DECISIONS.md`. Rows 6-7 were answered after the plan was written: the fine-tuning is staged
(P4 and P6 both run) and the generator LLM is a local Ollama model, with Gemini optional.

## 1. Decisions taken in this plan

| Topic | Choice | Why |
| --- | --- | --- |
| Package layout | One package `necrohub` (src layout), one subpackage per stage, heavy stacks as extras | The stages share contracts and ship as one unit with one lock. A uv workspace would add packaging overhead for no benefit, and the old workspace was broken (`libs/*` vs `services/*`) |
| LLM abstraction | One `ChatLLM` protocol used by the generator realizer, the LLM de-identifier and RAG | Three call sites share one client per backend (Gemini, Ollama) |
| Gold labels | Seeded sampler → `CaseFacts` → realizer draft with placeholders and markers → deterministic `fill()` → text + exact spans | The LLM never produces labels. Template and LLM realizers share the same fill and validation code |
| Span projection | `difflib` opcodes map gold spans through PDF extraction and normalization. A span whose endpoints fall in a changed region is rejected and counted | Generic, exact, and it measures data loss instead of hiding it |
| Fake identifiers | CPF/CNPJ are format-valid but checksum-invalid. RG, CRM and phone come from a committed fake pool. Emails use `example.org` | Generated values can never be a real person's document. The leak check reads the same pool, so there is one source of truth |
| Storage formats | JSONL for every record layer, `.npy` for vectors, `_manifest` outside DVC outs | Readable and diffable. Data outs stay byte-deterministic, so idempotency can be tested by hash |
| CI models | `tiny-scratch`: a 2-layer BERT config plus a WordPiece tokenizer trained offline on the train split. `HF_HUB_OFFLINE=1` in CI | CI never touches the network |
| Serving index | DVC stops at file artifacts (`data/serving/index`). `necrohub index load` upserts them into pgvector, keyed by `index_version` | The pipeline stays DB-free and reproducible. The DB is an idempotent sink, and rollback means switching versions |
| Similarity search | Exact cosine (`<=>`) filtered by `index_version`. Untyped `vector` column, no ANN index | Demo and full corpora are under 100k chunks, and embedder dims vary |
| Demo LLM | `ExtractiveAnswerer` (no LLM) when `NECROHUB_LLM=extractive`. Ollama is optional | `make demo` runs on CPU with no network |
| Removed from redaction | ETNIA, SEXO, standalone UF | They over-match (`substância branca`, `NÃO SE OBSERVA`) and they are medical descriptors, not direct identifiers. The data card records this |

## 2. Module tree

```
src/necrohub/
  __init__.py            # __version__ only; no work at import
  py.typed
  cli.py                 # Typer app; each command imports its stage lazily
  settings.py            # pydantic-settings, env only (secrets, hosts, device)
  params.py              # Pydantic models for each params.yaml section + loader
  runs.py                # RunManifest, stage_run() ctx manager, atomic_dir(), sha256_tree(), jsonl io
  ids.py                 # CPF/CNPJ check digits, fake pool loader, identifier regexes
  data/fake_identifiers.toml  # fake pool + allowlist (package data)
  contracts/  spans.py gold.py documents.py datasets.py serving.py
  llm/        base.py (ChatLLM) gemini.py ollama.py
  generate/   sampler.py vocab.py templates/*.txt realizers.py fill.py render_pdf.py stage.py
  ingest/     readers.py stage.py
  preprocess/ normalize.py nlp_optional.py sections.py align.py pipeline.py stage.py
              deid/ base.py regex.py hf_ner.py llm.py
  datasets/   build.py split.py card.py stage.py
  train/      tiny.py encoder.py stage.py
  evaluate/   metrics.py stage.py
  embed/      base.py chunking.py hashing.py hf.py
  index/      store.py memory.py pgvector.py stage.py
  serve/      rag.py prompts/*.txt api.py ui.py
notebooks/qlora_extraction.ipynb   # P6, only if decision 6 includes (b)
deploy/ollama/Modelfile            # P6/P7
tests/ unit/ ml/ db/ external/ fixtures/ golden/ conftest.py
```

## 3. CLI surface

| Command | Input → output | Phase |
| --- | --- | --- |
| `necrohub check-identifiers [--paths ...]` | tracked files (default `git ls-files`) → exit 0/1, masked findings | P1 |
| `necrohub generate --params params.yaml --out data/raw` | params → raw layer | P2 |
| `necrohub ingest --in data/raw --out data/interim` | txt/pdf → `ingested.jsonl` | P3 |
| `necrohub preprocess --in data/interim --out data/processed` | → `documents.jsonl` | P3 |
| `necrohub build-dataset --processed data/processed --gold data/raw/gold --out data/datasets` | → per-task splits | P3 |
| `necrohub train --task ner\|sections --data data/datasets --out models/encoder-<task>` | → HF checkpoint | P4 |
| `necrohub evaluate --models models --data data/datasets --out metrics/eval.json` | → metrics | P4 |
| `necrohub index build --in data/processed --out data/serving/index` / `index load` | → chunks + vectors / → pgvector | P5 |
| `necrohub query "<question>" [--store memory\|pgvector]` | → answer + citations | P5 |
| `necrohub serve` / `necrohub ui` | FastAPI / Streamlit | P5/P7 |
| `necrohub export-sft --tasks extraction,qa --out data/datasets/sft` | → chat-format JSONL | P6 |

Every stage command loads its params section into a typed model, runs inside `stage_run()` and
writes through `atomic_dir()`.

## 4. Data layers

| Layer | Path | Content | Owner |
| --- | --- | --- | --- |
| raw (landing) | `data/raw/reports/<case_id>.{txt,pdf}`, `data/raw/gold/<case_id>.json` | rendered reports + `GoldRecord` | generate |
| interim | `data/interim/ingested.jsonl`, `rejects.jsonl` | `IngestedDocument` | ingest |
| processed | `data/processed/documents.jsonl`, `rejects.jsonl` | `ProcessedDocument` | preprocess |
| datasets | `data/datasets/{ner,sections,extraction,qa}/{train,validation,test}.jsonl`, `README.md` (dataset card) | task records | build-dataset |
| models | `models/encoder-{ner,sections}/` | HF checkpoints + `training_args.json` | train |
| serving | `data/serving/index/{chunks.jsonl,embeddings.npy,index.json}`, then Postgres | redacted chunks + vectors | index |
| run metadata | `runs/<stage>/<run_id>.json` (gitignored), `metrics/<stage>.json` (git, DVC `cache: false`) | manifests, deterministic stats | all |

Idempotency rule: data outs are a pure function of (input content, params, code). IDs are derived
(`case_id = f"c{seed}-{i:05d}"`, `chunk_id = sha256(doc_id, section, ordinal, embedder_id)[:16]`).
Records carry no timestamps. Writes go to `<out>.tmp-<run_id>` and are swapped in with an atomic
rename. Rerunning gives byte-identical outs, and `tests/unit/test_idempotency.py` checks this with
`sha256_tree`.

## 5. Contracts (Pydantic v2, `extra="forbid"`, `schema_version: Literal[1]`)

- `Span`: `start:int≥0`, `end:int>start`, `label:EntityLabel`, `text:str`. `PredictedSpan(Span)`
  adds `score:float`, `source:Literal["regex","hf_ner","llm"]`.
- `EntityLabel` (StrEnum): PII = `PER CPF RG CRM CNPJ PHONE EMAIL ADDRESS DATE DOC_NUMBER LOC`.
  Clinical = `ORGAN INJURY FINDING CAUSE_OF_DEATH MANNER_OF_DEATH SUBSTANCE`. Property `is_pii`.
- `SectionLabel`: `PREAMBLE IDENTIFICATION HISTORY EXTERNAL_EXAM INTERNAL_EXAM COMPLEMENTARY_EXAMS
  CONCLUSION SIGNATURE`. `SectionSpan`: `label, start, end`.
- `CaseFacts`: `case_id`, `scenario` (firearm_homicide, sharp_force, traffic, fall, hanging,
  drowning, cardiac_natural, other_natural, poisoning, undetermined), `institute{name,city,uf}`,
  `report{number,exam_date,start_time,end_time,police_report_number}`,
  `examiner{name,crm,crm_uf}`, `assistant_name`,
  `decedent{name,sex,age,skin_color,birthplace,address,rg,cpf,mother_name}`,
  `circumstances{place_found,body_position,estimated_death_at}`,
  `external{height_cm,weight_kg,rigor,livor,decomposition,injuries:list[Injury{kind,region,side,size_cm,features}]}`,
  `internal:list[OrganFinding{organ,finding,abnormal}]`, `complementary_exams:list[str]`,
  `conclusion{cause_of_death,mechanism,manner_of_death,time_since_death_h:tuple[int,int]}`.
- `GoldRecord`: `case:CaseFacts`, `realizer`, `template_id`, `text_sha256`,
  `entities:list[Span]`, `sections:list[SectionSpan]`, `files{txt,pdf|None}`. Validator:
  `text[s.start:s.end] == s.text` for every span.
- `IngestedDocument`: `doc_id`, `source_path` (relative), `source_format`, `source_sha256`,
  `extractor` (name@version), `text`, `n_chars`, `warnings`.
- `ProcessedDocument`: `doc_id`, `text` (normalized, PII still present; the corpus is synthetic),
  `normalize_steps`, `sections`, `entities:list[PredictedSpan]`, `redacted_text`,
  `deidentifiers`.
- Datasets: `NerRecord{id,doc_id,section,text,spans}` with char spans (BIO is built at train time
  from offsets). `SectionRecord{id,doc_id,text,label}` with header-stripped paragraphs.
  `ExtractionRecord{id,doc_id,input_text (redacted),target:CaseExtraction}`, where
  `CaseExtraction` is the non-PII subset (cause, mechanism, manner, time window, injuries, organ
  findings). `QaRecord{id,doc_id,question,answer,section}` from question templates over gold.
- Serving: `Chunk{chunk_id,doc_id,section,ordinal,text,embedder_id}`,
  `RetrievedChunk{chunk,score}`, `Answer{question,answer,citations:list[Chunk],backend,index_version}`.
- `RunManifest`: `run_id` (uuid4 hex), `stage`, `started_at`/`finished_at` (UTC ISO-8601 `Z`),
  `status`, `code_version` (package version + `git rev-parse --short HEAD` when available),
  `params`, `params_sha256`, `inputs`/`outputs: list[{path,sha256,records}]`,
  `counts{read,written,rejected}`, `rejects_path`, `checks: list[{name,passed,detail}]`. A failed
  check sets `status="failed"`, exits 1 and leaves the previous out in place.

## 6. Quality checks per boundary

| Boundary | Checks (any failure fails the stage) |
| --- | --- |
| generate → raw | gold validates. Every span matches the text. Required slots present (name, CRM, cause, manner). No leftover `{{`/`[[`/`<<`. No identifier-shaped string outside a slot (LLM-invented IDs). Realizer rejects ≤ `max_reject_fraction` (template: 0) |
| raw → interim | one doc per report. `doc_id` unique. No U+FFFD. PDF char count within ±15 % of its txt twin |
| interim → processed | sections ordered, non-overlapping, covering ≥ 95 % of non-whitespace chars. Spans in bounds. Regex re-run on `redacted_text` finds 0 PII |
| processed → datasets | splits disjoint by `case_id` (and held-out `template_id`s go only to test). Alignment rejects ≤ `max_align_reject_fraction`. No PII label values inside extraction/QA targets. Label histogram written to the card |
| datasets → models | dataset sha256 recorded. Eval set never used in training |
| processed → serving | rows in `chunks.jsonl` equal rows in `embeddings.npy`. dim matches the embedder. No NaN. Norms within 1e-3 of 1. Chunk text passes the regex PII check |
| serving → Postgres | loaded count equals file count for the `index_version` |

## 7. Interfaces (signatures only)

```python
class ChatLLM(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str: ...


# GeminiLLM(api_key: SecretStr, model: str), OllamaLLM(host: str, model: str)


class TextRealizer(Protocol):
    name: str

    def realize(self, case: CaseFacts, seed: int) -> Draft: ...  # Draft{template_id, body}


# TemplateRealizer(templates_dir), LlmRealizer(llm: ChatLLM, prompt: str)
def fill(
    draft: Draft, case: CaseFacts
) -> Realized | Reject: ...  # Realized{text, entities, sections}


# draft syntax: {{SLOT}} -> CaseFacts value + PII span; [[LABEL|surface]] -> clinical span; <<SEC:LABEL>> line marker


class Deidentifier(Protocol):
    name: str

    def detect(self, text: str) -> list[PredictedSpan]: ...


# RegexDeidentifier(), HfNerDeidentifier(model: str, threshold: float, device: str), LlmDeidentifier(llm: ChatLLM)
def merge_spans(
    groups: Sequence[Sequence[PredictedSpan]], how: Literal["longest"] = "longest"
) -> list[PredictedSpan]: ...
def redact(
    text: str, spans: Sequence[Span]
) -> str: ...  # "[LABEL]", single pass, right to left, no overlaps


class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...  # (n, dim), L2-normalized


# HashingEmbedder(dim: int = 256)  # blake2b feature hashing of word unigrams+bigrams, signed
# HfEmbedder(model: str, layer_weighting: Literal["last", "linear_decay"], max_length: int, stride: int, device: str)
def windows(
    n_tokens: int, size: int, stride: int
) -> list[tuple[int, int]]: ...  # raises if stride >= size


class VectorStore(Protocol):
    def upsert(
        self, index_version: str, chunks: Sequence[Chunk], vectors: npt.NDArray[np.float32]
    ) -> int: ...
    def search(
        self, index_version: str, vector: npt.NDArray[np.float32], k: int
    ) -> list[RetrievedChunk]: ...
    def count(self, index_version: str) -> int: ...


# InMemoryVectorStore(), PgVectorStore(dsn: SecretStr)


def answer(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    index_version: str,
    llm: ChatLLM | None,
    k: int,
) -> Answer: ...  # llm=None -> extractive answer
```

## 8. Migration map

| Source | Target | Action | Notes |
| --- | --- | --- | --- |
| necropsy-hub `preprocessing/normalizer.py` (fix_text, NFC/NFKD, trim, remove_accents, lowercase) | `preprocess/normalize.py` | keep + rewrite | Default chain `fix_text, nfc, newlines, nbsp, trailing_ws, blank_lines` keeps accents, case and newlines |
| `normalizer.py` remove_stopwords, lemmatize, stem | `preprocess/nlp_optional.py` (`nlp` extra) | rewrite | load only, never download, clear error. Not in the default chain |
| `normalizer.py` remove_emojis, remove_html_tags, remove_newlines, remove_non_relevant_characters | none | drop | Reports have no HTML or emoji. Newline removal breaks sections. Drops `emoji`, `bs4`, `lxml` |
| `preprocessing/models.py` RegexPatternEntity | `preprocess/deid/regex.py` + `ids.py` | rewrite | context-anchored patterns with a `value` group (see defects) |
| `models.py` HF/spaCy model enums, `common/models.py` AbstractModelEnum | params strings + README table | drop | The enum overrides `Enum.name`. Drop `medgemma-4b-it` (not an embedding model) |
| `ner.py` retrieve_from_huggingface | `deid/hf_ner.py` HfNerDeidentifier | rewrite | load once in `__init__`, device from settings, subword fix-up extracted as pure `merge_subwords()`, label map PESSOA→PER, LOCAL→LOC. Also loads the P4 checkpoint |
| `ner.py` merge, replace | `deid/base.py` merge_spans, redact | rewrite | |
| `ner.py` retrieve_from_spacy | none | drop | spaCy NER is not needed. `nlp_optional` covers spaCy |
| `pipeline.py` obfuscate, process | `preprocess/pipeline.py` `process(text, deids) -> ProcessedText` | rewrite | |
| `pipeline.py` biomed_entities | `HfNerDeidentifier` with a `clinicalnerpt-*` model, `external` test only | rewrite | Gold clinical spans come from the generator |
| `tokenizer.py` | `embed/chunking.py` `tokens_with_offsets` | keep regex tokens, drop the rest | |
| `embedding/__init__.py` obtain | `embed/hf.py` HfEmbedder | rewrite | |
| `embedding/__init__.py` compare | `embed/base.py` `cosine_matrix(a, b)` (numpy) | rewrite | Drop PCA and `torch-pca` |
| `common/utils/tensor.py` chunk, pool | `embed/chunking.py` `windows` (pure Python) | rewrite | |
| `common/configs.py`, `logging.py`, `utils/{url,filesystem,nlp}.py` | `settings.py`, stdlib logging set up in `cli.py` | drop/rewrite | |
| ragnosis `scripts/generate.py` | `generate/sampler.py`, `realizers.py`, `stage.py` | rewrite | Keep the Faker pt_BR field ideas only |
| ragnosis `scripts/preprocess.py` | `deid/llm.py` | rewrite | The LLM returns `[{text,label}]` JSON, located in the text, never rewritten text |
| ragnosis `scripts/embedder.py` | `embed/chunking.py`, `index/stage.py` | rewrite | Keep section-then-window chunking and the sha256 dedupe idea |
| ragnosis `scripts/get_pdf.py` | `generate/render_pdf.py` (fpdf2) | rewrite | |
| ragnosis `app.py`, `utils/resources.py`, `utils/models.py` | `serve/ui.py`, `serve/rag.py`, `contracts/serving.py` | rewrite | |
| ragnosis `utils/genai.py`, `utils/config.py` | `llm/gemini.py`, `settings.py` | rewrite | AI Studio key (`GEMINI_API_KEY`), not Vertex |
| ragnosis `utils/storage.py`, `teste_*.py`, `read_metrics.py`, `tokenizer_counter.py`, `validate_ia.py`, `gemini_file_uploader.py`, Dockerfile, skaffold, clouddeploy, run-job-*.yaml | none | drop | |
| necropsy-llm-hub `services/llm/ollama.py` | `llm/ollama.py` OllamaLLM | rewrite | Keep `client.chat(format=schema)` |
| necropsy-llm-hub everything else | none | drop | Cassandra, Milvus, translation, English BioBERT, torchtext |
| `necropsy-hub/.../tests/laudo.txt` + `texto2`/`texto3` from `embedding/__init__.py:201-377` | `tests/fixtures/reports/{firearm_a,firearm_b,natural}.txt` | keep | All identifiers are on the allowlist |

### Known defects to fix during migration

| Where | Defect | Fix |
| --- | --- | --- |
| `common/utils/tensor.py:103-108` | `pool("max"/"min")` returns a `(values, indices)` tuple, so `.to("cpu")` fails | pool over the chunk axis, `.values` |
| `embedding/__init__.py:104` | `pool(dim=1)` on `(n_chunks, H)` collapses the hidden dim instead of the chunk axis | `dim=0` |
| `tensor.py:30-35` | `stride >= chunk_size` loops forever | validate in `windows()` |
| `embedding/__init__.py:47,151`; `ner.py:42` | device hardcoded to CUDA with a warning fallback | `NECROHUB_DEVICE=auto\|cpu\|cuda` |
| `embedding/__init__.py:52-53`; `ner.py:61-74`; `pipeline.py:24` | model and tokenizer reloaded on every call (`obfuscate` reloads bertimbau-large on every call) | load once per instance |
| `embedding/__init__.py:55-79` | window = `max_position_embeddings` (514 on RoBERTa-style models). Middle windows lack `[CLS]`/`[SEP]` | window over ids without specials using `tokenizer.model_max_length - 2` capped at 512, then wrap each window |
| `embedding/__init__.py:16-18` | overloads declare `max_length`, which the implementation lacks | single typed signature |
| `embedding/__init__.py:167-192` | dead code after `return`, `pooling` and `-diag` modes never applied | remove |
| `embedding/__init__.py:144-149` | PCA path `vstack`s tensors of different widths (raises), and the comparison is meaningless anyway | drop |
| `preprocessing/models.py:159-161` | CPF/CNPJ/RG prefix `(\bCPF\b.*)?` is greedy, so it redacts the label and any text up to the digits | match the value only (`value` group). RG needs `RG` context |
| `models.py:162,164` | TELEFONE and CEP match any 8 digits | phone needs a DDD or separator. CEP needs a hyphen or `CEP` context |
| `models.py:165` | CRM alternation: `\b` binds to one branch only, `.*CRM` is greedy | `\bCRM(?:[/-]?\s*[A-Z]{2})?\s*(?:n[ºo°]\.?\s*)?(?P<value>\d{4,6})\b` |
| `models.py:166-168` | UF hits `SE`/`TO`/`ES` in uppercase text. ETNIA hits `substância branca`. SEXO prefix can never match (space before value) | drop from redaction |
| `models.py:169` | DOCUMENTO matches any `nº 45` (street numbers) | `DOC_NUMBER` anchored on Laudo/B.O./boletim/inquérito |
| `pipeline.py:26-27` | regex substitutions run sequentially on already-mutated text and return no spans | detect on the original, merge, redact once |
| `pipeline.py:69-79` | `process()` strips newlines and accents before section/NER work, runs `fix_text` after `remove_accents`, and downloads spaCy lg/NLTK at runtime | new default chain, optional steps out |
| `normalizer.py:149` | `\(\)-\/` builds the range `)-/` (allows `* + ,`) | step dropped |
| `normalizer.py:96` | `remove_emojis` demojizes instead of removing | step dropped |
| `ner.py:154-188` | `merge` typed `*entity: Entity` but takes lists. `how` is a Literal but accepts a list. "nearest" joins across labels and mutates the inputs | typed and pure, same-label only |
| `ner.py:217-227` | `replace` with overlapping spans corrupts the text | merge before redact |
| `ner.py:145` | spaCy score `float("inf")` is not valid JSON | removed |
| `ner.py:261`, `tokenizer.py:97` | `__main__` blocks raise (`*models` positional; `bert_base_cased` does not exist) | no `__main__` blocks |
| `common/configs.py:12-13` | `PROJECT_ROOT/"libs"` does not exist, so the fixture path is broken | settings + pytest fixtures |
| ragnosis `generate.py:156-172` | tenacity `@retry` on a generator function, so errors during iteration are never retried | retry the API call |
| `generate.py:321-322` | thinking flags go into `**patient_details_kwargs` and are silently dropped | explicit params |
| `generate.py:193,197` | `fake.seed()` on an instance raises TypeError. `uuid4` ids are not reproducible | `seed_instance` per case, derived ids |
| `generate.py:39-40,169` | prompts read at import (the `prompts/` dir is missing, so import fails). API key default read at import | lazy reads, settings |
| `generate.py:329,340-343` | patient name used as a directory name. `metadata.csv` re-read and rewritten for every report | `case_id` paths, one JSONL write |
| Faker `cpf()` | emits checksum-valid CPFs (could belong to a real person) | checksum-invalid generator |
| `preprocess.py:103-107` | uploads every report to the Gemini Files API, never uses the upload, never deletes it | removed |
| `embedder.py:478` | `seen: set[str] = {}` is a dict, so the default path raises `AttributeError` | no mutable default |
| `embedder.py:124` | unpacks `fetchone()` without a None check | gone (pgvector) |
| `embedder.py:369,386` | `text.index(line)` gives wrong offsets for repeated lines | `finditer` offsets |
| `embedder.py:451,536` | pydantic private `_end_pos` cannot be set through the constructor | explicit `end` |
| `utils/genai.py:183,76,125,147` | client built at import (Vertex), mutable defaults, stream appends to the caller's list | lazy client, tuples |
| `utils/models.py:71`, `app.py:21` | hardcoded cloud bucket, chat session uploaded to cloud storage on every rerun | removed |
| llm-hub `ollama.py:48,71` | local `embedding` shadows the module (UnboundLocalError). BaseModel called with a positional arg (TypeError) | rewrite |

## 9. Infrastructure

### pyproject.toml

```toml
[project]
name = "necrohub"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
license = "MIT"
authors = [{ name = "Emanuel Luis", email = "emanuel.filho08@gmail.com" }]
dependencies = ["pydantic>=2.9", "pydantic-settings>=2.6", "typer>=0.15", "pyyaml>=6", "numpy>=2.1", "ftfy>=6.3"]
# P2 adds faker, fpdf2; P3 adds pypdf (small, pure Python, needed by the demo)

[project.optional-dependencies]
ml = ["transformers>=4.57", "datasets>=3", "accelerate>=1.1"]
cpu = ["torch>=2.8"]
cu128 = ["torch>=2.8"]
nlp = ["spacy>=3.8", "nltk>=3.9"]
genai = ["google-genai>=1.0", "tenacity>=9"]      # P2
ollama = ["ollama>=0.4"]                          # P5
db = ["psycopg[binary]>=3.2", "pgvector>=0.3"]    # P5
serve = ["fastapi>=0.115", "uvicorn>=0.32"]       # P5
ui = ["streamlit>=1.40", "httpx>=0.27"]           # P7
pipeline = ["dvc>=3.55"]                          # P3

[project.scripts]
necrohub = "necrohub.cli:app"

[dependency-groups]
dev = ["ruff>=0.8", "mypy>=1.13", "pytest>=8.3", "pytest-timeout>=2.3", "hypothesis>=6.115", "types-PyYAML"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu128" }]]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", extra = "cpu" }, { index = "pytorch-cu128", extra = "cu128" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
extend-select = ["BLE", "I"]

[tool.mypy]
strict = true
files = ["src"]
plugins = ["pydantic.mypy"]
# add [[tool.mypy.overrides]] ignore_missing_imports only for modules mypy reports

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-ra", "--strict-markers", "--strict-config", "-m", "not ml and not db and not external and not slow"]
timeout = 120
markers = [
  "ml: needs ml + cpu extras; offline tiny models only",
  "db: needs Postgres+pgvector at DATABASE_URL",
  "external: network, real model downloads or API keys; never in CI",
  "slow: over ~10 s",
]
```

Install commands: GPU machine `uv sync --locked --extra ml --extra cu128`. CI
`--extra ml --extra cpu`. P4 must check that `torch.cuda.get_arch_list()` includes `sm_75`
(GTX 1650) on cu128. If it does not, add a `cu126` extra the same way.

### dvc.yaml (P3-P5)

| Stage | deps | params | outs |
| --- | --- | --- | --- |
| generate | `src/necrohub/generate`, `src/necrohub/contracts` | `seed`, `generate` | `data/raw` |
| ingest | `data/raw/reports`, `src/necrohub/ingest` | `ingest` | `data/interim` |
| preprocess | `data/interim`, `src/necrohub/preprocess` | `preprocess` | `data/processed` |
| build_dataset | `data/processed`, `data/raw/gold`, `src/necrohub/datasets` | `seed`, `dataset` | `data/datasets` |
| train_ner / train_sections | `data/datasets/<task>`, `src/necrohub/train` | `seed`, `train` | `models/encoder-<task>` |
| evaluate | `models`, `data/datasets`, `data/processed` | `evaluate` | metrics `metrics/eval.json` (`cache: false`) |
| index | `data/processed`, `src/necrohub/embed`, `src/necrohub/index` | `embed`, `index` | `data/serving/index` |

Each stage also writes `metrics/<stage>.json` (counts, rejects, content sha256) with
`cache: false`. There is no DVC remote: the data regenerates from the seed. `dvc.lock` is
committed for lineage.

### params.yaml (demo-sized defaults; full runs use `dvc exp run -S ...`)

```yaml
seed: 7
generate: {n_cases: 24, realizer: template, pdf_fraction: 0.25, holdout_templates: [t3], max_reject_fraction: 0.2,
           llm: {backend: gemini, model: gemini-2.5-flash, temperature: 0.8}}
ingest: {formats: [txt, pdf]}
preprocess: {normalize: [fix_text, nfc, newlines, nbsp, trailing_ws, blank_lines], deidentifiers: [regex],
             hf_ner: {model: null, threshold: 0.7}}
dataset: {splits: {train: 0.8, validation: 0.1, test: 0.1}, tasks: [ner, sections, extraction, qa],
          max_align_reject_fraction: 0.05}
train: {base_model: tiny-scratch, tasks: [ner, sections], max_length: 256, stride: 64, epochs: 2,
        batch_size: 8, learning_rate: 5.0e-5, fp16: false}
evaluate: {baselines: [regex]}
embed: {backend: hashing, dim: 256, hf: {model: pucpr/biobertpt-clin, layer_weighting: last, max_length: 512, stride: 64}}
index: {chunk_tokens: 128, chunk_stride: 32}
```

Runtime settings come from env only and are documented in `.env.example`: `DATABASE_URL`,
`GEMINI_API_KEY`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `NECROHUB_LLM=extractive|ollama`,
`NECROHUB_INDEX_VERSION`, `NECROHUB_DEVICE`, `NECROHUB_LOG_LEVEL`, `HF_HUB_OFFLINE`.

### Docker compose (P7)

| Service | Image / command | Healthcheck | Depends on |
| --- | --- | --- | --- |
| db | `pgvector/pgvector:pg17` (pinned digest), volume `pgdata` | `pg_isready` | none |
| loader | app image, `necrohub index load` (one-shot), `./data/serving:ro` | none | db healthy |
| api | app image, `necrohub serve` :8000 | python urllib `GET /healthz` | db healthy, loader `service_completed_successfully` |
| ui | app image, `necrohub ui` :8501 | `GET /_stcore/health` | api healthy |
| ollama | `ollama/ollama` (pinned), profile `ollama`, volume `ollama` | `ollama list` | none |

Dockerfile: multi-stage uv build on a pinned `python:3.12-slim-bookworm` digest,
`ARG EXTRAS=serve,db,ui,ollama` (add `ml,cpu` for an HF-embedder index), non-root user,
`HEALTHCHECK`. The api refuses to start if the configured embedder id differs from `index.json`.

### CI (`ci.yml`, pinned SHAs, `permissions: contents: read`, `persist-credentials: false`, `timeout-minutes` on every job)

| Job | Needs | Runs |
| --- | --- | --- |
| lint | none | `uv sync --locked`, `ruff check`, `ruff format --check`, `necrohub check-identifiers` |
| type | lint | `uv sync --locked --extra ml --extra cpu --extra nlp` (+ extras as phases add them), `mypy` |
| test-unit | lint | core + dev, `pytest` |
| test-ml | lint | `--extra ml --extra cpu`, `HF_HUB_OFFLINE=1`, `pytest -m ml` |
| test-db (P5) | lint | service `pgvector/pgvector:pg17`, `pytest -m db` |
| demo (P3) | test-unit | `make demo`, then `uv run dvc status` must report up to date, upload `metrics/` |
| compose-smoke (P7) | demo | `docker compose up -d --wait`, `POST /ask`, `down -v` |

Also `security.yml` (pip-audit on `uv export`, zizmor) and weekly grouped Dependabot for actions
and uv.

## 10. Phases (one PR each; parallel where marked)

### P1: scaffold, migrated preprocessing and embedding, golden tests, leak check, CI

Files: `pyproject.toml` (section 9, P1 deps only), `uv.lock`, `.python-version` (3.12),
`.env.example`, `Makefile` (`help install install-gpu lint type test test-fast leak-check`;
`demo/up/down` arrive when they work), `README.md` (skeleton with the disclaimer "not for clinical
or forensic use", credits to the three origin repos), `CHANGELOG.md` (`[Unreleased]`),
`CLAUDE.md`, `.github/workflows/ci.yml` (lint, type, test-unit, test-ml),
`src/necrohub/{__init__,py.typed,cli,settings,ids}.py`, `src/necrohub/data/fake_identifiers.toml`,
`contracts/{spans,documents}.py`, `preprocess/{normalize,sections,pipeline}.py`,
`preprocess/deid/{base,regex,hf_ner}.py`, `embed/{base,chunking,hashing,hf}.py`.
Per DECISIONS rows 9-10, P1 skips `nlp_optional.py` and the `nlp` extra, and ships only the `cpu`
torch extra (`cu128` and the extras conflict arrive in P4). P1 runs as two implementer passes:
P1a (scaffold, contracts, normalize, sections, regex de-id, leak check, CI, docs skeleton) and
P1b (embeddings, HF NER, `ml` tests and CI job).

Fixtures and goldens:
- `tests/fixtures/reports/{firearm_a,firearm_b,natural}.txt`, migrated verbatim. Their identifiers
  (CPF 123.456.789-00 is checksum-invalid; RG 125623240, 1234567, 7654321; CRM 12345, 22456) go
  into `fake_identifiers.toml`.
- CRLF, nbsp and mojibake inputs are built inside the tests, because `.gitattributes` forces LF.
- Goldens live in `tests/golden/<suite>/<fixture>.{txt,json}`. `conftest.py` adds
  `--update-golden` and a `golden(name, actual)` fixture (about 20 lines, no plugin).
- Tiny models for `ml` tests are built in `tmp_path` from
  `BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)` and a vocab written by
  the test, with `torch.manual_seed(0)`.

Tests:
- `tests/unit/test_normalize.py`: `test_default_chain_golden[fixture]`,
  `test_preserves_newlines_and_accents`, `test_fix_text_repairs_mojibake`, `test_crlf_and_nbsp`,
  `test_idempotent` (hypothesis).
- `tests/unit/test_deid_regex.py`: `test_spans_golden[fixture]`, `test_redacted_golden[fixture]`,
  `test_cpf_redacts_value_keeps_label`, `test_ignores_substancia_branca`,
  `test_ignores_uppercase_se_to`, `test_phone_needs_ddd_or_separator`,
  `test_street_number_not_doc_number`, `test_redacted_has_no_matches`.
- `tests/unit/test_deid_merge.py`: `test_longest_wins`, `test_overlaps_redacted_once`,
  `test_merge_is_pure`, `test_merge_subwords`.
- `tests/unit/test_sections.py`: `test_sections_golden[fixture]`, `test_cover_without_overlap`,
  `test_numbered_and_unnumbered_headers`, `test_accent_insensitive_headers`.
- `tests/unit/test_chunking.py`: `test_windows_golden`, `test_windows_rejects_bad_stride`,
  `test_windows_cover_all` (hypothesis), `test_text_chunks_golden[fixture]`.
- `tests/unit/test_hashing_embedder.py`: `test_vectors_golden` (6 decimals),
  `test_deterministic_unit_norm`, `test_cosine_matrix`.
- `tests/unit/test_ids.py`: `test_cpf_cnpj_check_digits`, `test_fake_cpf_checksum_invalid`
  (hypothesis).
- `tests/unit/test_check_identifiers.py`: valid CPF built at runtime is flagged and masked in
  output; allowlisted RG/CRM pass; `example.org` and the author email pass; `uv.lock` and
  `dvc.lock` are skipped.
- `tests/unit/test_cli.py`: `--version` works, and `check-identifiers` exits 0/1.
- `tests/ml/test_hf_embedder.py`: shape and norm, pooling returns arrays (tuple regression), every
  window wrapped in specials, deterministic.
- `tests/ml/test_hf_ner.py`: spans stay in bounds with the tiny model.
- `tests/external/test_real_models.py`: biobertpt similarity(firearm_a, firearm_b) >
  similarity(firearm_a, natural).

Done when all of these exit 0:

```
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run necrohub check-identifiers
uv sync --locked --extra ml --extra cpu && HF_HUB_OFFLINE=1 uv run pytest -m ml
```

`git log` must show only the personal author, and CI must be green on the PR.

### P2: generator with gold labels (after P1)

Files: `contracts/gold.py`, `generate/*`, `llm/{base,gemini,ollama}.py`, `runs.py`. There are at
least 3 template families per section (pt-BR, latin-1 charset for fpdf2 core fonts).

Done when:
- `necrohub generate` runs twice and gives the same `sha256_tree(data/raw)`.
- All gold spans verify.
- `necrohub check-identifiers --paths data/raw` exits 0.
- The LLM realizer passes with a scripted `FakeLLM`, including reject cases (missing slot,
  invented CPF).
- Real Gemini and Ollama tests are marked `external`.

### P3: ingest, preprocess stage, dataset build, DVC, `make demo` (after P2)

Files: `ingest/*`, `preprocess/{align,stage}.py`, `datasets/*`, `dvc.yaml`, `params.yaml`,
`.dvc/`, Makefile `demo`.

Done when:
- `make demo` (= `uv run dvc repro`) runs green.
- A second `dvc repro` is a no-op.
- `test_idempotency.py` and `test_align.py` (PDF whitespace drift and normalization drift both
  project or reject correctly) pass.
- The dataset card is generated.

### P4: encoder fine-tune and evaluation (after P3; parallel with P5 and P6)

Files: `train/{tiny,encoder,stage}.py`, `evaluate/*`.

Metrics: span P/R/F1, strict and partial, per label, for regex and the fine-tuned encoder, on the
random split and the held-out template split; section macro-F1.

Done when the CI demo trains `tiny-scratch` end to end and a full local run is documented:

```
dvc exp run -S generate.n_cases=1500 -S train.base_model=pucpr/biobertpt-clin -S train.fp16=true
```

GTX 1650 settings: batch 8, max_length 256, gradient checkpointing. The model card draft holds the
real numbers. Local training holds the heavy lock.

### P5: index and RAG API (after P3; parallel with P4)

Files: `embed` integration, `index/*`, `serve/{rag,api}.py`, `necrohub query`.

Done when:
- `make demo` ends with `necrohub query "Qual a causa da morte no caso c7-00003?"` printing an
  answer with citations.
- `pytest -m db` passes against the service container, including the idempotent re-load and the
  count check.
- `/healthz` and `/ask` pass with the in-memory store.

### P6: generative fine-tune notebook (only if decision 6 includes (b); after P3)

Done when:
- `necrohub export-sft` exists.
- `notebooks/qlora_extraction.ipynb` installs the repo at a pinned commit, regenerates the dataset
  from the seed, and asserts its sha256 against `metrics/build_dataset.json`.
- QLoRA runs on a small open instruct model (candidate: Qwen3-1.7B; check the licence before use)
  on a free T4.
- Extraction field accuracy is reported against the base model.
- The model is exported to GGUF with a `deploy/ollama/Modelfile`.

Publishing weights to the Hugging Face Hub needs the user's OK.

### P7: serving, UI, compose (after P5)

Files: `serve/ui.py` (Streamlit over HTTP to the api), `Dockerfile`, `.dockerignore`,
`compose.yaml`, Makefile `up`/`down`, CI `compose-smoke`.

Done when `make demo && make up` serves the UI, and `docker compose --profile ollama up` switches
answering to Ollama.

### P8: docs and CI polish

Deliverables:
- README following the standards: "What this demonstrates", a Mermaid diagram of the layers,
  quickstart, data card, model card, disclaimer, credits.
- CHANGELOG 0.1.0.
- `security.yml` and Dependabot.
- Badges, but only real ones.

The public GitHub repo is created after P1 is verified and reviewed (program decision 8).

Parallelism: once P1 merges, `embed/` + `index/` (P5 library parts) can be built alongside P2
because the files are disjoint. P4, P5 and P6 can run in parallel after P3.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| Template-only corpus is repetitive, so metrics come out optimistic | Several families per section, test-only held-out templates, the hand-written fixtures as an out-of-distribution check, and an LLM realizer for diversity |
| An LLM realizer drops or edits placeholders | Validation rejects, a reject budget, and a temperature/prompt kept in params |
| PDF extraction and ftfy shift offsets | difflib projection with counted rejects and a threshold gate |
| fpdf2 core fonts are latin-1 only | The template charset stays latin-1. The renderer transliterates `– — “ ”` and counts the replacements |
| Faker or torch upgrades change dataset and model hashes | Pinned lock. Dependabot PRs show `metrics/` and `dvc.lock` diffs |
| cu128 wheels might lack `sm_75` | Check in P4, fall back to a `cu126` extra |
| `make` is not native on Windows | Document `uv run dvc repro` and WSL. CI is Linux |
| The hashing embedder retrieves poorly | Labelled as a mechanics-only default. The HF embedder is a param switch |
| Faker names can match real people | Unavoidable and harmless. Stated in the data card |

## 12. Open questions (resolved)

1. Fine-tuning target (DECISIONS row 6): staged (a)+(b). P4 and P6 both run; P6 stays a cloud
   notebook.
2. Generator LLM backend (DECISIONS row 7): Ollama first, with a pinned small model; Gemini is an
   optional backend, marked `external` in tests and never called in CI.
