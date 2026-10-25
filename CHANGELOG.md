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
  blank lines) that keeps accents, case and newlines.
- Section splitting for necropsy reports, accent-insensitive and tolerant of numbered headers.
- Regex de-identification for CPF, CNPJ, RG, CRM, phone, e-mail, postal code, document numbers,
  dates, person names and addresses. It redacts the value and keeps its label.
- `necrohub check-identifiers`: fails on checksum-valid CPF and CNPJ numbers and on RG, CRM, phone
  and e-mail values that are not in the committed fake pool.
- Golden tests on three synthetic reports.
- Embeddings: `Embedder` protocol, `cosine_matrix`, token `windows` (rejects `stride >= size`),
  a deterministic `HashingEmbedder` (blake2b feature hashing) and `HfEmbedder` (windows over ids
  without specials, every window wrapped in cls/sep, masked mean over tokens then over windows,
  models loaded once).
- `HfNerDeidentifier` (token-classification pipeline built once, label map to `EntityLabel`,
  threshold) and the pure `merge_subwords`.
- `NECROHUB_DEVICE=auto|cpu|cuda` through `resolve_device`.
- CI job `test-ml` (tiny offline models), `make install-ml` and `make test-ml`.

### Origin

Carries over, rewritten, the useful parts of three earlier prototypes: ragnosis, necropsy-hub and
necropsy-llm-hub. See `docs/REFACTOR_PLAN.md` for the migration map.
