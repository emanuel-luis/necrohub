"""The preprocess stage: interim/ingested.jsonl -> processed/documents.jsonl plus metrics."""

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from necrohub.contracts.documents import DocReject, IngestedDocument, ProcessedDocument
from necrohub.contracts.gold import GoldRecord
from necrohub.params import PreprocessParams
from necrohub.preprocess.align import project, whitespace_drift
from necrohub.preprocess.deid.base import Deidentifier
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.preprocess.pipeline import process
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    atomic_dir,
    canonical_json,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_jsonl,
    write_text_atomic,
)

MIN_SECTION_COVERAGE = 0.95


def build_deidentifiers(p: PreprocessParams) -> list[Deidentifier]:
    deids: list[Deidentifier] = []
    for name in p.deidentifiers:
        if name == "regex":
            deids.append(RegexDeidentifier())
        else:
            from necrohub.preprocess.deid.hf_ner import HfNerDeidentifier

            assert p.hf_ner.model is not None  # PreprocessParams rejects hf_ner without a model
            deids.append(HfNerDeidentifier(p.hf_ner.model, threshold=p.hf_ner.threshold))
    return deids


def section_coverage(doc: ProcessedDocument) -> float:
    """Share of the text's non-whitespace characters that fall inside a section."""
    total = sum(1 for c in doc.text if not c.isspace())
    inside = sum(1 for s in doc.sections for c in doc.text[s.start : s.end] if not c.isspace())
    return inside / total if total else 1.0


def doc_problems(doc: ProcessedDocument) -> list[str]:
    """interim -> processed boundary problems of one document (reasons, no values)."""
    problems = []
    end = 0
    for s in doc.sections:
        if s.start < end:
            problems.append("sections_overlap_or_unordered")
        end = s.end
    if section_coverage(doc) < MIN_SECTION_COVERAGE:
        problems.append("section_coverage")
    if any(e.end > len(doc.text) or doc.text[e.start : e.end] != e.text for e in doc.entities):
        problems.append("span_out_of_bounds")
    if RegexDeidentifier().detect(doc.redacted_text):
        problems.append("pii_after_redaction")
    return problems


def align_gold(raw: Path, docs: list[tuple[IngestedDocument, ProcessedDocument]]) -> dict[str, Any]:
    """Project each document's gold spans from its raw txt onto the processed text and count.

    The stage does not keep the projected spans (build-dataset does); this measures the loss.
    """
    counts: Counter[str] = Counter()
    for ingested, doc in docs:
        gold_path = raw / "gold" / f"{doc.doc_id}.json"
        if not gold_path.is_file():
            counts["no_gold"] += 1
            continue
        gold = GoldRecord.model_validate_json(gold_path.read_text("utf-8"))
        source = (raw / gold.files.txt).read_text("utf-8")
        projected, rejected = project(source, doc.text, gold.entities)
        fmt = ingested.source_format
        drift = whitespace_drift(gold.entities, projected, rejected)
        counts["projected"] += len(projected)
        counts["exact"] += len(projected) - drift
        counts["whitespace_drift"] += drift
        counts[f"{fmt}_whitespace_drift"] += drift
        counts["rejected"] += len(rejected)
        counts[f"{fmt}_projected"] += len(projected)
        counts[f"{fmt}_rejected"] += len(rejected)
    total = counts["projected"] + counts["rejected"]
    return {
        **dict(sorted(counts.items())),
        "reject_fraction": round(counts["rejected"] / total, 6) if total else None,
    }


def run_preprocess(
    p: PreprocessParams,
    interim: Path,
    out: Path,
    *,
    raw: Path | None = None,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/documents.jsonl (sorted by doc_id) and <out>/rejects.jsonl.

    `raw` (default: a `raw` directory next to `interim`) is read only for the gold alignment
    metric; when it has no `gold` directory the metric is null. Any reject fails a boundary check:
    StageFailed, manifest failed, previous out kept.
    """
    raw = raw if raw is not None else interim.parent / "raw"
    with stage_run("preprocess", {"preprocess": p.model_dump()}, runs_dir) as run:
        source = interim / "ingested.jsonl"
        ingested = sorted(read_jsonl(source, IngestedDocument), key=lambda d: d.doc_id)
        run.inputs = [FileRef(path=source.as_posix(), sha256=sha256_tree(interim))]
        with atomic_dir(out, run.run_id) as tmp:
            deids = build_deidentifiers(p)
            pairs: list[tuple[IngestedDocument, ProcessedDocument]] = []
            rejects: list[DocReject] = []
            for ing in ingested:
                try:
                    pt = process(ing.text, deids, p.normalize)
                    doc = ProcessedDocument(doc_id=ing.doc_id, **pt.model_dump())
                except ValidationError as exc:
                    rejects.append(
                        DocReject(doc_id=ing.doc_id, reason="invalid", detail=str(exc)[:200])
                    )
                    continue
                if problems := doc_problems(doc):
                    rejects.append(DocReject(doc_id=ing.doc_id, reason=",".join(problems)))
                else:
                    pairs.append((ing, doc))
            docs = [d for _, d in pairs]
            write_jsonl(tmp / "documents.jsonl", docs)
            write_jsonl(tmp / "rejects.jsonl", rejects)

            reasons = Counter(x for r in rejects for x in r.reason.split(","))
            ids = [d.doc_id for d in docs]
            run.check("doc_id_unique", len(ids) == len(set(ids)), f"{len(ids)} ids")
            run.check("records_valid", not reasons["invalid"], f"{reasons['invalid']} invalid")
            for name in (
                "sections_overlap_or_unordered",
                "section_coverage",
                "span_out_of_bounds",
                "pii_after_redaction",
            ):
                run.check(f"no_{name}", not reasons[name], f"{reasons[name]} document(s)")
            tree = sha256_tree(tmp)
            run.counts = Counts(read=len(ingested), written=len(docs), rejected=len(rejects))
            run.rejects_path = (out / "rejects.jsonl").as_posix()
            run.raise_if_failed()
            run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=len(docs))]

    coverages = [section_coverage(d) for d in docs]
    metrics = {
        "read": len(ingested),
        "written": len(docs),
        "rejected": len(rejects),
        "processed_sha256": tree,
        "entities_by_label": dict(
            sorted(Counter(str(e.label) for d in docs for e in d.entities).items())
        ),
        "sections_by_label": dict(
            sorted(Counter(str(s.label) for d in docs for s in d.sections).items())
        ),
        "section_coverage_mean": round(sum(coverages) / len(coverages), 6) if coverages else None,
        "section_coverage_min": round(min(coverages), 6) if coverages else None,
        "gold_alignment": align_gold(raw, pairs) if (raw / "gold").is_dir() else None,
    }
    write_text_atomic(metrics_dir / "preprocess.json", canonical_json(metrics))
    return run
