"""The ingest stage: raw layer -> interim/ingested.jsonl (one IngestedDocument per case)."""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Literal

from necrohub.contracts.documents import DocReject, IngestedDocument
from necrohub.ingest.readers import extractor, pdf_text, txt_text
from necrohub.params import IngestParams
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    atomic_dir,
    canonical_json,
    sha256_tree,
    stage_run,
    write_jsonl,
    write_text_atomic,
)

_REPORT_SUFFIXES = (".txt", ".pdf")


def ingest_case(
    raw: Path, case_id: str, p: IngestParams
) -> tuple[IngestedDocument | DocReject, float | None]:
    """Read one case from its PDF when it has one (and pdf is in formats), else from its txt.

    Returns the document or a reject, and the PDF's length drift against its txt twin (or None).
    """
    reports = raw / "reports"
    formats: tuple[Literal["pdf", "txt"], ...] = ("pdf", "txt")
    candidates = [
        (fmt, reports / f"{case_id}.{fmt}")
        for fmt in formats
        if fmt in p.formats and (reports / f"{case_id}.{fmt}").is_file()
    ]
    if not candidates:
        return DocReject(doc_id=case_id, reason="missing_report"), None
    fmt, path = candidates[0]
    data = path.read_bytes()
    try:
        text = pdf_text(data) if fmt == "pdf" else txt_text(data)
    except Exception as exc:  # noqa: BLE001 - any parse failure makes the file unreadable
        return DocReject(doc_id=case_id, reason="unreadable", detail=type(exc).__name__), None
    if n := text.count("�"):
        return DocReject(doc_id=case_id, reason="replacement_char", detail=f"{n} U+FFFD"), None
    warnings: list[str] = []
    drift = None
    twin = reports / f"{case_id}.txt"
    if fmt == "pdf":
        if not twin.is_file():
            warnings.append("pdf without a txt twin: length not checked")
        else:
            n_twin = len(txt_text(twin.read_bytes()))
            drift = abs(len(text) - n_twin) / max(n_twin, 1)
            warnings.append(f"pdf text has {len(text)} chars, txt twin {n_twin}")
            if drift > p.max_pdf_length_drift:
                detail = f"drift {drift:.3f} > {p.max_pdf_length_drift}"
                return DocReject(doc_id=case_id, reason="pdf_length_drift", detail=detail), drift
    doc = IngestedDocument(
        doc_id=case_id,
        source_path=path.relative_to(raw).as_posix(),
        source_format=fmt,
        source_sha256=hashlib.sha256(data).hexdigest(),
        extractor=extractor(fmt),
        text=text,
        n_chars=len(text),
        warnings=warnings,
    )
    return doc, drift


def run_ingest(
    p: IngestParams,
    raw: Path,
    out: Path,
    *,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/ingested.jsonl (sorted by doc_id) and <out>/rejects.jsonl.

    Every case (a report file stem or a gold record) gives one document or one reject, and any
    reject fails a boundary check: StageFailed, manifest failed, previous out kept.
    """
    with stage_run("ingest", {"ingest": p.model_dump()}, runs_dir) as run:
        raw_tree = sha256_tree(raw)
        run.inputs = [FileRef(path=raw.as_posix(), sha256=raw_tree)]
        with atomic_dir(out, run.run_id) as tmp:
            stems = {f.stem for f in (raw / "reports").glob("*") if f.suffix in _REPORT_SUFFIXES}
            gold_ids = {f.stem for f in (raw / "gold").glob("*.json")}
            docs: list[IngestedDocument] = []
            rejects: list[DocReject] = []
            drifts: list[float] = []
            for case_id in sorted(stems | gold_ids):
                result, drift = ingest_case(raw, case_id, p)
                if isinstance(result, IngestedDocument):
                    docs.append(result)
                else:
                    rejects.append(result)
                if drift is not None:
                    drifts.append(drift)
            write_jsonl(tmp / "ingested.jsonl", docs)
            write_jsonl(tmp / "rejects.jsonl", rejects)

            ids = [d.doc_id for d in docs]
            reasons = Counter(r.reason for r in rejects)
            missing = sorted((stems | gold_ids) - set(ids))
            orphans = sorted(set(ids) - gold_ids)
            run.check(
                "one_doc_per_report",
                not missing and not orphans,
                f"{len(missing)} case(s) without a document (first: {missing[:1]}), "
                f"{len(orphans)} document(s) without gold (first: {orphans[:1]})",
            )
            run.check("doc_id_unique", len(ids) == len(set(ids)), f"{len(ids)} ids")
            run.check(
                "no_replacement_char",
                not reasons["replacement_char"],
                f"{reasons['replacement_char']} document(s) with U+FFFD",
            )
            run.check(
                "pdf_length_within_budget",
                not reasons["pdf_length_drift"],
                f"{reasons['pdf_length_drift']} PDF(s) over {p.max_pdf_length_drift}",
            )
            tree = sha256_tree(tmp)
            run.counts = Counts(
                read=len(stems | gold_ids), written=len(docs), rejected=len(rejects)
            )
            run.rejects_path = (out / "rejects.jsonl").as_posix()
            run.raise_if_failed()
            run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=len(docs))]

    metrics = {
        "read": run.counts.read,
        "written": len(docs),
        "rejected": len(rejects),
        "interim_sha256": tree,
        "by_format": dict(Counter(d.source_format for d in docs)),
        "reject_reasons": dict(reasons),
        "pdf_length_drift_max": round(max(drifts), 6) if drifts else None,
        "pdf_length_drift_mean": round(sum(drifts) / len(drifts), 6) if drifts else None,
    }
    write_text_atomic(metrics_dir / "ingest.json", canonical_json(metrics))
    return run
