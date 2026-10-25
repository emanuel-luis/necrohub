"""The build-dataset stage: processed layer + raw gold -> datasets/<task>/<split>.jsonl."""

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from necrohub.contracts.datasets import NerRecord, SectionRecord
from necrohub.contracts.documents import ProcessedDocument
from necrohub.contracts.gold import GoldRecord
from necrohub.datasets.build import (
    extraction_record,
    ner_records,
    pii_leaks,
    pii_values,
    project_gold,
    qa_records,
    section_records,
)
from necrohub.datasets.card import render_card
from necrohub.datasets.split import SPLITS, Split, assign_splits
from necrohub.params import Params
from necrohub.preprocess.align import whitespace_drift
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


def _verifies(r: NerRecord) -> bool:
    try:
        for s in r.spans:
            s.verify(r.text)
    except ValueError:
        return False
    return True


def run_build_dataset(
    params: Params,
    processed: Path,
    raw: Path,
    out: Path,
    *,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/<task>/{train,validation,test}.jsonl for each task and <out>/README.md.

    `raw` gives the gold records and the txt each gold span refers to. A failed boundary check
    raises StageFailed, marks the manifest failed and keeps the previous out.
    """
    d = params.dataset
    holdout = params.generate.holdout_templates
    stage_params = {"seed": params.seed, "holdout_templates": holdout, "dataset": d.model_dump()}
    with stage_run("build_dataset", stage_params, runs_dir) as run:
        source = processed / "documents.jsonl"
        docs = sorted(read_jsonl(source, ProcessedDocument), key=lambda x: x.doc_id)
        run.inputs = [
            FileRef(path=processed.as_posix(), sha256=sha256_tree(processed)),
            FileRef(path=(raw / "gold").as_posix(), sha256=sha256_tree(raw / "gold")),
        ]
        with atomic_dir(out, run.run_id) as tmp:
            pairs: list[tuple[ProcessedDocument, GoldRecord]] = []
            for doc in docs:
                path = raw / "gold" / f"{doc.doc_id}.json"
                if path.is_file():
                    pairs.append((doc, GoldRecord.model_validate_json(path.read_text("utf-8"))))
            split_of = assign_splits(
                {g.case.case_id: g.template_id for _, g in pairs}, d.splits, holdout, params.seed
            )
            records: dict[str, dict[Split, list[BaseModel]]] = {
                t: {s: [] for s in SPLITS} for t in d.tasks
            }
            ner: list[NerRecord] = []
            sections: list[SectionRecord] = []
            n_projected = n_drift = n_rejected = n_dropped = n_ext_dropped = 0
            leaks = input_leaks = 0
            for doc, gold in pairs:
                spans, rejected, lost = project_gold(
                    gold, (raw / gold.files.txt).read_text("utf-8"), doc
                )
                n_projected += len(spans)
                n_drift += whitespace_drift(gold.entities, spans, rejected)
                n_rejected += len(rejected)
                n_dropped += sum(1 for s in doc.sections if s.label in lost)
                extraction = extraction_record(doc, gold.case, spans)
                qa = qa_records(doc, gold.case)
                pii = pii_values(gold)
                leaks += pii_leaks([extraction.target.model_dump_json()], pii)
                leaks += pii_leaks((f"{r.question}\n{r.answer}" for r in qa), pii)
                # a PII span that did not project could not be redacted: leave the case out
                keep_extraction = not any(s.label.is_pii for s in rejected)
                n_ext_dropped += not keep_extraction
                if keep_extraction:
                    input_leaks += pii_leaks([extraction.input_text], pii)
                built: dict[str, list[BaseModel]] = {
                    "ner": list(ner_records(doc, gold.case.case_id, spans, lost)),
                    "sections": list(section_records(doc)),
                    "extraction": [extraction] if keep_extraction else [],
                    "qa": list(qa),
                }
                split = split_of[gold.case.case_id]
                for task in d.tasks:
                    records[task][split] += built[task]
                    ner += [r for r in built[task] if isinstance(r, NerRecord)]
                    sections += [r for r in built[task] if isinstance(r, SectionRecord)]
            for name, by_split in records.items():
                for part, recs in by_split.items():
                    write_jsonl(tmp / name / f"{part}.jsonl", recs)

            total = n_projected + n_rejected
            reject_fraction = round(n_rejected / total, 6) if total else 0.0
            missing = len(docs) - len(pairs)
            cases = {s: {c for c, x in split_of.items() if x == s} for s in SPLITS}
            overlap = len(cases["train"] & cases["validation"]) + len(
                (cases["train"] | cases["validation"]) & cases["test"]
            )
            held = [g.case.case_id for _, g in pairs if g.template_id in holdout]
            bad_spans = sum(1 for r in ner if not _verifies(r))
            budget = d.max_align_reject_fraction
            run.check("every_doc_has_gold", not missing, f"{missing} document(s) without gold")
            run.check("splits_disjoint_by_case", not overlap, f"{overlap} case(s) in two splits")
            run.check(
                "holdout_only_in_test",
                all(split_of[c] == "test" for c in held),
                f"{len(held)} held-out case(s)",
            )
            run.check(
                "align_rejects_within_budget",
                reject_fraction <= budget,
                f"reject fraction {reject_fraction}, budget {budget}",
            )
            run.check("no_pii_in_targets", not leaks, f"{leaks} extraction/QA record(s)")
            run.check(
                "no_pii_in_extraction_inputs", not input_leaks, f"{input_leaks} extraction input(s)"
            )
            run.check("ner_spans_verify", not bad_spans, f"{bad_spans} NER record(s)")

            metrics: dict[str, Any] = {
                "seed": params.seed,
                "documents": len(pairs),
                "holdout_templates": list(holdout),
                "cases": {s: len(cases[s]) for s in SPLITS},
                "records": {
                    t: {s: len(recs) for s, recs in by_split.items()}
                    for t, by_split in records.items()
                },
                "ner_labels": dict(
                    sorted(Counter(str(x.label) for r in ner for x in r.spans).items())
                ),
                "section_labels": dict(sorted(Counter(str(r.label) for r in sections).items())),
                "alignment": {
                    "projected": n_projected,
                    "exact": n_projected - n_drift,
                    "whitespace_drift": n_drift,
                    "rejected": n_rejected,
                    "reject_fraction": reject_fraction,
                    "ner_records_dropped": n_dropped,
                    "extraction_records_dropped": n_ext_dropped,
                },
            }
            write_text_atomic(tmp / "README.md", render_card(metrics))
            tree = sha256_tree(tmp)
            written = sum(len(r) for by_split in records.values() for r in by_split.values())
            run.counts = Counts(read=len(docs), written=written, rejected=missing)
            run.raise_if_failed()
            run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=written)]

    metrics["datasets_sha256"] = tree
    write_text_atomic(metrics_dir / "build_dataset.json", canonical_json(metrics))
    return run
