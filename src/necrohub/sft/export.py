"""The export-sft stage: dataset records -> chat-format SFT examples and the extraction schema."""

import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from necrohub.contracts.datasets import CaseExtraction, ExtractionRecord, QaRecord
from necrohub.datasets.split import SPLITS
from necrohub.params import Params
from necrohub.preprocess.deid.regex import RegexDeidentifier
from necrohub.runs import (
    Counts,
    FileRef,
    RunManifest,
    atomic_dir,
    canonical_json,
    read_jsonl,
    sha256_tree,
    stage_run,
    write_text_atomic,
)

Messages = list[dict[str, str]]


def system_prompt(task: str) -> str:
    """The packaged system prompt of a task (`extraction` or `qa`), without the final newline."""
    raw = files("necrohub.sft").joinpath(f"prompts/{task}_system.txt").read_text("utf-8")
    return raw.strip()


def extraction_messages(rec: ExtractionRecord) -> Messages:
    """System prompt, the redacted report, and the canonical JSON of the target."""
    return [
        {"role": "system", "content": system_prompt("extraction")},
        {"role": "user", "content": rec.input_text},
        {"role": "assistant", "content": rec.target.model_dump_json()},
    ]


def qa_messages(rec: QaRecord, context: str) -> Messages:
    """System prompt, the redacted report with the question, and the answer."""
    return [
        {"role": "system", "content": system_prompt("qa")},
        {"role": "user", "content": f"Laudo:\n{context}\n\nPergunta: {rec.question}"},
        {"role": "assistant", "content": rec.answer},
    ]


def extraction_schema() -> str:
    """The JSON schema of the extraction target (Ollama's `format`), in field order."""
    return json.dumps(CaseExtraction.model_json_schema(), ensure_ascii=False, indent=2) + "\n"


def _line(task: str, id_: str, doc_id: str, messages: Messages) -> str:
    row = {"messages": messages, "task": task, "id": id_, "doc_id": doc_id}
    return json.dumps(row, ensure_ascii=False) + "\n"


def _chars(messages: Messages) -> int:
    return sum(len(m["content"]) for m in messages)


def _round_trips(messages: Messages) -> bool:
    try:
        CaseExtraction.model_validate_json(messages[-1]["content"])
    except ValidationError:
        return False
    return True


def _pii_texts(rows: Iterable[Messages]) -> int:
    deid = RegexDeidentifier()
    return sum(1 for m in rows if any(deid.detect(x["content"]) for x in m[1:]))


def run_export_sft(
    params: Params,
    data_dir: Path,
    out: Path,
    *,
    runs_dir: Path = Path("runs"),
    metrics_dir: Path = Path("metrics"),
) -> RunManifest:
    """Write <out>/{train,validation,test}.jsonl and <out>/extraction.schema.json.

    Reads data_dir/extraction and data_dir/qa. A QA record whose report is not among the
    extraction records is a counted reject (the dataset stage drops extraction records whose PII
    did not project). A failed check raises StageFailed and keeps the previous out.
    """
    tasks = params.sft.tasks
    with stage_run("export_sft", {"sft": params.sft.model_dump()}, runs_dir) as run:
        ext = {
            s: read_jsonl(data_dir / "extraction" / f"{s}.jsonl", ExtractionRecord) for s in SPLITS
        }
        qa = {s: read_jsonl(data_dir / "qa" / f"{s}.jsonl", QaRecord) for s in SPLITS}
        run.inputs = [
            FileRef(path=(data_dir / t).as_posix(), sha256=sha256_tree(data_dir / t))
            for t in ("extraction", "qa")
        ]
        context = {r.doc_id: r.input_text for recs in ext.values() for r in recs}
        n_reject = 0
        rows: dict[str, list[tuple[str, str, str, Messages]]] = {s: [] for s in SPLITS}
        for split in SPLITS:
            if "extraction" in tasks:
                for e in ext[split]:
                    rows[split].append(("extraction", e.id, e.doc_id, extraction_messages(e)))
            if "qa" in tasks:
                for q in qa[split]:
                    if q.doc_id in context:
                        msgs = qa_messages(q, context[q.doc_id])
                        rows[split].append(("qa", q.id, q.doc_id, msgs))
                    else:
                        n_reject += 1
        docs = {s: {r[2] for r in rows[s]} for s in SPLITS}
        leaked_docs = docs["test"] & (docs["train"] | docs["validation"])
        bad_json = sum(
            1
            for rs in rows.values()
            for t, _, _, m in rs
            if t == "extraction" and not _round_trips(m)
        )
        pii = _pii_texts(m for rs in rows.values() for _, _, _, m in rs)
        with atomic_dir(out, run.run_id) as tmp:
            for split in SPLITS:
                text = "".join(_line(*r) for r in rows[split])
                write_text_atomic(tmp / f"{split}.jsonl", text)
            write_text_atomic(tmp / "extraction.schema.json", extraction_schema())
            run.check("test_split_isolated", not leaked_docs, f"{len(leaked_docs)} doc(s)")
            run.check("assistant_json_valid", not bad_json, f"{bad_json} example(s)")
            run.check("no_pii_in_messages", not pii, f"{pii} example(s)")
            sizes = [_chars(r[3]) for rs in rows.values() for r in rs]
            written = len(sizes)
            run.counts = Counts(
                read=sum(map(len, ext.values())) + sum(map(len, qa.values())),
                written=written,
                rejected=n_reject,
            )
            run.raise_if_failed()
            tree = sha256_tree(tmp)
            run.outputs = [FileRef(path=out.as_posix(), sha256=tree, records=written)]

    metrics: Mapping[str, Any] = {
        "tasks": list(tasks),
        "examples": {s: {t: sum(1 for r in rows[s] if r[0] == t) for t in tasks} for s in SPLITS},
        "rejects": {"qa_without_context": n_reject},
        "chars": {
            "max": max(sizes, default=0),
            "mean": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        },
        "sft_sha256": tree,
    }
    write_text_atomic(metrics_dir / "export_sft.json", canonical_json(metrics))
    return run
