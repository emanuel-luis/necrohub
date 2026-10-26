import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.datasets import CaseExtraction, ExtractionRecord, QaRecord
from necrohub.datasets.stage import run_build_dataset
from necrohub.generate.stage import run_generate
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, Params
from necrohub.preprocess.stage import run_preprocess
from necrohub.runs import StageFailed, read_jsonl, sha256_tree, write_jsonl
from necrohub.sft.export import (
    extraction_messages,
    extraction_schema,
    qa_messages,
    run_export_sft,
    system_prompt,
)

PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=12,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.0,
    ),
)
SPLITS = ("train", "validation", "test")
runner = CliRunner()


@pytest.fixture(scope="module")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("sft-layers")
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(PARAMS, base / "raw", **kw)
    run_ingest(PARAMS.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(PARAMS.preprocess, base / "interim", base / "processed", **kw)
    run_build_dataset(PARAMS, base / "processed", base / "raw", base / "datasets", **kw)
    return base / "datasets"


def _export(data: Path, tmp: Path, name: str = "sft") -> Path:
    out = tmp / name
    run_export_sft(PARAMS, data, out, runs_dir=tmp / "runs", metrics_dir=tmp / f"m-{name}")
    return out


def _rec(data: Path) -> tuple[ExtractionRecord, QaRecord]:
    ext = read_jsonl(data / "extraction" / "train.jsonl", ExtractionRecord)[0]
    qa = next(
        q for q in read_jsonl(data / "qa" / "train.jsonl", QaRecord) if q.doc_id == ext.doc_id
    )
    return ext, qa


def test_extraction_messages_roles_and_round_trip(datasets: Path) -> None:
    ext, _ = _rec(datasets)
    msgs = extraction_messages(ext)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == system_prompt("extraction")
    assert msgs[1]["content"] == ext.input_text
    assert msgs[2]["content"] == ext.target.model_dump_json()
    assert "\n" not in msgs[2]["content"]
    assert CaseExtraction.model_validate_json(msgs[2]["content"]) == ext.target


def test_qa_messages_carry_the_report_as_context(datasets: Path) -> None:
    ext, qa = _rec(datasets)
    msgs = qa_messages(qa, ext.input_text)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert ext.input_text in msgs[1]["content"] and qa.question in msgs[1]["content"]
    assert msgs[2]["content"] == qa.answer


def test_export_is_deterministic_and_complete(datasets: Path, tmp_path: Path) -> None:
    a, b = _export(datasets, tmp_path, "a"), _export(datasets, tmp_path, "b")
    assert sha256_tree(a) == sha256_tree(b)
    ma, mb = (tmp_path / f"m-{x}" / "export_sft.json" for x in "ab")
    assert ma.read_bytes() == mb.read_bytes()
    schema = json.loads((a / "extraction.schema.json").read_text("utf-8"))
    assert schema["properties"].keys() == CaseExtraction.model_fields.keys()
    metrics = json.loads(ma.read_text("utf-8"))
    assert metrics["sft_sha256"] == sha256_tree(a)
    docs: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = [json.loads(x) for x in (a / f"{split}.jsonl").read_text("utf-8").splitlines()]
        assert len(rows) == sum(metrics["examples"][split].values())
        assert {r["task"] for r in rows} <= {"extraction", "qa"}
        docs[split] = {r["doc_id"] for r in rows}
    assert not docs["test"] & (docs["train"] | docs["validation"])


def test_schema_text_matches_the_model() -> None:
    assert json.loads(extraction_schema()) == CaseExtraction.model_json_schema()


def test_qa_without_context_is_a_counted_reject(datasets: Path, tmp_path: Path) -> None:
    data = tmp_path / "datasets"
    shutil.copytree(datasets, data)
    recs = read_jsonl(data / "extraction" / "train.jsonl", ExtractionRecord)
    write_jsonl(data / "extraction" / "train.jsonl", recs[1:])
    qa = read_jsonl(data / "qa" / "train.jsonl", QaRecord)
    dropped = sum(1 for q in qa if q.doc_id == recs[0].doc_id)
    _export(data, tmp_path)
    metrics = json.loads((tmp_path / "m-sft" / "export_sft.json").read_text("utf-8"))
    assert metrics["rejects"]["qa_without_context"] == dropped > 0


def test_pii_in_a_message_fails_the_stage_and_keeps_the_previous_out(
    datasets: Path, tmp_path: Path
) -> None:
    data = tmp_path / "datasets"
    shutil.copytree(datasets, data)
    good = _export(data, tmp_path)
    before = sha256_tree(good)
    recs = read_jsonl(data / "extraction" / "train.jsonl", ExtractionRecord)
    leaky = recs[0].model_copy(update={"input_text": recs[0].input_text + "\nDr. Fulano de Tal"})
    write_jsonl(data / "extraction" / "train.jsonl", [leaky, *recs[1:]])
    with pytest.raises(StageFailed):
        _export(data, tmp_path)
    assert sha256_tree(good) == before


def test_export_sft_cli(datasets: Path, tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(
        "seed: 7\n"
        "generate: {n_cases: 1, realizer: template, holdout_templates: [], max_reject_fraction: 0.2}\n"
        "sft: {tasks: [extraction]}\n",
        "utf-8",
    )
    args = ["export-sft", "--params", str(params), "--data", str(datasets)]
    args += ["--out", str(tmp_path / "o"), "--runs-dir", str(tmp_path / "r")]
    args += ["--metrics-dir", str(tmp_path / "m")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "m" / "export_sft.json").read_text("utf-8"))
    assert metrics["tasks"] == ["extraction"]
