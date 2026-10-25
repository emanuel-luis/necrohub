import json
import random
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.documents import ProcessedDocument
from necrohub.generate import stage as generate_stage
from necrohub.ids import fake_cpf
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, IngestParams, Params, PreprocessParams
from necrohub.preprocess import stage
from necrohub.preprocess.pipeline import process
from necrohub.runs import StageFailed, read_jsonl, sha256_tree

runner = CliRunner()
PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=8,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.5,
    ),
)


@pytest.fixture(scope="module")
def layers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<base>/raw and <base>/interim from one generate and ingest run."""
    base = tmp_path_factory.mktemp("layers")
    generate_stage.run_generate(PARAMS, base / "raw", runs_dir=base / "runs", metrics_dir=base)
    run_ingest(
        IngestParams(), base / "raw", base / "interim", runs_dir=base / "runs", metrics_dir=base
    )
    return base


@pytest.fixture
def base(layers: Path, tmp_path: Path) -> Path:
    for name in ("raw", "interim"):
        shutil.copytree(layers / name, tmp_path / name)
    return tmp_path


def run(base: Path) -> None:
    stage.run_preprocess(
        PreprocessParams(),
        base / "interim",
        base / "processed",
        runs_dir=base / "runs",
        metrics_dir=base / "metrics",
    )


def test_preprocess_writes_documents_and_metrics(base: Path) -> None:
    run(base)
    docs = read_jsonl(base / "processed" / "documents.jsonl", ProcessedDocument)
    assert [d.doc_id for d in docs] == sorted(f.stem for f in (base / "raw" / "gold").iterdir())
    assert all(d.deidentifiers == ["regex"] and "[CPF]" in d.redacted_text for d in docs)
    assert (base / "processed" / "rejects.jsonl").read_text("utf-8") == ""
    m = json.loads((base / "metrics" / "preprocess.json").read_text("utf-8"))
    assert m["written"] == 8 and m["processed_sha256"] == sha256_tree(base / "processed")
    assert m["section_coverage_min"] >= 0.95 and m["sections_by_label"]["CONCLUSION"] == 8
    align = m["gold_alignment"]
    assert align["txt_rejected"] == 0  # a txt only goes through normalize, a no-op here
    assert align["pdf_projected"] > 0 and align["reject_fraction"] <= 0.05


def test_alignment_metric_is_null_without_gold(base: Path) -> None:
    shutil.rmtree(base / "raw")
    run(base)
    m = json.loads((base / "metrics" / "preprocess.json").read_text("utf-8"))
    assert m["gold_alignment"] is None and m["written"] == 8


def test_failed_check_keeps_the_previous_out(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run(base)
    before = sha256_tree(base / "processed")
    monkeypatch.setattr(stage, "doc_problems", lambda doc: ["pii_after_redaction"])
    with pytest.raises(StageFailed, match="no_pii_after_redaction"):
        run(base)
    assert sha256_tree(base / "processed") == before
    assert sorted(p.name for p in base.iterdir()) == [
        "interim",
        "metrics",
        "processed",
        "raw",
        "runs",
    ]
    statuses = sorted(
        json.loads(m.read_text("utf-8"))["status"] for m in (base / "runs" / "preprocess").iterdir()
    )
    assert statuses == ["failed", "succeeded"]


def test_doc_problems_finds_pii_left_after_redaction(report: tuple[str, str]) -> None:
    doc = ProcessedDocument(doc_id="x", **process(report[1]).model_dump())
    assert stage.doc_problems(doc) == []
    leaked = doc.model_copy(
        update={"redacted_text": f"{doc.redacted_text}\nCPF nº {fake_cpf(random.Random(0))}\n"}
    )
    assert stage.doc_problems(leaked) == ["pii_after_redaction"]


def test_doc_problems_finds_low_section_coverage() -> None:
    doc = ProcessedDocument(doc_id="x", **process("Laudo\nCONCLUSÃO\nasfixia").model_dump())
    assert stage.section_coverage(doc) == 1.0
    partial = doc.model_copy(update={"sections": doc.sections[1:]})
    assert stage.section_coverage(partial) < 0.95
    assert stage.doc_problems(partial) == ["section_coverage"]


def test_hf_ner_without_a_model_is_rejected_by_params() -> None:
    with pytest.raises(ValidationError, match="hf_ner"):
        PreprocessParams(deidentifiers=["regex", "hf_ner"])


def test_cli_preprocess(base: Path) -> None:
    params = Path(__file__).parents[2] / "params.yaml"
    args = ["preprocess", "--params", str(params), "--in", str(base / "interim")]
    args += ["--out", str(base / "processed"), "--runs-dir", str(base / "runs")]
    args += ["--metrics-dir", str(base / "m")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "preprocess: 8 documents" in result.output
    m = json.loads((base / "m" / "preprocess.json").read_text("utf-8"))
    assert m["gold_alignment"]["projected"] > 0  # raw found next to --in
