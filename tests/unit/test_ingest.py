import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.documents import DocReject, IngestedDocument
from necrohub.generate import stage as generate_stage
from necrohub.generate.render_pdf import render_pdf
from necrohub.ingest.readers import extractor, pdf_text, txt_text
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, IngestParams, Params
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
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("gen")
    generate_stage.run_generate(PARAMS, base / "raw", runs_dir=base / "runs", metrics_dir=base)
    return base / "raw"


@pytest.fixture
def raw(generated: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(generated, tmp_path / "raw"))


def ingest(raw: Path, out: Path, p: IngestParams | None = None) -> None:
    run_ingest(p or IngestParams(), raw, out, runs_dir=raw.parent / "runs", metrics_dir=raw.parent)


def case_ids(raw: Path, suffix: str) -> list[str]:
    return sorted(f.stem for f in (raw / "reports").glob(f"*.{suffix}"))


def test_txt_reader_marks_bad_bytes() -> None:
    assert txt_text("ação".encode()) == "ação"
    assert txt_text(b"a\xffb") == "a�b"


def test_pdf_reader_extracts_rendered_text() -> None:
    text = "Laudo de Necropsia\n\nConclusão: asfixia mecânica.\n"
    extracted = pdf_text(render_pdf(text)[0])
    assert "Conclusão: asfixia mecânica." in extracted
    assert extractor("pdf").startswith("pypdf@") and extractor("txt").startswith("necrohub-utf8@")


def test_ingest_reads_pdfs_when_present_and_txt_otherwise(raw: Path) -> None:
    out = raw.parent / "interim"
    ingest(raw, out)
    docs = read_jsonl(out / "ingested.jsonl", IngestedDocument)
    pdfs = case_ids(raw, "pdf")
    assert pdfs and len(pdfs) < 8
    assert [d.doc_id for d in docs] == case_ids(raw, "txt")
    for d in docs:
        fmt = "pdf" if d.doc_id in pdfs else "txt"
        assert d.source_format == fmt and d.source_path == f"reports/{d.doc_id}.{fmt}"
        assert d.extractor.startswith("pypdf@" if fmt == "pdf" else "necrohub-utf8@")
        if fmt == "txt":
            assert d.text == (raw / d.source_path).read_text("utf-8") and d.warnings == []
        else:
            assert len(d.warnings) == 1
    assert (out / "rejects.jsonl").read_text("utf-8") == ""
    metrics = json.loads((raw.parent / "ingest.json").read_text("utf-8"))
    assert metrics["written"] == 8 and metrics["by_format"]["pdf"] == len(pdfs)
    assert metrics["interim_sha256"] == sha256_tree(out)
    assert metrics["pdf_length_drift_max"] < 0.15


def test_txt_only_formats_ignore_pdfs(raw: Path) -> None:
    out = raw.parent / "interim"
    ingest(raw, out, IngestParams(formats=["txt"]))
    docs = read_jsonl(out / "ingested.jsonl", IngestedDocument)
    assert {d.source_format for d in docs} == {"txt"}


def _expect_failure(raw: Path, check: str, detail: str) -> None:
    out = raw.parent / "interim"
    out.mkdir()
    (out / "previous.txt").write_text("kept", encoding="utf-8")
    with pytest.raises(StageFailed, match=check):
        ingest(raw, out)
    assert sorted(p.name for p in out.iterdir()) == ["previous.txt"]
    manifests = [json.loads(m.read_text("utf-8")) for m in (raw.parent / "runs/ingest").iterdir()]
    assert [m["status"] for m in manifests] == ["failed"] and manifests[0]["outputs"] == []
    assert detail in json.dumps(manifests[0]["checks"])


def test_replacement_char_fails(raw: Path) -> None:
    txt_only = sorted(set(case_ids(raw, "txt")) - set(case_ids(raw, "pdf")))
    path = raw / "reports" / f"{txt_only[0]}.txt"
    path.write_bytes(path.read_bytes() + b"\xff")
    _expect_failure(raw, "no_replacement_char", "U+FFFD")


def test_missing_report_fails(raw: Path) -> None:
    txt_only = sorted(set(case_ids(raw, "txt")) - set(case_ids(raw, "pdf")))
    (raw / "reports" / f"{txt_only[0]}.txt").unlink()
    _expect_failure(raw, "one_doc_per_report", "1 case(s) without a document")


def test_pdf_length_drift_over_budget_fails(raw: Path) -> None:
    pdf = raw / "reports" / f"{case_ids(raw, 'pdf')[0]}.pdf"
    text = (pdf.with_suffix(".txt")).read_text("utf-8")
    pdf.write_bytes(render_pdf(text[: len(text) // 2])[0])  # a PDF that lost half its text
    _expect_failure(raw, "pdf_length_within_budget", "PDF(s) over 0.15")


def test_unreadable_pdf_is_rejected(raw: Path) -> None:
    pdf = raw / "reports" / f"{case_ids(raw, 'pdf')[0]}.pdf"
    pdf.write_bytes(b"not a pdf")
    out = raw.parent / "interim"
    with pytest.raises(StageFailed, match="one_doc_per_report"):
        ingest(raw, out)
    assert not out.exists()


def test_reject_records_round_trip() -> None:
    r = DocReject(doc_id="c7-00001", reason="pdf_length_drift", detail="drift 0.5 > 0.15")
    assert DocReject.model_validate_json(r.model_dump_json()) == r


def test_cli_ingest(raw: Path) -> None:
    params = Path(__file__).parents[2] / "params.yaml"
    out = raw.parent / "interim"
    args = ["ingest", "--params", str(params), "--in", str(raw), "--out", str(out)]
    args += ["--runs-dir", str(raw.parent / "runs"), "--metrics-dir", str(raw.parent / "m")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "ingest: 8 documents" in result.output
