import json
import shutil
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from typer.testing import CliRunner

from necrohub.cli import app
from necrohub.contracts.datasets import (
    CaseExtraction,
    ExtractionRecord,
    NerRecord,
    QaRecord,
    SectionRecord,
)
from necrohub.contracts.spans import EntityLabel, SectionLabel, Span
from necrohub.datasets import build, stage
from necrohub.datasets.split import assign_splits
from necrohub.generate.stage import run_generate
from necrohub.ingest.stage import run_ingest
from necrohub.params import DatasetParams, GenerateParams, Params, SplitParams
from necrohub.preprocess.stage import run_preprocess
from necrohub.runs import StageFailed, read_jsonl, sha256_tree

PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=12,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.5,
    ),
)
SPLITS = ("train", "validation", "test")


@pytest.fixture(scope="module")
def layers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<base>/raw and <base>/processed from one generate, ingest and preprocess run."""
    base = tmp_path_factory.mktemp("layers")
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(PARAMS, base / "raw", **kw)
    run_ingest(PARAMS.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(PARAMS.preprocess, base / "interim", base / "processed", **kw)
    return base


@pytest.fixture
def base(layers: Path, tmp_path: Path) -> Path:
    for name in ("raw", "processed"):
        shutil.copytree(layers / name, tmp_path / name)
    return tmp_path


def run(base: Path, params: Params = PARAMS) -> None:
    stage.run_build_dataset(
        params,
        base / "processed",
        base / "raw",
        base / "datasets",
        runs_dir=base / "runs",
        metrics_dir=base / "metrics",
    )


def _gold_templates(base: Path) -> dict[str, str]:
    return {
        f.stem: json.loads(f.read_text("utf-8"))["template_id"]
        for f in (base / "raw" / "gold").iterdir()
    }


def test_contracts_round_trip_and_forbid_extras() -> None:
    ner = NerRecord(
        id="d:PREAMBLE:0",
        doc_id="d",
        case_id="d",
        section=SectionLabel.PREAMBLE,
        text="Rins normais",
        spans=[Span(start=0, end=4, label=EntityLabel.ORGAN, text="Rins")],
    )
    target = CaseExtraction(
        scenario="fall",
        cause_of_death="x",
        mechanism="y",
        manner_of_death="z",
        time_since_death_h=(1, 2),
        injuries=[],
        organ_findings=[],
    )
    records = [
        ner,
        SectionRecord(id="d:PREAMBLE:0", doc_id="d", text="t", label=SectionLabel.PREAMBLE),
        ExtractionRecord(id="d:extraction", doc_id="d", input_text="t", target=target),
        QaRecord(
            id="d:qa:x", doc_id="d", question="q", answer="a", section=SectionLabel.CONCLUSION
        ),
    ]
    for r in records:
        assert type(r).model_validate_json(r.model_dump_json()) == r
    with pytest.raises(ValidationError):
        QaRecord.model_validate({**records[3].model_dump(), "extra": 1})
    with pytest.raises(ValidationError, match="does not match"):
        NerRecord.model_validate({**ner.model_dump(), "text": "rins normais"})


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        SplitParams(train=0.8, validation=0.1, test=0.2)


@given(
    ids=st.sets(st.from_regex(r"c[0-9]{1,5}", fullmatch=True), max_size=60),
    seed=st.integers(0, 1000),
)
def test_splits_are_deterministic_disjoint_and_keep_holdout_in_test(
    ids: set[str], seed: int
) -> None:
    templates = {c: ("t3" if int(c[1:]) % 4 == 0 else "t1") for c in ids}
    splits = SplitParams()
    first = assign_splits(templates, splits, ["t3"], seed)
    reversed_ = assign_splits(dict(reversed(list(templates.items()))), splits, ["t3"], seed)
    assert first == reversed_  # independent of input order
    assert set(first) == ids  # every case in exactly one split
    assert all(first[c] == "test" for c, t in templates.items() if t == "t3")
    rest = [c for c, t in templates.items() if t != "t3"]
    assert sum(first[c] == "train" for c in rest) == round(len(rest) * 0.8)


def test_build_writes_every_task_split_and_card(base: Path) -> None:
    run(base)
    out = base / "datasets"
    m = json.loads((base / "metrics" / "build_dataset.json").read_text("utf-8"))
    assert m["datasets_sha256"] == sha256_tree(out)
    assert m["documents"] == 12 and sum(m["cases"].values()) == 12
    assert m["alignment"]["reject_fraction"] <= 0.05
    models = {"ner": NerRecord, "sections": SectionRecord, "extraction": ExtractionRecord}
    models["qa"] = QaRecord
    templates = _gold_templates(base)
    docs_by_split: dict[str, set[str]] = {}
    for task, model in models.items():
        for split in SPLITS:
            recs = read_jsonl(out / task / f"{split}.jsonl", model)
            assert len(recs) == m["records"][task][split]
            docs_by_split.setdefault(split, set()).update(r.doc_id for r in recs)
    assert not docs_by_split["train"] & docs_by_split["test"]
    assert not docs_by_split["train"] & docs_by_split["validation"]
    held = {c for c, t in templates.items() if t == "t3"}
    assert held and held <= docs_by_split["test"]
    ner = read_jsonl(out / "ner" / "train.jsonl", NerRecord)
    assert {s.label for r in ner for s in r.spans} >= {EntityLabel.PER, EntityLabel.ORGAN}
    sections = read_jsonl(out / "sections" / "train.jsonl", SectionRecord)
    conclusion = next(r for r in sections if r.label == SectionLabel.CONCLUSION)
    assert "conclus" not in conclusion.text.splitlines()[0].lower()  # header stripped
    ext = read_jsonl(out / "extraction" / "train.jsonl", ExtractionRecord)
    assert all("[CPF]" in r.input_text for r in ext)
    card = (out / "README.md").read_text("utf-8")
    assert "Synthetic data only" in card and "t3" in card


def test_pii_leak_in_a_target_fails_the_check(base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = build.qa_records

    def leaky(doc, case):  # type: ignore[no-untyped-def]
        recs = real(doc, case)
        return [recs[0].model_copy(update={"answer": case.decedent.name}), *recs[1:]]

    monkeypatch.setattr(stage, "qa_records", leaky)
    with pytest.raises(StageFailed, match="no_pii_in_targets"):
        run(base)


def test_alignment_budget_failure_keeps_previous_out(
    base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(base)
    before = sha256_tree(base / "datasets")
    real = build.project

    def lossy(source, target, spans):  # type: ignore[no-untyped-def]
        projected, rejected = real(source, target, spans)
        return projected[1:], rejected + projected[:1]

    monkeypatch.setattr(build, "project", lossy)
    strict = PARAMS.model_copy(update={"dataset": DatasetParams(max_align_reject_fraction=0.0)})
    with pytest.raises(StageFailed, match="align_rejects_within_budget"):
        run(base, strict)
    assert sha256_tree(base / "datasets") == before
    statuses = sorted(
        json.loads(p.read_text("utf-8"))["status"]
        for p in (base / "runs" / "build_dataset").iterdir()
    )
    assert statuses == ["failed", "succeeded"]


def test_cli_build_dataset(base: Path) -> None:
    params = Path(__file__).parents[2] / "params.yaml"
    args = ["build-dataset", "--params", str(params), "--processed", str(base / "processed")]
    args += ["--raw", str(base / "raw"), "--out", str(base / "datasets")]
    args += ["--runs-dir", str(base / "runs"), "--metrics-dir", str(base / "m")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (base / "m" / "build_dataset.json").is_file()
