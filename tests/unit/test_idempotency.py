from pathlib import Path

from necrohub.generate.stage import run_generate
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, Params
from necrohub.preprocess.stage import run_preprocess
from necrohub.runs import sha256_tree

PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=6,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.5,
    ),
)


def _pipeline(base: Path) -> dict[str, str]:
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(PARAMS, base / "raw", **kw)
    run_ingest(PARAMS.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(PARAMS.preprocess, base / "interim", base / "processed", **kw)
    return {layer: sha256_tree(base / layer) for layer in ("raw", "interim", "processed")}


def test_generate_ingest_preprocess_are_byte_identical_across_runs(tmp_path: Path) -> None:
    first, second = _pipeline(tmp_path / "a"), _pipeline(tmp_path / "b")
    assert first == second
    assert any((tmp_path / "a" / "raw" / "reports").glob("*.pdf"))  # the PDF path is covered
    assert (tmp_path / "a" / "metrics" / "preprocess.json").read_text("utf-8") == (
        tmp_path / "b" / "metrics" / "preprocess.json"
    ).read_text("utf-8")
