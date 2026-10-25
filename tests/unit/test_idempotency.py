from pathlib import Path

from necrohub.datasets.stage import run_build_dataset
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
    run_build_dataset(PARAMS, base / "processed", base / "raw", base / "datasets", **kw)
    layers = ("raw", "interim", "processed", "datasets")
    return {layer: sha256_tree(base / layer) for layer in layers}


def test_every_layer_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first, second = _pipeline(tmp_path / "a"), _pipeline(tmp_path / "b")
    assert first == second
    assert any((tmp_path / "a" / "raw" / "reports").glob("*.pdf"))  # the PDF path is covered
    for name in ("preprocess.json", "build_dataset.json"):
        a, b = (tmp_path / x / "metrics" / name for x in ("a", "b"))
        assert a.read_text("utf-8") == b.read_text("utf-8")
