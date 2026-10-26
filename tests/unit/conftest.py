from pathlib import Path

import pytest

from necrohub.generate.stage import run_generate
from necrohub.index.build import run_index_build
from necrohub.ingest.stage import run_ingest
from necrohub.params import GenerateParams, Params
from necrohub.preprocess.stage import run_preprocess

PARAMS = Params(
    seed=7,
    generate=GenerateParams(
        n_cases=6,
        realizer="template",
        holdout_templates=["t3"],
        max_reject_fraction=0.2,
        pdf_fraction=0.0,
    ),
)


@pytest.fixture(scope="session")
def serving_layers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<base>/{raw,interim,processed,index} from one small generate-to-index run (seed 7)."""
    base = tmp_path_factory.mktemp("serving")
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(PARAMS, base / "raw", **kw)
    run_ingest(PARAMS.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(PARAMS.preprocess, base / "interim", base / "processed", **kw)
    run_index_build(PARAMS, base / "processed", base / "raw" / "gold", base / "index", **kw)
    return base
