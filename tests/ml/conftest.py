import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertTokenizerFast

from necrohub.params import GenerateParams, Params, TinyParams, TrainParams

REPORTS = Path(__file__).parents[1] / "fixtures" / "reports"


def fixture_text(name: str) -> str:
    return (REPORTS / f"{name}.txt").read_text("utf-8")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A WordPiece tokenizer whose vocab is the words and punctuation of the fixtures."""
    out = tmp_path_factory.mktemp("tiny-tokenizer")
    words: set[str] = set()
    for path in REPORTS.glob("*.txt"):
        words.update(re.findall(r"\w+|[^\w\s]", path.read_text("utf-8").lower()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(words)]
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(
        vocab_file=str(out / "vocab.txt"), strip_accents=False, model_max_length=64
    )
    tokenizer.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def p4_layers(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """<base>/raw and <base>/datasets from a small generate-to-build-dataset run."""
    from necrohub.datasets.stage import run_build_dataset
    from necrohub.generate.stage import run_generate
    from necrohub.ingest.stage import run_ingest
    from necrohub.preprocess.stage import run_preprocess

    base = tmp_path_factory.mktemp("p4-layers")
    params = p4_params()
    kw = {"runs_dir": base / "runs", "metrics_dir": base / "metrics"}
    run_generate(params, base / "raw", **kw)
    run_ingest(params.ingest, base / "raw", base / "interim", **kw)
    run_preprocess(params.preprocess, base / "interim", base / "processed", **kw)
    run_build_dataset(params, base / "processed", base / "raw", base / "datasets", **kw)
    return base


def p4_params(_base: Path | None = None) -> Params:
    """The params the p4_layers corpus was built with."""
    return Params(
        seed=7,
        generate=GenerateParams(
            n_cases=16,
            realizer="template",
            holdout_templates=["t3"],
            max_reject_fraction=0.2,
            pdf_fraction=0.0,
        ),
        train=TrainParams(epochs=1, tiny=TinyParams(vocab_size=600)),
    )
