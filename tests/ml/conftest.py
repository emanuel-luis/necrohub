import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertTokenizerFast

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
