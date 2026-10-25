from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import fixture_text
from transformers import BertConfig, BertModel, BertTokenizerFast

from necrohub.embed.hf import HfEmbedder, window_ids

pytestmark = pytest.mark.ml

MAX_LENGTH = 32


@pytest.fixture(scope="module")
def tiny_model(tiny_tokenizer_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> str:
    tokenizer = BertTokenizerFast.from_pretrained(tiny_tokenizer_dir)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    out = tmp_path_factory.mktemp("tiny-bert")
    BertModel(config).save_pretrained(out)
    tokenizer.save_pretrained(out)
    return str(out)


@pytest.mark.parametrize("weighting", ["last", "linear_decay"])
def test_shape_norm_dtype_and_determinism(tiny_model: str, weighting: str) -> None:
    emb = HfEmbedder(
        tiny_model, layer_weighting=weighting, max_length=MAX_LENGTH, stride=4, device="cpu"
    )
    texts = [fixture_text("firearm_a"), fixture_text("natural")]
    out = emb.embed(texts)
    assert isinstance(out, np.ndarray)  # pooling returns arrays, not (values, indices) tuples
    assert out.shape == (2, 32) and emb.dim == 32
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(out, emb.embed(texts))
    assert emb.id == f"hf:{tiny_model}:{weighting}"


def test_weightings_differ(tiny_model: str) -> None:
    text = fixture_text("firearm_b")
    last = HfEmbedder(tiny_model, "last", MAX_LENGTH, 4, "cpu").embed([text])
    decay = HfEmbedder(tiny_model, "linear_decay", MAX_LENGTH, 4, "cpu").embed([text])
    assert not np.allclose(last, decay)


def test_every_window_wrapped_in_specials(tiny_model: str) -> None:
    emb = HfEmbedder(tiny_model, max_length=MAX_LENGTH, stride=4, device="cpu")
    ids = emb._tokenizer(fixture_text("firearm_a"), add_special_tokens=False)["input_ids"]
    chunks = window_ids(ids, emb._size, emb._stride, emb._cls, emb._sep)
    assert len(chunks) > 3
    assert all(c[0] == emb._cls and c[-1] == emb._sep for c in chunks)
    assert all(len(c) <= MAX_LENGTH for c in chunks)


def test_empty_text_raises(tiny_model: str) -> None:
    emb = HfEmbedder(tiny_model, max_length=MAX_LENGTH, stride=4, device="cpu")
    with pytest.raises(ValueError):
        emb.embed([""])
