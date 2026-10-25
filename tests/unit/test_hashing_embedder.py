from collections.abc import Callable

import numpy as np
import pytest

from necrohub.embed.base import cosine_matrix
from necrohub.embed.hashing import HashingEmbedder


def test_vectors_golden(report: tuple[str, str], golden: Callable[[str, object], None]) -> None:
    name, text = report
    vec = HashingEmbedder().embed([text])[0]
    nonzero = np.flatnonzero(vec)[:12]
    golden(
        f"hashing/{name}.json",
        {"id": "hashing-256", "head": [[int(i), round(float(vec[i]), 6)] for i in nonzero]},
    )


def test_deterministic_unit_norm(report: tuple[str, str]) -> None:
    emb = HashingEmbedder(dim=64)
    out = emb.embed([report[1], "outro texto"])
    assert out.shape == (2, 64) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out, HashingEmbedder(dim=64).embed([report[1], "outro texto"]))
    assert emb.id == "hashing-64"


def test_cosine_matrix() -> None:
    emb = HashingEmbedder()
    a = emb.embed(["ferimento por arma de fogo", "infarto agudo do miocárdio"])
    sim = cosine_matrix(a, a)
    assert sim.shape == (2, 2)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)


@pytest.mark.parametrize("text", ["", "   \n", "!!! ..."])
def test_text_without_words_raises(text: str) -> None:
    with pytest.raises(ValueError):
        HashingEmbedder().embed([text])
