from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """One row per text, shape (n, dim), each row L2-normalized."""
        ...


def cosine_matrix(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Cosine similarity of every row of a with every row of b, shape (len(a), len(b))."""
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.asarray(an @ bn.T, dtype=np.float32)
