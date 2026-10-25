import hashlib
import re
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import numpy.typing as npt

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Feature hashing of lowercased word unigrams and bigrams. No model, no network.

    Uses blake2b, never hash(), so vectors are identical across processes and platforms.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.id = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in zip(out, texts, strict=True):
            words = _WORD.findall(text.lower())
            if not words:
                raise ValueError("text has no word tokens, cannot embed it")
            for feature in [*words, *(f"{a} {b}" for a, b in pairwise(words))]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
                row[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            # never zero: 2n-1 features of +-1 cannot all cancel
            row /= np.linalg.norm(row)
        return out
