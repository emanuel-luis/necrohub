"""Seeded train/validation/test assignment by case."""

import hashlib
from collections.abc import Collection, Mapping
from typing import Literal

from necrohub.params import SplitParams

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")


def _rank(seed: int, case_id: str) -> bytes:
    return hashlib.blake2b(f"{seed}:{case_id}".encode(), digest_size=8).digest()


def assign_splits(
    template_by_case: Mapping[str, str],
    splits: SplitParams,
    holdout_templates: Collection[str],
    seed: int,
) -> dict[str, Split]:
    """Map each case_id to a split. Cases from a held-out template go to test only.

    The other cases are ordered by a seeded hash of their id and cut by the split fractions, so
    small corpora still get the planned proportions. The result depends only on the case ids,
    their templates, the fractions and the seed, never on input order.
    """
    out: dict[str, Split] = {}
    rest = []
    for case_id, template_id in template_by_case.items():
        if template_id in holdout_templates:
            out[case_id] = "test"
        else:
            rest.append(case_id)
    rest.sort(key=lambda c: (_rank(seed, c), c))
    n_train = round(len(rest) * splits.train)
    n_val = min(round(len(rest) * splits.validation), len(rest) - n_train)
    for i, case_id in enumerate(rest):
        out[case_id] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
    return out
