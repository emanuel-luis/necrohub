from pathlib import Path

import pytest

from necrohub.embed.base import cosine_matrix

pytestmark = pytest.mark.external

REPORTS = Path(__file__).parents[1] / "fixtures" / "reports"


def test_similar_reports_score_higher() -> None:
    from necrohub.embed.hf import HfEmbedder

    emb = HfEmbedder("pucpr/biobertpt-clin")
    a, b, c = emb.embed(
        [(REPORTS / f"{n}.txt").read_text("utf-8") for n in ("firearm_a", "firearm_b", "natural")]
    )
    sims = cosine_matrix(a[None], b[None])[0, 0], cosine_matrix(a[None], c[None])[0, 0]
    assert sims[0] > sims[1]
