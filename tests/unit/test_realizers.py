import random

import pytest

from necrohub.contracts.gold import Draft, Realized
from necrohub.contracts.spans import SectionLabel as L
from necrohub.generate.fill import fill
from necrohub.generate.realizers import TemplateRealizer, load_families
from necrohub.generate.sampler import sample_case
from necrohub.generate.vocab import SCENARIOS
from necrohub.preprocess.sections import split_sections

REALIZER = TemplateRealizer()
FULL = [
    L.PREAMBLE,
    L.IDENTIFICATION,
    L.HISTORY,
    L.EXTERNAL_EXAM,
    L.INTERNAL_EXAM,
    L.COMPLEMENTARY_EXAMS,
    L.CONCLUSION,
    L.SIGNATURE,
]


def test_at_least_three_families_including_the_holdout() -> None:
    assert {"t1", "t2", "t3"} <= set(load_families())


@pytest.mark.parametrize("family", sorted(load_families()))
@pytest.mark.parametrize("index", range(len(SCENARIOS)), ids=SCENARIOS)
def test_every_family_renders_every_scenario(family: str, index: int) -> None:
    for seed in (7, 8):  # two cases per cell: different phrase variants and optional injuries
        case = sample_case(seed, index)
        body = REALIZER.render(case, family, random.Random(seed))
        result = fill(Draft(template_id=family, body=body), case)
        assert isinstance(result, Realized), result
        result.text.encode("latin-1")
        assert result.text.endswith("\n") and "\r" not in result.text
        assert result.sections == split_sections(result.text)
        assert [s.label for s in result.sections] == FULL
        for span in result.entities:
            assert result.text[span.start : span.end] == span.text


def test_realize_is_deterministic() -> None:
    case = sample_case(7, 1)
    assert REALIZER.realize(case, 7) == REALIZER.realize(case, 7)
