import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from necrohub.contracts.gold import GoldFiles, GoldRecord, text_sha256
from necrohub.contracts.spans import EntityLabel, SectionLabel, SectionSpan, Span
from necrohub.generate.sampler import sample_case
from necrohub.ids import IDENTIFIER_PATTERNS, fake_values, load_pool
from necrohub.params import load_params

ROOT = Path(__file__).parents[2]


def _gold(text: str) -> GoldRecord:
    return GoldRecord(
        case=sample_case(7, 0),
        realizer="template",
        template_id="t1",
        text_sha256=text_sha256(text),
        entities=[Span(start=0, end=4, label=EntityLabel.ORGAN, text="Rins")],
        sections=[SectionSpan(start=0, end=len(text), label=SectionLabel.PREAMBLE)],
        files=GoldFiles(txt="reports/c7-00000.txt"),
    )


def test_gold_verify_accepts_its_text() -> None:
    _gold("Rins normais\n").verify("Rins normais\n")


def test_gold_verify_rejects_another_text_or_a_shifted_span() -> None:
    gold = _gold("Rins normais\n")
    with pytest.raises(ValueError, match="sha256"):
        gold.verify("rins normais\n")
    moved = gold.model_copy(update={"text_sha256": text_sha256("rins normais\n")})
    with pytest.raises(ValueError, match="does not match"):
        moved.verify("rins normais\n")


def test_gold_verify_needs_sections_to_tile_the_text() -> None:
    text = "Rins normais\n"
    gold = _gold(text).model_copy(
        update={"sections": [SectionSpan(start=0, end=5, label=SectionLabel.PREAMBLE)]}
    )
    with pytest.raises(ValueError, match="end of the text"):
        gold.verify(text)


def test_gold_round_trips_and_forbids_extra_fields() -> None:
    gold = _gold("Rins normais\n")
    assert GoldRecord.model_validate_json(gold.model_dump_json()) == gold
    with pytest.raises(ValidationError):
        GoldRecord.model_validate({**gold.model_dump(), "extra": 1})


def test_params_file_loads() -> None:
    params = load_params(ROOT / "params.yaml")
    assert params.seed == 7 and params.generate.n_cases == 24
    assert params.generate.holdout_templates == ["t3"]


def test_params_ignore_unknown_sections_but_not_unknown_keys(tmp_path: Path) -> None:
    base = (ROOT / "params.yaml").read_text("utf-8")
    ok = tmp_path / "ok.yaml"
    ok.write_text(base + "train:\n  epochs: 2\n", encoding="utf-8")
    assert load_params(ok).generate.realizer == "template"
    bad = tmp_path / "bad.yaml"
    bad.write_text(base + "  typo: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_params(bad)


@pytest.mark.parametrize(
    ("kind", "minimum", "context"), [("rg", 30, "RG nº "), ("crm", 30, "CRM "), ("phone", 20, "")]
)
def test_fake_pool_values_are_distinct_and_match_their_pattern(
    kind: str, minimum: int, context: str
) -> None:
    values = fake_values(kind)  # type: ignore[arg-type]
    keys = [re.sub(r"\D", "", v) for v in values]
    assert len(values) >= minimum and len(set(keys)) == len(keys)
    for value in values:
        line = f"{context}{value}"
        hits = [m["value"] for p in IDENTIFIER_PATTERNS[kind.upper()] for m in p.finditer(line)]
        assert value in hits, value
    assert set(keys) <= getattr(load_pool(), kind)
