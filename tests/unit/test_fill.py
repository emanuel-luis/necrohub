import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from necrohub.contracts.gold import Draft, Realized, Reject
from necrohub.contracts.spans import EntityLabel, SectionLabel
from necrohub.generate.fill import fill
from necrohub.generate.sampler import sample_case

CASE = sample_case(7, 0)
# the smallest draft fill accepts: required slots and labels, sections that split_sections sees
BODY = (
    "Laudo\n"
    "<<SEC:IDENTIFICATION>>1. Identificação\n"
    "Nome: {{DECEDENT_NAME}}, CRM {{CRM}}.\n"
    "<<SEC:CONCLUSION>>2. Conclusão\n"
    "Causa: [[CAUSE_OF_DEATH|asfixia]]. Natureza: [[MANNER_OF_DEATH|natural]].\n"
    "<<SEC:SIGNATURE>>{{CITY_UF}}, {{EXAM_DATE_LONG}}.\n"
)


def _fill(body: str = BODY) -> Realized:
    result = fill(Draft(template_id="t0", body=body), CASE)
    assert isinstance(result, Realized), result
    return result


def test_slots_markers_and_sections_give_exact_spans() -> None:
    r = _fill()
    assert "{{" not in r.text and "[[" not in r.text and "<<" not in r.text
    for span in r.entities:
        assert r.text[span.start : span.end] == span.text
    by_label = {s.label: s.text for s in r.entities}
    assert by_label[EntityLabel.PER] == CASE.decedent.name
    assert by_label[EntityLabel.CRM] == CASE.examiner.crm
    assert by_label[EntityLabel.CAUSE_OF_DEATH] == "asfixia"
    assert [s.label for s in r.sections] == [
        SectionLabel.PREAMBLE,
        SectionLabel.IDENTIFICATION,
        SectionLabel.CONCLUSION,
        SectionLabel.SIGNATURE,
    ]
    assert r.text[r.sections[1].start :].startswith("1. Identificação\n")
    assert r.sections[-1].end == len(r.text)


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("{{CRM}}", "{{NOPE}}", "unknown_slot"),
        ("[[CAUSE_OF_DEATH|asfixia]]", "[[CAUSE_OF_DEATH|asfixia]] [[NOPE|x]]", "unknown_label"),
        ("[[CAUSE_OF_DEATH|asfixia]]", "[[CAUSE_OF_DEATH|asfixia]] [[PER|x]]", "unknown_label"),
        ("<<SEC:CONCLUSION>>", "<<SEC:NOPE>>", "unknown_section"),
        ("{{CRM}}", "{{CRM}", "malformed_marker"),
        ("[[CAUSE_OF_DEATH|asfixia]]", "[[CAUSE_OF_DEATH| asfixia]]", "malformed_marker"),
        ("[[CAUSE_OF_DEATH|asfixia]]", "[[CAUSE_OF_DEATH|]]", "malformed_marker"),
        ("Laudo\n", "Laudo <<SEC:HISTORY>>\n", "section_not_at_line_start"),
        ("{{DECEDENT_NAME}}", "fulano", "missing_required"),
        ("[[MANNER_OF_DEATH|natural]]", "natural", "missing_required"),
        ("2. Conclusão", "2. Considerações", "section_mismatch"),
    ],
)
def test_every_reject_reason(old: str, new: str, reason: str) -> None:
    assert old in BODY
    result = fill(Draft(template_id="t0", body=BODY.replace(old, new, 1)), CASE)
    assert isinstance(result, Reject)
    assert (result.reason, result.case_id, result.template_id) == (reason, CASE.case_id, "t0")


_VALUE = st.text(
    alphabet=st.characters(categories=["L", "N", "Zs", "P"], exclude_characters="{}[]<>|\n\r"),
    min_size=1,
    max_size=40,
)


@settings(max_examples=60, deadline=None)
@given(name=_VALUE, address=_VALUE)
def test_slot_values_round_trip(name: str, address: str) -> None:
    decedent = CASE.decedent.model_copy(update={"name": name, "address": address})
    case = CASE.model_copy(update={"decedent": decedent})
    body = BODY.replace("Nome: {{DECEDENT_NAME}}", "Nome: {{DECEDENT_NAME}} em {{ADDRESS}}")
    result = fill(Draft(template_id="t0", body=body), case)
    assert isinstance(result, Realized)
    spans = {s.label: s for s in result.entities}
    assert spans[EntityLabel.PER].text == name
    assert spans[EntityLabel.ADDRESS].text == address
    for span in result.entities:
        assert result.text[span.start : span.end] == span.text
