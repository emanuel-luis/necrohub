"""Turn a realizer draft into report text with exact gold spans. Pure; shared by every realizer.

Draft syntax:
- `{{SLOT}}`: a CaseFacts value from SLOTS, emitted with a PII span of the slot's label.
- `[[LABEL|surface]]`: the surface text, emitted with a clinical span (non-PII labels only).
- `<<SEC:LABEL>>` at the start of a line: a section starts at that line.
"""

import re
from collections.abc import Callable

from necrohub.contracts.gold import CaseFacts, Draft, Realized, Reject, RejectReason
from necrohub.contracts.spans import EntityLabel, SectionLabel, SectionSpan, Span
from necrohub.generate.vocab import MONTHS
from necrohub.preprocess.sections import split_sections

_E = EntityLabel


def _long_date(c: CaseFacts) -> str:
    d = c.report.exam_date
    return f"{d.day} de {MONTHS[d.month - 1]} de {d.year}"


SLOTS: dict[str, tuple[EntityLabel, Callable[[CaseFacts], str]]] = {
    "DECEDENT_NAME": (_E.PER, lambda c: c.decedent.name),
    "MOTHER_NAME": (_E.PER, lambda c: c.decedent.mother_name),
    "EXAMINER_NAME": (_E.PER, lambda c: c.examiner.name),
    "ASSISTANT_NAME": (_E.PER, lambda c: c.assistant_name),
    "CPF": (_E.CPF, lambda c: c.decedent.cpf),
    "RG": (_E.RG, lambda c: c.decedent.rg),
    "CRM": (_E.CRM, lambda c: c.examiner.crm),
    "PHONE": (_E.PHONE, lambda c: c.institute.phone),
    "ADDRESS": (_E.ADDRESS, lambda c: c.decedent.address),
    "BIRTHPLACE": (_E.LOC, lambda c: c.decedent.birthplace),
    "CITY": (_E.LOC, lambda c: c.institute.city),
    "CITY_UF": (_E.LOC, lambda c: f"{c.institute.city}/{c.institute.uf}"),
    "EXAM_DATE": (_E.DATE, lambda c: c.report.exam_date.strftime("%d/%m/%Y")),
    "EXAM_DATE_LONG": (_E.DATE, _long_date),
    "DEATH_DATE": (_E.DATE, lambda c: c.circumstances.estimated_death_at.strftime("%d/%m/%Y")),
    "REPORT_NUMBER": (_E.DOC_NUMBER, lambda c: c.report.number),
    "POLICE_REPORT_NUMBER": (_E.DOC_NUMBER, lambda c: c.report.police_report_number),
}
REQUIRED_SLOTS = frozenset({"DECEDENT_NAME", "CRM"})
REQUIRED_LABELS = frozenset({_E.CAUSE_OF_DEATH, _E.MANNER_OF_DEATH})

_TOKEN = re.compile(
    r"\{\{(?P<slot>[^{}\n]*)\}\}"
    r"|\[\[(?P<label>[^\[\]|\n]*)\|(?P<surface>[^\[\]|\n]*)\]\]"
    r"|<<SEC:(?P<section>[^<>\n]*)>>"
)
_LEFTOVER = re.compile(r"\{\{|\}\}|\[\[|\]\]|<<|>>")


class _Rejected(Exception):
    def __init__(self, reason: RejectReason, detail: str) -> None:
        self.reason, self.detail = reason, detail


def _literal(segment: str) -> str:
    if m := _LEFTOVER.search(segment):
        raise _Rejected("malformed_marker", f"stray {m.group()!r}")
    return segment


def _render(draft: Draft, case: CaseFacts) -> Realized:
    parts: list[str] = []
    pos = 0  # offset in the output text
    entities: list[Span] = []
    marks: list[tuple[int, SectionLabel]] = []
    slots_used: set[str] = set()
    last = 0
    for m in _TOKEN.finditer(draft.body):
        literal = _literal(draft.body[last : m.start()])
        parts.append(literal)
        pos += len(literal)
        last = m.end()
        if (slot := m["slot"]) is not None:
            if slot not in SLOTS:
                raise _Rejected("unknown_slot", slot)
            label, get = SLOTS[slot]
            value = get(case)
            entities.append(Span(start=pos, end=pos + len(value), label=label, text=value))
            slots_used.add(slot)
        elif (name := m["label"]) is not None:
            value = m["surface"]
            if name not in EntityLabel.__members__ or EntityLabel(name).is_pii:
                raise _Rejected("unknown_label", name)
            if not value or value != value.strip() or _LEFTOVER.search(value):
                raise _Rejected("malformed_marker", f"surface of {name}")
            entities.append(
                Span(start=pos, end=pos + len(value), label=EntityLabel(name), text=value)
            )
        else:
            section = m["section"]
            if section not in SectionLabel.__members__:
                raise _Rejected("unknown_section", section)
            if m.start() > 0 and draft.body[m.start() - 1] != "\n":
                raise _Rejected("section_not_at_line_start", section)
            marks.append((pos, SectionLabel(section)))
            value = ""
        parts.append(value)
        pos += len(value)
    parts.append(_literal(draft.body[last:]))
    text = "".join(parts)

    missing = sorted(REQUIRED_SLOTS - slots_used) + sorted(
        REQUIRED_LABELS - {e.label for e in entities}
    )
    if missing:
        raise _Rejected("missing_required", ", ".join(missing))
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, SectionLabel.PREAMBLE))
    ends = [start for start, _ in marks[1:]] + [len(text)]
    sections = [
        SectionSpan(label=label, start=start, end=end)
        for (start, label), end in zip(marks, ends, strict=True)
        if end > start
    ]
    if sections != split_sections(text):
        raise _Rejected("section_mismatch", "markers disagree with split_sections")
    return Realized(text=text, entities=entities, sections=sections)


def fill(draft: Draft, case: CaseFacts) -> Realized | Reject:
    """Substitute slots, strip markers and return the text with exact spans, or why not."""
    try:
        return _render(draft, case)
    except _Rejected as r:
        return Reject(
            case_id=case.case_id, template_id=draft.template_id, reason=r.reason, detail=r.detail
        )
