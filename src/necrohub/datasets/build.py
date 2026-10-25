"""Task records from one processed document and its gold record. Pure functions."""

from collections.abc import Callable, Iterable

from necrohub.contracts.datasets import (
    CaseExtraction,
    ExtractionRecord,
    NerRecord,
    QaRecord,
    SectionRecord,
)
from necrohub.contracts.documents import ProcessedDocument
from necrohub.contracts.gold import CaseFacts, GoldRecord
from necrohub.contracts.spans import SectionLabel, Span
from necrohub.preprocess.align import project

_L = SectionLabel
_NO_HEADER = frozenset({_L.PREAMBLE, _L.SIGNATURE})


def project_gold(
    gold: GoldRecord, source: str, doc: ProcessedDocument
) -> tuple[list[Span], set[SectionLabel], int]:
    """Gold spans projected onto the processed text, the source sections that lost a span to a
    projection reject (their NER records would miss a label), and the number of rejects."""
    projected, rejected = project(source, doc.text, gold.entities)
    lost = {s.label for r in rejected for s in gold.sections if s.start <= r.start < s.end}
    return projected, lost, len(rejected)


def ner_records(
    doc: ProcessedDocument, case_id: str, spans: list[Span], skip: set[SectionLabel]
) -> list[NerRecord]:
    """One record per section (except `skip`), with the spans inside it shifted to its text."""
    records = []
    for n, s in enumerate(doc.sections):
        if s.label in skip:
            continue
        inside = [
            x.model_copy(update={"start": x.start - s.start, "end": x.end - s.start})
            for x in spans
            if s.start <= x.start and x.end <= s.end
        ]
        records.append(
            NerRecord(
                id=f"{doc.doc_id}:{s.label}:{n}",
                doc_id=doc.doc_id,
                case_id=case_id,
                section=s.label,
                text=doc.text[s.start : s.end],
                spans=inside,
            )
        )
    return records


def section_records(doc: ProcessedDocument) -> list[SectionRecord]:
    """Section bodies without their header line; empty bodies are skipped."""
    records = []
    for n, s in enumerate(doc.sections):
        text = doc.text[s.start : s.end]
        if s.label not in _NO_HEADER:
            text = text.partition("\n")[2]
        if text := text.strip():
            records.append(
                SectionRecord(
                    id=f"{doc.doc_id}:{s.label}:{n}", doc_id=doc.doc_id, text=text, label=s.label
                )
            )
    return records


def extraction_record(doc: ProcessedDocument, case: CaseFacts) -> ExtractionRecord:
    c = case.conclusion
    target = CaseExtraction(
        scenario=case.scenario,
        cause_of_death=c.cause_of_death,
        mechanism=c.mechanism,
        manner_of_death=c.manner_of_death,
        time_since_death_h=c.time_since_death_h,
        injuries=case.external.injuries,
        organ_findings=case.internal,
    )
    return ExtractionRecord(
        id=f"{doc.doc_id}:extraction",
        doc_id=doc.doc_id,
        input_text=doc.redacted_text,
        target=target,
    )


def _organs(c: CaseFacts) -> str:
    return ", ".join(f.organ for f in c.internal if f.abnormal) or "Nenhum"


def _injuries(c: CaseFacts) -> str:
    return "; ".join(f"{i.kind} em {i.region}" for i in c.external.injuries) or "Nenhuma"


# (key, question, section where the answer is stated, answer from the case facts)
QUESTIONS: tuple[tuple[str, str, SectionLabel, Callable[[CaseFacts], str]], ...] = (
    ("cause", "Qual foi a causa da morte?", _L.CONCLUSION, lambda c: c.conclusion.cause_of_death),
    (
        "mechanism",
        "Qual foi o mecanismo da morte?",
        _L.CONCLUSION,
        lambda c: c.conclusion.mechanism,
    ),
    (
        "manner",
        "Qual foi a natureza da morte?",
        _L.CONCLUSION,
        lambda c: c.conclusion.manner_of_death,
    ),
    (
        "time",
        "Qual o intervalo post mortem estimado?",
        _L.CONCLUSION,
        lambda c: "{} a {} horas".format(*c.conclusion.time_since_death_h),
    ),
    ("organs", "Quais órgãos apresentaram alterações no exame interno?", _L.INTERNAL_EXAM, _organs),
    ("injuries", "Quais lesões foram descritas no exame externo?", _L.EXTERNAL_EXAM, _injuries),
)


def qa_records(doc: ProcessedDocument, case: CaseFacts) -> list[QaRecord]:
    return [
        QaRecord(
            id=f"{doc.doc_id}:qa:{key}",
            doc_id=doc.doc_id,
            question=q,
            answer=get(case),
            section=sec,
        )
        for key, q, sec, get in QUESTIONS
    ]


def pii_values(gold: GoldRecord) -> set[str]:
    return {s.text for s in gold.entities if s.label.is_pii}


def pii_leaks(texts: Iterable[str], pii: set[str]) -> int:
    """How many of the texts contain a gold PII value (counts only, never the value)."""
    return sum(1 for t in texts if any(v in t for v in pii))
