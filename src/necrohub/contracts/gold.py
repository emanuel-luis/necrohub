"""Generator contracts: sampled case facts, realizer drafts and the gold record of a report."""

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from necrohub.contracts.spans import SectionSpan, Span

Scenario = Literal[
    "firearm_homicide",
    "sharp_force",
    "traffic",
    "fall",
    "hanging",
    "drowning",
    "cardiac_natural",
    "other_natural",
    "poisoning",
    "undetermined",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Institute(_Model):
    name: str
    city: str
    uf: str
    phone: str


class ReportInfo(_Model):
    number: str
    exam_date: date
    start_time: str
    end_time: str
    police_report_number: str


class Examiner(_Model):
    name: str
    sex: Literal["masculino", "feminino"]
    crm: str
    crm_uf: str


class Decedent(_Model):
    name: str
    sex: Literal["masculino", "feminino"]
    age: int
    skin_color: str
    birthplace: str
    address: str
    rg: str
    cpf: str
    mother_name: str


class Circumstances(_Model):
    place_found: str
    body_position: str
    estimated_death_at: datetime


class Injury(_Model):
    kind: str
    region: str
    side: str | None
    size_cm: float | None
    features: list[str]


class External(_Model):
    height_cm: int
    weight_kg: int
    rigor: str
    livor: str
    decomposition: str
    injuries: list[Injury]


class OrganFinding(_Model):
    organ: str
    finding: str
    abnormal: bool


class Conclusion(_Model):
    cause_of_death: str
    mechanism: str
    manner_of_death: str
    time_since_death_h: tuple[int, int]


class CaseFacts(_Model):
    """Everything a report states, sampled before any text is written."""

    case_id: str
    scenario: Scenario
    institute: Institute
    report: ReportInfo
    examiner: Examiner
    assistant_name: str
    decedent: Decedent
    circumstances: Circumstances
    external: External
    internal: list[OrganFinding]
    complementary_exams: list[str]
    conclusion: Conclusion


class Draft(_Model):
    """Realizer output: report text with {{SLOT}}, [[LABEL|surface]] and <<SEC:LABEL>> markers."""

    template_id: str
    body: str


class Realized(_Model):
    text: str
    entities: list[Span]
    sections: list[SectionSpan]


RejectReason = Literal[
    "unknown_slot",
    "unknown_label",
    "unknown_section",
    "malformed_marker",
    "section_not_at_line_start",
    "missing_required",
    "section_mismatch",
]


class Reject(_Model):
    schema_version: Literal[1] = 1
    case_id: str
    template_id: str
    reason: RejectReason
    detail: str = ""


class GoldFiles(_Model):
    txt: str
    pdf: str | None = None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GoldRecord(_Model):
    schema_version: Literal[1] = 1
    case: CaseFacts
    realizer: str
    template_id: str
    text_sha256: str
    entities: list[Span]
    sections: list[SectionSpan]
    files: GoldFiles

    def verify(self, text: str) -> None:
        """Raise ValueError unless the text has this hash, every span matches it and the
        sections tile it from start to end."""
        if text_sha256(text) != self.text_sha256:
            raise ValueError(f"{self.case.case_id}: text sha256 does not match")
        for span in self.entities:
            span.verify(text)
        end = 0
        for section in self.sections:
            if section.start != end:
                raise ValueError(f"{self.case.case_id}: sections must tile the text")
            end = section.end
        if end != len(text):
            raise ValueError(f"{self.case.case_id}: sections must end at the end of the text")
