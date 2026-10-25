import re

from necrohub.contracts.spans import EntityLabel, PredictedSpan
from necrohub.ids import IDENTIFIER_PATTERNS

_MONTHS = "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
_UP = "A-ZÁÀÂÃÉÊÍÓÔÕÚÇ"
_LOW = "a-záàâãéêíóôõúç"
# capitalized tokens on one line, with the particles da/das/de/do/dos between them
_NAME = rf"[{_UP}][{_LOW}'’-]+(?:[ \t]+(?:(?:da|das|de|do|dos)[ \t]+)?[{_UP}][{_LOW}'’-]+)*"
_TITLE = r"(?:Dra?\.[ \t]+)?"
_ASSISTANT = r"(?:T[ée]cnic[oa] de Necropsia[ \t]+)?"

_OTHER: dict[str, tuple[re.Pattern[str], ...]] = {
    "DOC_NUMBER": (
        re.compile(
            r"\b(?:Laudo|Relat[óo]rio|B\.O\.?|boletim de ocorr[êe]ncia|inqu[ée]rito(?: policial)?)"
            r"(?:\s*n[ºo°]\.?|:)?[ \t]*(?P<value>\d[\w./-]*\w)",
            re.IGNORECASE,
        ),
    ),
    "DATE": (
        re.compile(r"\b(?P<value>\d{2}/\d{2}/\d{4})\b"),
        re.compile(rf"\b(?P<value>\d{{1,2}} de (?:{_MONTHS}) de \d{{4}})\b", re.IGNORECASE),
    ),
    # a name counts only after a strong cue, so capitalized common words are left alone
    "PER": tuple(
        re.compile(rf"{cue}(?P<value>(?!(?:Dra?|T[ée]cnic[oa])\b){_NAME})")
        for cue in (
            r"\bDra?\.[ \t]+",
            rf"\bPerito[^\n:]{{0,40}}:[ \t]*{_TITLE}",
            rf"\bAuxiliar[^\n:]{{0,40}}:[ \t]*{_ASSISTANT}",
            r"\bidentificad[oa] como[ \t]+",
            r"\bCorpo de[ \t]+(?!(?i:Delito|Bombeiros)\b)",
            r"\bT[ée]cnic[oa] de Necropsia[ \t]+",
        )
    ),
    # street, number, then an optional neighbourhood after a comma or hyphen
    "ADDRESS": (
        re.compile(
            r"\b(?P<value>(?:Rua|Avenida|Av\.|Travessa|Tv\.|Alameda|Estrada)[ \t]+[^,\n]{1,60}?,"
            rf"[ \t]*n[ºo°]\.?[ \t]*\d+[A-Za-z]?(?:[ \t]*[,-][ \t]*(?:Bairro[ \t]+)?{_NAME})?)"
        ),
    ),
}

_PATTERNS = {**IDENTIFIER_PATTERNS, **_OTHER}
_LABEL = {"CEP": EntityLabel.ADDRESS}  # a postal code is part of an address


class RegexDeidentifier:
    name = "regex"

    def detect(self, text: str) -> list[PredictedSpan]:
        """Every match of every pattern, overlaps included (merge_spans resolves them)."""
        return [
            PredictedSpan(
                start=m.start("value"),
                end=m.end("value"),
                label=_LABEL.get(label) or EntityLabel(label),
                text=m["value"],
                score=1.0,
                source="regex",
            )
            for label, patterns in _PATTERNS.items()
            for pattern in patterns
            for m in pattern.finditer(text)
        ]
