import re
import unicodedata

from necrohub.contracts.spans import SectionLabel, SectionSpan

_L = SectionLabel
# Matched against a header line that is lowercased, accent-stripped and without its numbering.
_HEADERS: tuple[tuple[SectionLabel, re.Pattern[str]], ...] = tuple(
    (label, re.compile(rf"(?:{pattern})\s*:?"))
    for label, pattern in (
        (_L.IDENTIFICATION, r"identificacao"),
        (_L.HISTORY, r"historico( e circunstancias)?|dados circunstanciais|circunstancias"),
        (_L.EXTERNAL_EXAM, r"exame externo"),
        (_L.INTERNAL_EXAM, r"exame interno"),
        (_L.COMPLEMENTARY_EXAMS, r"exames complementares( sugeridos)?"),
        (_L.CONCLUSION, r"(diagnostico e )?conclusao"),
    )
)
_NUMBERING = re.compile(r"[#*\s]*(?:\d+(?:\.\d+)*[.)]?\s*)?")
_SIGNATURE = re.compile(r"[a-z' .-]+/[a-z]{2},\s*\d{1,2} de [a-z]+ de \d{4}\.?")


def _fold(line: str) -> str:
    decomposed = unicodedata.normalize("NFD", line.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").strip()


def _header_label(line: str) -> SectionLabel | None:
    folded = _fold(line)
    rest = _NUMBERING.sub("", folded, count=1).rstrip("#* ")
    for label, pattern in _HEADERS:
        if pattern.fullmatch(rest):
            return label
    return None


def split_sections(text: str) -> list[SectionSpan]:
    """Partition the text into ordered sections. A section runs from its header line to the next."""
    marks: list[tuple[int, SectionLabel]] = []
    for line in re.finditer(r"^.*$", text, flags=re.MULTILINE):
        label = _header_label(line.group())
        if label is not None:
            marks.append((line.start(), label))
    if marks:
        # the signature is the last "<City>/<UF>, <dd> de <month> de <yyyy>." line below a header
        for line in re.finditer(r"^.*$", text[marks[-1][0] :], flags=re.MULTILINE):
            if _SIGNATURE.fullmatch(_fold(line.group())):
                marks.append((marks[-1][0] + line.start(), SectionLabel.SIGNATURE))
                break
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, SectionLabel.PREAMBLE))
    ends = [m[0] for m in marks[1:]] + [len(text)]
    return [
        SectionSpan(label=label, start=start, end=end)
        for (start, label), end in zip(marks, ends, strict=True)
        if end > start
    ]
