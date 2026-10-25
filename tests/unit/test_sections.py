from collections.abc import Callable
from itertools import pairwise

from necrohub.contracts.spans import SectionLabel as L
from necrohub.preprocess.sections import split_sections


def labels(text: str) -> list[L]:
    return [s.label for s in split_sections(text)]


def test_sections_golden(report: tuple[str, str], golden: Callable[[str, object], None]) -> None:
    name, text = report
    golden(
        f"sections/{name}.json",
        [
            {
                "label": s.label.value,
                "start": s.start,
                "end": s.end,
                "first_line": text[s.start : s.end].split("\n")[0],
            }
            for s in split_sections(text)
        ],
    )


def test_cover_without_overlap(report: tuple[str, str]) -> None:
    _, text = report
    sections = split_sections(text)
    assert sections[0].start == 0 and sections[-1].end == len(text)
    assert all(a.end == b.start for a, b in pairwise(sections))
    assert sections[0].label == L.PREAMBLE and sections[-1].label == L.SIGNATURE


def test_numbered_and_unnumbered_headers() -> None:
    text = "Titulo\n1. Identificação\na\nHistórico e Circunstâncias\nb\n5.\nc\nConclusão:\nd\n"
    assert labels(text) == [L.PREAMBLE, L.IDENTIFICATION, L.HISTORY, L.CONCLUSION]


def test_accent_insensitive_headers() -> None:
    text = "2. HISTORICO\n3. exame EXTERNO\n4. Exame Interno\n5. DIAGNOSTICO E CONCLUSAO\n"
    assert labels(text) == [L.HISTORY, L.EXTERNAL_EXAM, L.INTERNAL_EXAM, L.CONCLUSION]


def test_header_variants_and_separators() -> None:
    text = "Dados Circunstanciais\n---\nExames Complementares Sugeridos\nx\n---\n"
    sections = split_sections(text)
    assert [s.label for s in sections] == [L.HISTORY, L.COMPLEMENTARY_EXAMS]
    assert text[sections[1].start : sections[1].end].endswith("---\n")


def test_subitems_and_prose_are_not_headers() -> None:
    assert labels("5. Conclusão\n5.1. Causa da morte: x\nA conclusão é clara.\n") == [L.CONCLUSION]


def test_no_headers_is_one_preamble_and_empty_text_has_none() -> None:
    assert labels("só texto\n") == [L.PREAMBLE]
    assert split_sections("") == []
