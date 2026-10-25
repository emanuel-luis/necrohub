"""Render report text to PDF with fpdf2 core fonts (latin-1). The txt stays the span source."""

import random
from datetime import UTC, datetime

# typographic characters an LLM likes and latin-1 lacks
_TYPOGRAPHIC = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "•": "*",
    }
)
_FIXED_DATE = datetime(2000, 1, 1, tzinfo=UTC)  # no timestamps: same text, same bytes


def transliterate(text: str) -> tuple[str, int]:
    """Replace typographic characters with latin-1 ones; returns the text and how many changed."""
    n = sum(1 for c in text if ord(c) in _TYPOGRAPHIC)
    return (text.translate(_TYPOGRAPHIC), n) if n else (text, 0)


def wants_pdf(seed: int, case_id: str, fraction: float) -> bool:
    """Deterministic per-case choice of the cases that also get a PDF."""
    return random.Random(f"{seed}:{case_id}:pdf").random() < fraction


def render_pdf(text: str) -> tuple[bytes, int]:
    """PDF bytes of the text and the number of characters transliterated or replaced by '?'."""
    from fpdf import FPDF

    text, n = transliterate(text)
    encoded = text.encode("latin-1", errors="replace").decode("latin-1")
    n += sum(1 for a, b in zip(text, encoded, strict=True) if a != b)
    pdf = FPDF(format="A4")
    pdf.set_creation_date(_FIXED_DATE)
    pdf.set_producer("necrohub")
    pdf.set_creator("necrohub")
    pdf.set_margins(20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, encoded)
    return bytes(pdf.output()), n
