import re
import unicodedata
from collections.abc import Callable, Sequence

import ftfy

_NBSP = str.maketrans({" ": " ", " ": " "})

STEPS: dict[str, Callable[[str], str]] = {
    "fix_text": ftfy.fix_text,
    "nfc": lambda t: unicodedata.normalize("NFC", t),
    "newlines": lambda t: t.replace("\r\n", "\n").replace("\r", "\n"),
    "nbsp": lambda t: t.translate(_NBSP),
    "trailing_ws": lambda t: re.sub(r"[ \t]+$", "", t, flags=re.MULTILINE),
    "blank_lines": lambda t: re.sub(r"\n{3,}", "\n\n", t),
}

DEFAULT_STEPS: tuple[str, ...] = tuple(STEPS)


def normalize(text: str, steps: Sequence[str] = DEFAULT_STEPS) -> str:
    """Apply the named steps in order. Accents, case and newlines are kept."""
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown normalize step(s): {', '.join(unknown)}")
    for step in steps:
        text = STEPS[step](text)
    return text
