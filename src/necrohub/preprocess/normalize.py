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


_MAX_PASSES = 4


def normalize(text: str, steps: Sequence[str] = DEFAULT_STEPS) -> str:
    """Apply the named steps in order until the text stops changing. Accents, case and newlines
    are kept.

    One pass is not always a fixed point: ftfy judges mojibake from the surrounding characters,
    and a later step (nbsp, trailing_ws) can change them, so the next pass may fix more.
    """
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown normalize step(s): {', '.join(unknown)}")
    for _ in range(_MAX_PASSES):
        before = text
        for step in steps:
            text = STEPS[step](text)
        if text == before:
            break
    return text
