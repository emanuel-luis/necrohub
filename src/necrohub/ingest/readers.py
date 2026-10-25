"""Text of a raw report file: UTF-8 txt, or PDF text extracted with pypdf."""

import io


def txt_text(data: bytes) -> str:
    """UTF-8 text; undecodable bytes become U+FFFD so the ingest check can count them."""
    return data.decode("utf-8", errors="replace")


def pdf_text(data: bytes) -> str:
    """The text of every page, pages joined by a newline. Raises on a file pypdf cannot parse."""
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)


def extractor(source_format: str) -> str:
    """name@version of what turned the file's bytes into text."""
    if source_format == "pdf":
        import pypdf

        return f"pypdf@{pypdf.__version__}"
    from necrohub import __version__

    return f"necrohub-utf8@{__version__}"
