"""The Streamlit UI (`ui` extra). It only talks to the API over HTTP (`NECROHUB_API_URL`).

Run it with `necrohub ui`. The pure helpers need neither Streamlit nor httpx, so unit tests import
them alone; `render` imports both. Streamlit executes this file as a script, hence the guarded
call at the bottom.
"""

import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

DEFAULT_API_URL = "http://localhost:8000"
NOTICE = "Synthetic data, not for clinical or forensic use."


class CitationView(BaseModel):
    doc_id: str
    section: str
    ordinal: int
    text: str


class AnswerView(BaseModel):
    answer: str
    backend: str
    index_version: str
    citations: list[CitationView]


def api_url() -> str:
    return os.environ.get("NECROHUB_API_URL", DEFAULT_API_URL).rstrip("/")


def initial_question(query_params: Mapping[str, str]) -> str:
    """The question from a `?q=` link, so a link can open the page with its answer shown."""
    return query_params.get("q", "").strip()[:1000]  # the API accepts up to 1000 characters


def ask_payload(question: str, k: int) -> dict[str, Any]:
    """The POST /ask body; the API validates the bounds again."""
    return {"question": question.strip(), "k": max(1, min(20, k))}


def answer_view(body: dict[str, Any]) -> AnswerView:
    """The fields the page shows, from the API's Answer JSON."""
    return AnswerView(
        answer=body["answer"],
        backend=body["backend"],
        index_version=body["index_version"],
        citations=[
            CitationView(
                doc_id=c["doc_id"], section=c["section"], ordinal=c["ordinal"], text=c["text"]
            )
            for c in body["citations"]
        ],
    )


def error_message(status: int | None, detail: object = None) -> str:
    """A one-line message for a failed call; status None means the API was unreachable."""
    if status is None:
        return "The API is unreachable. Is it running?"
    if status == 422 and isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return (
        f"The API answered {status}: {str(detail)[:300]}"
        if detail
        else f"The API answered {status}."
    )


def render() -> None:
    import httpx
    import streamlit as st

    base = api_url()
    st.set_page_config(page_title="necrohub", page_icon=None)
    st.title("necrohub")
    st.warning(NOTICE)

    with st.sidebar:
        st.subheader("API")
        st.caption(base)
        try:
            health = httpx.get(f"{base}/healthz", timeout=5)
            st.write(health.json())
        except (httpx.HTTPError, ValueError):
            st.error(error_message(None))

    initial = initial_question(st.query_params)
    question = st.text_input(
        "Question", value=initial, placeholder="Qual a causa da morte no caso c7-00003?"
    )
    k = st.slider("Chunks (k)", 1, 20, 4)
    # the button is created first so it always renders; a ?q= link asks without a click
    if st.button("Ask", disabled=not question.strip()) or (initial and question == initial):
        try:
            response = httpx.post(f"{base}/ask", json=ask_payload(question, k), timeout=120)
        except httpx.HTTPError:
            st.error(error_message(None))
            return
        if response.status_code != 200:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            st.error(error_message(response.status_code, detail))
            return
        view = answer_view(response.json())
        st.subheader("Answer")
        st.write(view.answer)
        st.caption(f"backend: {view.backend} | index_version: {view.index_version}")
        st.subheader("Cited chunks")
        for c in view.citations:
            with st.expander(f"{c.doc_id} | {c.section} | {c.ordinal}"):
                st.text(c.text)


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
    except ImportError:
        return False
    return bool(exists())


if _in_streamlit():
    render()
