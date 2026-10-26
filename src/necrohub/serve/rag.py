"""Retrieval-augmented answers over the serving index: extractive by default, LLM optional."""

import re
from importlib.resources import files

import numpy as np

from necrohub.contracts.serving import Answer, Chunk, RetrievedChunk
from necrohub.embed.base import Embedder
from necrohub.index.store import VectorStore
from necrohub.llm.base import ChatLLM, ChatMessage, LlmError, get_llm
from necrohub.settings import Settings

__all__ = ["LlmError", "answer", "case_id_in", "llm_from_settings"]

CASE_ID = re.compile(r"\bc\d+-\d{5}\b")
_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*#>]+\s*)+")
_WORD = re.compile(r"\w")
NOT_FOUND = "Não encontrei essa informação nos trechos recuperados."


def case_id_in(question: str) -> str | None:
    match = CASE_ID.search(question)
    return match.group() if match else None


def llm_from_settings(settings: Settings) -> ChatLLM | None:
    """None for the extractive backend; otherwise the configured Ollama or Gemini client."""
    if settings.llm == "extractive":
        return None
    return get_llm(settings.llm, settings.llm_model)


def _sentences(chunk: Chunk) -> list[str]:
    out = []
    for line in chunk.text.splitlines():
        for part in _SENTENCE_END.split(line):
            part = _BULLET.sub("", part).strip()
            if _WORD.search(part):
                out.append(part)
    return out


def _extractive(question: str, hits: list[RetrievedChunk], embedder: Embedder) -> tuple[str, Chunk]:
    """The sentence most similar to the question; ties go to the higher-ranked chunk."""
    candidates = [(s, h.chunk) for h in hits for s in _sentences(h.chunk)]
    if not candidates:
        return hits[0].chunk.text.strip(), hits[0].chunk
    vectors = embedder.embed([question, *(s for s, _ in candidates)])
    best = int(np.argmax(vectors[1:] @ vectors[0]))  # first maximum: stable ranking order
    return candidates[best]


def _prompt(question: str, hits: list[RetrievedChunk]) -> list[ChatMessage]:
    system = files("necrohub.serve").joinpath("prompts/rag_system.txt").read_text("utf-8")
    context = "\n\n".join(
        f"[{i}] ({h.chunk.section}) {h.chunk.text}" for i, h in enumerate(hits, 1)
    )
    user = f"Contexto:\n{context}\n\nPergunta: {question}"
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def answer(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    index_version: str,
    llm: ChatLLM | None,
    k: int,
) -> Answer:
    """Answer from the k best chunks, filtered to one document when the question names a case.

    With llm=None the answer is the best-matching sentence, cited to its chunk. With an LLM the
    answer is generated from the numbered chunks and cites the ones it names ([n]), else all of
    them. A backend failure raises LlmError.
    """
    query = embedder.embed([question])[0]
    hits = store.search(index_version, query, k, doc_id=case_id_in(question))
    if not hits:
        return Answer(
            question=question,
            answer=NOT_FOUND,
            citations=[],
            backend="extractive" if llm is None else llm.name,
            index_version=index_version,
        )
    if llm is None:
        text, source = _extractive(question, hits, embedder)
        return Answer(
            question=question,
            answer=text,
            citations=[source],
            backend="extractive",
            index_version=index_version,
        )
    reply = llm.complete(_prompt(question, hits), temperature=0.2).strip()
    cited = {int(n) for n in _CITATION.findall(reply) if 1 <= int(n) <= len(hits)}
    citations = [h.chunk for i, h in enumerate(hits, 1) if i in cited] or [h.chunk for h in hits]
    return Answer(
        question=question,
        answer=reply,
        citations=citations,
        backend=llm.name,
        index_version=index_version,
    )
