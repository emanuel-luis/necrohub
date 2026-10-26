"""The FastAPI app: GET /healthz and POST /ask over a vector store (`serve` extra)."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from necrohub.contracts.serving import Answer
from necrohub.embed.base import Embedder
from necrohub.embed.factory import build_embedder
from necrohub.llm.base import ChatLLM, LlmError
from necrohub.params import load_params
from necrohub.serve.rag import answer, llm_from_settings
from necrohub.serve.runtime import open_store
from necrohub.settings import Settings


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=20)


def create_app(
    settings: Settings, *, embedder: Embedder | None = None, llm: ChatLLM | None = None
) -> FastAPI:
    """Build the app. `embedder` and `llm` replace the configured ones (tests)."""
    embedder = embedder or build_embedder(load_params(settings.params_path).embed)
    llm = llm or llm_from_settings(settings)
    store, version, _ = open_store(settings, embedder.id)
    app = FastAPI(title="necrohub", version="1")

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        try:
            n = store.count(version)
        except Exception:  # noqa: BLE001 - any store failure means not ready
            n = 0
        body = {
            "status": "ok" if n > 0 else "unavailable",
            "index_version": version,
            "n_chunks": n,
            "embedder_id": embedder.id,
        }
        return JSONResponse(body, status_code=200 if n > 0 else 503)

    @app.post("/ask")
    def ask(request: AskRequest) -> Answer:
        try:
            return answer(
                request.question,
                embedder=embedder,
                store=store,
                index_version=version,
                llm=llm,
                k=request.k,
            )
        except LlmError as exc:
            raise HTTPException(status_code=502, detail=f"LLM backend failed: {exc}"[:300]) from exc
        except ValueError as exc:  # a question with no word to embed
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
