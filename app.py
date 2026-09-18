"""FastAPI entry point for the production legal RAG assistant."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from runtime.service import LegalRAGService
from runtime.settings import Settings


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class Source(BaseModel):
    text: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    classification: str
    supported: bool
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.startup_error = "Models are loading. Check /api/health for readiness."

    async def initialize() -> None:
        try:
            # Model construction is blocking, so keep the HTTP server available during it.
            app.state.rag = await asyncio.to_thread(LegalRAGService, Settings())
            app.state.startup_error = None
        except Exception as error:
            app.state.startup_error = str(error)

    task = asyncio.create_task(initialize())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Legal AI Assistant API", lifespan=lifespan)
settings = Settings()
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/api/health")
def health() -> dict:
    ready = app.state.rag is not None
    return {"status": "ok" if ready else "degraded", "ready": ready,
            "detail": app.state.startup_error}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    if app.state.rag is None:
        detail = app.state.startup_error or "The RAG service is still starting."
        raise HTTPException(status_code=503, detail=f"Backend is not ready: {detail}")
    try:
        return app.state.rag.ask(request.question)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from error
