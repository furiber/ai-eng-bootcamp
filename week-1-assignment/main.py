"""Week 1 assignment — a typed /ask endpoint with structured model output.

Run:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import os
import time

from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from env_setup import load_env

# Reads the repo-root .env; the key never lives in this folder.
load_env()

app = FastAPI(title="Week 1 Assignment — Ask API", version="0.1.0")

# openai>=3 raises at construction when no key can be found — including for an
# explicit api_key=None — so a placeholder is passed when the real key is absent.
# That keeps the app booting (and /health passing) without a key; /ask checks
# API_KEY itself and returns a clear 500 rather than crashing at import time.
API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=API_KEY or "missing-key", timeout=60.0, max_retries=2)

DEFAULT_MODEL = "gpt-4o-mini"


class Answer(BaseModel):
    """Structured model output — this is what makes the endpoint a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["What is RAG?"])
    model: str = Field(default=DEFAULT_MODEL, examples=[DEFAULT_MODEL])


class AskResponse(BaseModel):
    answer: Answer
    model: str
    tokens_used: int
    latency_ms: int


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness check — answers without touching OpenAI, so it costs nothing."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(request: AskRequest) -> AskResponse:
    """Send one question to the model and return a validated, typed answer."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    start = time.perf_counter()
    try:
        completion = await client.chat.completions.parse(
            model=request.model,
            messages=[{"role": "user", "content": request.question}],
            response_format=Answer,
        )
    except Exception as exc:  # ponytail: one upstream mapping; split if a caller needs detail
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

    message = completion.choices[0].message
    if message.parsed is None:
        # Refusals and unparseable output must not reach the caller as a 200.
        raise HTTPException(status_code=502, detail=message.refusal or "No parseable output")

    try:
        answer = Answer.model_validate(message.parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail=f"Schema validation failed: {exc}") from exc

    return AskResponse(
        answer=answer,
        model=completion.model,
        tokens_used=completion.usage.total_tokens if completion.usage else 0,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
