"""Week 1 assignment — a typed /ask endpoint with structured model output.

Run:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import APIStatusError, AsyncOpenAI
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

# Overridable per environment so deployments can swap models without a code change.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# USD per 1K tokens, (input, output). List prices as of 2026-08-18 — these are
# hardcoded figures that OpenAI can change at any time, so treat cost_usd as an
# estimate and re-check the pricing page rather than trusting it indefinitely.
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """
    Estimate spend for one call. Returns None for a model with no price on file,
    because a wrong number is worse than an absent one.

    The API reports a dated model id ("gpt-4o-mini-2024-07-18"), so an exact
    lookup is tried first and the longest matching prefix second.
    """
    prices = MODEL_PRICES_PER_1K.get(model)
    if prices is None:
        matches = [name for name in MODEL_PRICES_PER_1K if model.startswith(name)]
        if not matches:
            return None
        prices = MODEL_PRICES_PER_1K[max(matches, key=len)]

    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


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
    # None when the returned model has no price on file — see compute_cost_usd.
    cost_usd: float | None


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
    except APIStatusError as exc:
        # A bad model name comes back as 404 model_not_found, not 400, so both are
        # folded into one "your request was bad" 400. Everything else upstream
        # (auth, quota, provider outage) is our problem, not the caller's: 502.
        status = 400 if exc.status_code in (400, 404) else 502
        raise HTTPException(status_code=status, detail=f"OpenAI error: {exc}") from exc
    except Exception as exc:  # ponytail: one mapping for the rest; split if a caller needs detail
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc

    message = completion.choices[0].message
    if message.parsed is None:
        # Refusals and unparseable output must not reach the caller as a 200.
        raise HTTPException(status_code=502, detail=message.refusal or "No parseable output")

    try:
        answer = Answer.model_validate(message.parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail=f"Schema validation failed: {exc}") from exc

    usage = completion.usage
    cost_usd = compute_cost_usd(
        completion.model,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )

    return AskResponse(
        answer=answer,
        model=completion.model,
        tokens_used=usage.total_tokens if usage else 0,
        latency_ms=int((time.perf_counter() - start) * 1000),
        cost_usd=round(cost_usd, 6) if cost_usd is not None else None,
    )


# Mounted last so /health, /ask, /docs and /openapi.json win over this catch-all.
# html=True serves static/index.html at "/".
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
