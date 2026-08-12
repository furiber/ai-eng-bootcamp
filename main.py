"""FastAPI service exposing a single /ask endpoint backed by the OpenAI API.

The static frontend in ./static is served from the same container, so there is
no CORS configuration and no separate deployment to keep in sync. Interactive
API docs (Swagger UI) come from FastAPI itself at /docs.
"""

import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

app = FastAPI(
    title="Ask API",
    description="Ask a question, get an answer from an OpenAI model.",
    version="0.1.0",
)

# api_key is read lazily so the app can boot (and /health can pass) without a
# key present; a missing key surfaces as a 500 on /ask instead of a crash
# at import time, which would make container start-up failures hard to read.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0, max_retries=2)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["Why is the sky blue?"])
    model: str = Field(default=DEFAULT_MODEL, examples=["gpt-4o-mini"])


class AskResponse(BaseModel):
    answer: str
    model: str


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe used by Render's health check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(request: AskRequest) -> AskResponse:
    """Send a question to an OpenAI model and return its answer."""
    if not client.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    try:
        response = await client.responses.create(
            model=request.model,
            input=request.question,
        )
    except openai.APIStatusError as exc:
        # Pass client errors (bad model name, quota) through as 4xx; treat
        # everything else from upstream as a bad gateway.
        status = exc.status_code if 400 <= exc.status_code < 500 else 502
        raise HTTPException(status_code=status, detail=f"OpenAI error: {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Could not reach the OpenAI API") from exc

    return AskResponse(answer=response.output_text, model=response.model)


# Mounted last so /health, /ask, /docs and /openapi.json win over the
# catch-all static route. html=True serves static/index.html at "/".
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
