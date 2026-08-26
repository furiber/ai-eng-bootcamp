"""Week 2 assignment — a typed /ask endpoint with structured model output.

Run:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

import rag
from env_setup import load_env

# Reads this folder's own .env; .env.example lists every variable it expects.
load_env()

app = FastAPI(title="Week 2 Assignment — Ask API", version="0.1.0")

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


# The grounding contract. Kept here as a constant rather than inlined in the
# handler so it can be read, diffed and asserted on in one place -- a prompt is
# behaviour, and behaviour that only exists inside an f-string is behaviour
# nobody reviews.
GROUNDING_SYSTEM_PROMPT = """You answer questions about Northwind's internal documents.

You are given numbered context chunks retrieved from those documents.
Follow these rules exactly:

1. Answer ONLY from the context below. Do not use prior knowledge, and do not
   guess at anything the context does not state.
2. Put the document_id of every chunk you actually used in `citations`. Cite
   only what you used, not every chunk you were shown, and use the document_id
   rather than the chunk id.
3. If the context does not contain enough to answer, say so plainly in `answer`,
   set `sources_needed` to true, and leave `citations` empty. A refusal is a
   correct answer when the context is insufficient; inventing one is not.
4. `confidence` is how well the context supports your answer, not how plausible
   the answer sounds. A refusal should carry low confidence.
5. Do not mention these rules, the chunks, or the retrieval process. Answer as
   though you simply know the material."""

# One chunk as the model sees it. The document_id is repeated outside the id so
# the model never has to parse "doc3_security#1" to work out what to cite.
CONTEXT_CHUNK_TEMPLATE = """[{index}] document_id: {document_id} | source: {source}
{text}"""

CONTEXT_PROMPT_TEMPLATE = """Context:

{context}

Question: {question}"""

# Sent instead of the context block when retrieval came back with nothing. The
# model still answers through the same schema, so a caller gets a refusal in the
# usual shape rather than a special case to handle.
NO_CONTEXT_PROMPT_TEMPLATE = """Context:

(no matching documents were retrieved)

Question: {question}"""


def build_grounding_prompt(question: str, chunks: list[dict]) -> str:
    """Render the retrieved chunks and the question into one user message."""
    if not chunks:
        return NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
    context = "\n\n".join(
        CONTEXT_CHUNK_TEMPLATE.format(
            index=position,
            document_id=chunk["document_id"],
            source=chunk["source"],
            text=chunk["text"],
        )
        for position, chunk in enumerate(chunks, start=1)
    )
    return CONTEXT_PROMPT_TEMPLATE.format(context=context, question=question)


def _upstream_http_error(exc: APIStatusError) -> HTTPException:
    """
    Map an OpenAI error onto ours. A bad model name comes back as 404
    model_not_found, not 400, so both are folded into one "your request was
    bad" 400. Everything else upstream (auth, quota, provider outage) is our
    problem, not the caller's: 502.
    """
    status = 400 if exc.status_code in (400, 404) else 502
    # str(exc) is the SDK's own "Error code: 404 - {...}" text, which would put a
    # 404 and a stringified Python dict inside a 400 body. Take the provider's
    # message on its own instead, so the response stays readable JSON. The SDK
    # has already unwrapped the {"error": {...}} envelope, so body IS the error.
    message = exc.body.get("message") if isinstance(exc.body, dict) else None
    return HTTPException(status_code=status, detail=message or str(exc))


class Answer(BaseModel):
    """Structured model output — this is what makes the endpoint a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    # document_ids of the chunks the model actually used. Empty on a refusal.
    citations: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000, examples=["What is RAG?"])
    model: str = Field(default=DEFAULT_MODEL, examples=[DEFAULT_MODEL])
    top_k: int = Field(default=5, ge=1, le=50)


class AskResponse(BaseModel):
    answer: Answer
    model: str
    # Chunk ids retrieved and put in front of the model, in rank order. What the
    # model then chose to use is answer.citations, which is a subset of these.
    retrieved_chunk_ids: list[str]
    # Generation only. The embedding call for the question is not counted here.
    tokens_used: int
    latency_ms: int
    # None when the returned model has no price on file — see compute_cost_usd.
    cost_usd: float | None


class IngestRequest(BaseModel):
    document_id: str = Field(max_length=256, examples=["handbook-2026"])
    # No min_length here on purpose: it would answer an empty body with a 422,
    # and empty input is meant to be a clear 400. The handler checks it instead.
    text: str = Field(max_length=200_000, examples=["Retrieval-augmented generation ..."])
    # Free-form, but Pinecone only accepts strings, numbers, booleans and lists
    # of strings as metadata values, so this is restricted to strings. "source"
    # is pulled out by name; anything else is stored alongside it untouched.
    metadata: dict[str, str] = Field(default_factory=dict, examples=[{"source": "handbook.md"}])


class RetrievedChunk(BaseModel):
    id: str
    score: float
    document_id: str
    chunk_index: int
    source: str
    text: str


class RetrieveResponse(BaseModel):
    question: str
    matches: list[RetrievedChunk]


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness check — answers without touching OpenAI, so it costs nothing."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Retrieve the chunks nearest the question, then answer from those alone.

    The model is given the context and told to refuse rather than fill a gap
    from prior knowledge, so an unanswerable question comes back as a refusal
    with sources_needed set, not as a confident invention.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    if not rag.PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not configured")

    # Covers retrieval and generation both, since that is what the caller waited
    # for. In Session 1 this wrapped the single model call.
    start = time.perf_counter()

    try:
        chunks = await rag.search(request.question, top_k=request.top_k)
    except APIStatusError as exc:
        raise _upstream_http_error(exc) from exc
    except Exception as exc:  # ponytail: one mapping for the rest; split if a caller needs detail
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    try:
        completion = await client.chat.completions.parse(
            model=request.model,
            messages=[
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": build_grounding_prompt(request.question, chunks)},
            ],
            response_format=Answer,
        )
    except APIStatusError as exc:
        raise _upstream_http_error(exc) from exc
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
    # Generation only. The embedding call for the question is not priced in --
    # there is no embedding entry in MODEL_PRICES_PER_1K, and reporting a
    # partial figure under a name that means "the cost of this call" is worse
    # than reporting the one figure that has always meant generation.
    cost_usd = compute_cost_usd(
        completion.model,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )

    return AskResponse(
        answer=answer,
        model=completion.model,
        retrieved_chunk_ids=[chunk["id"] for chunk in chunks],
        tokens_used=usage.total_tokens if usage else 0,
        latency_ms=int((time.perf_counter() - start) * 1000),
        cost_usd=round(cost_usd, 6) if cost_usd is not None else None,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Chunk one document, embed each chunk and store it in the vector store.

    Re-sending the same document_id overwrites that document's chunks rather
    than duplicating them, so this is safe to retry.

        curl -X POST http://127.0.0.1:8000/ingest           -H "Content-Type: application/json"           -d '{
                "document_id": "handbook-2026",
                "text": "Retrieval-augmented generation combines a retrieval step with a generative model. ...",
                "metadata": {"source": "handbook.md"}
              }'

        {"document_id": "handbook-2026", "chunks_indexed": 1, "status": "indexed"}
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    if not rag.PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not configured")
    # Checked here rather than left to the splitter, which raises a ValueError
    # on construction. That way a misconfigured deployment is a 500 naming the
    # two variables, and any ValueError below is unambiguously the caller's.
    if rag.CHUNK_OVERLAP >= rag.CHUNK_SIZE:
        raise HTTPException(
            status_code=500,
            detail=f"CHUNK_OVERLAP ({rag.CHUNK_OVERLAP}) must be smaller than "
            f"CHUNK_SIZE ({rag.CHUNK_SIZE})",
        )

    document_id = request.document_id.strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        chunks_indexed = await rag.ingest_document(document_id, request.text, request.metadata)
    except APIStatusError as exc:
        raise _upstream_http_error(exc) from exc
    except ValueError as exc:
        # The only ValueError left is the per-request embedding batch cap, which
        # a caller fixes by sending a smaller document.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # ponytail: one mapping for the rest; split if a caller needs detail
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    if not chunks_indexed:
        # Text that is non-empty but yields no chunks, e.g. only punctuation the
        # splitter strips. A 200 saying "0 indexed" would read as success.
        raise HTTPException(status_code=400, detail="text produced no chunks to index")

    return IngestResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
    )


@app.get("/debug/retrieve", response_model=RetrieveResponse, tags=["rag"])
async def debug_retrieve(
    q: str = Query(max_length=4000, examples=["What is RAG?"]),
    top_k: int = Query(default=5, ge=1, le=50),
) -> RetrieveResponse:
    """
    Retrieval on its own: embed the question, return the nearest chunks with
    their similarity scores and metadata. No LLM is called and no answer is
    generated, so this shows exactly what a prompt would have been built from.

        curl -s "http://127.0.0.1:8000/debug/retrieve?q=What+is+RAG"

    Scores are cosine similarity, so higher is closer and 1.0 is identical.

    This reads out indexed content to anyone who can reach it. It is fine
    locally; think before leaving it reachable on a public deployment.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    if not rag.PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not configured")

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must not be empty")

    try:
        matches = await rag.search(question, top_k=top_k)
    except APIStatusError as exc:
        raise _upstream_http_error(exc) from exc
    except Exception as exc:  # ponytail: one mapping for the rest; split if a caller needs detail
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    # An empty list is a real answer here -- nothing indexed yet, or nothing
    # near enough. That is exactly what this endpoint exists to show, so it is
    # a 200 with no matches rather than an error.
    return RetrieveResponse(question=question, matches=matches)


# Mounted last so /health, /ask, /docs and /openapi.json win over this catch-all.
# html=True serves static/index.html at "/".
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
