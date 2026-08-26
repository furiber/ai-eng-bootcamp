"""Offline checks for main.py — the OpenAI call is stubbed, so nothing is billed.

Run: python test_main.py
"""

import asyncio
from types import SimpleNamespace

import httpx
import openai

from fastapi.testclient import TestClient

import main
from main import Answer, app

client = TestClient(app)


def _fake_completion(parsed, *, refusal=None, total_tokens=42, model="gpt-4o-mini"):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            total_tokens=total_tokens,
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens - total_tokens // 2,
        ),
        model=model,
    )


A_CHUNK = {
    "id": "doc3_security#1",
    "score": 0.71,
    "document_id": "doc3_security",
    "chunk_index": 1,
    "source": "doc3_security.txt",
    "text": "Passwords must be at least 14 characters and rotated every 90 days.",
}

# What the model was handed on the last stubbed call, so the prompt itself can
# be asserted on rather than taken on trust.
LAST_CALL = {}


def _stub_retrieval(chunks=None, raises=None):
    """Replace the embed-and-query half of /ask."""

    async def fake_search(question, top_k=5):
        LAST_CALL["top_k"] = top_k
        if raises is not None:
            raise raises
        return [A_CHUNK] if chunks is None else list(chunks)

    main.rag.search = fake_search
    main.rag.PINECONE_API_KEY = "test-key"


def _stub(completion, chunks=None):
    """Replace both network calls: retrieval, then the canned completion."""
    _stub_retrieval(chunks=chunks)

    async def fake_parse(**kwargs):
        LAST_CALL["messages"] = kwargs["messages"]
        return completion

    main.client.chat.completions.parse = fake_parse


def _stub_raises(exc):
    """Replace the network call with a coroutine that raises."""
    _stub_retrieval()

    async def fake_parse(**_kwargs):
        raise exc

    main.client.chat.completions.parse = fake_parse


def _api_error(cls, message, code=None):
    """Build a real openai.APIStatusError subclass without touching the network."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(cls.status_code, request=request)
    # The SDK strips the {"error": {...}} envelope before it reaches the exception,
    # so body is the bare error object. Verified against a real 404 from the API.
    body = {"message": message, "type": "invalid_request_error", "param": None, "code": code}
    return cls(f"Error code: {cls.status_code} - {{'error': {body}}}", response=response, body=body)


def test_health_needs_no_key():
    response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok"}
    print("PASS  /health returns ok without a key")


def test_ask_returns_structured_answer():
    _stub(_fake_completion(Answer(answer="42", confidence=0.9, sources_needed=False, citations=["doc-a"])))
    response = client.post("/ask", json={"question": "meaning of life?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == {
        "answer": "42",
        "confidence": 0.9,
        "sources_needed": False,
        "citations": ["doc-a"],
    }, body
    # Everything retrieved, in rank order. answer.citations is what was used.
    assert body["retrieved_chunk_ids"] == ["doc3_security#1"], body
    assert body["tokens_used"] == 42
    assert body["model"] == "gpt-4o-mini"
    assert isinstance(body["latency_ms"], int)
    # 21 prompt + 21 completion tokens at gpt-4o-mini rates.
    assert body["cost_usd"] == round(21 / 1000 * 0.00015 + 21 / 1000 * 0.0006, 6)
    print("PASS  /ask returns a structured answer with usage metadata and cost")


def test_cost_uses_dated_model_id():
    """The API returns "gpt-4o-mini-2024-07-18"; prefix matching must still price it."""
    _stub(
        _fake_completion(
            Answer(answer="42", confidence=0.9, sources_needed=False, citations=["doc-a"]),
            model="gpt-4o-mini-2024-07-18",
        )
    )
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["cost_usd"] is not None and body["cost_usd"] > 0
    print("PASS  a dated model id is priced via prefix match")


def test_unknown_model_costs_null_not_wrong():
    _stub(
        _fake_completion(
            Answer(answer="42", confidence=0.9, sources_needed=False, citations=["doc-a"]),
            model="some-future-model",
        )
    )
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["cost_usd"] is None, body
    print("PASS  unknown model returns cost_usd null rather than a wrong figure")


def test_mini_is_cheaper_than_4o():
    from main import compute_cost_usd

    assert compute_cost_usd("gpt-4o-mini", 1000, 1000) < compute_cost_usd("gpt-4o", 1000, 1000)
    print("PASS  gpt-4o-mini prices below gpt-4o for identical usage")


def test_refusal_is_502_not_200():
    _stub(_fake_completion(None, refusal="I cannot help with that"))
    response = client.post("/ask", json={"question": "something refused"})
    assert response.status_code == 502, response.text
    assert "cannot help" in response.json()["detail"]
    print("PASS  a refusal surfaces as 502, never a 200 with empty content")


def test_blank_question_rejected_before_spending_tokens():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422, response.text
    print("PASS  empty question rejected by validation (no API call made)")


def test_static_page_is_served_at_root():
    response = client.get("/")
    assert response.status_code == 200, response.text
    assert "Week 2 Assignment" in response.text
    print("PASS  the static page is served at /")


def test_static_mount_does_not_shadow_the_api():
    """The catch-all mount must stay last, or these all become 404s."""
    assert client.get("/health").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    print("PASS  /health, /docs and /openapi.json still win over the static mount")


def test_missing_key_is_500():
    original = main.API_KEY
    main.API_KEY = None
    try:
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 500, response.text
        assert "OPENAI_API_KEY" in response.json()["detail"]
    finally:
        main.API_KEY = original
    print("PASS  missing key reported as a clear 500")


def test_invalid_model_is_400_not_502():
    """The real API answers an unknown model with 404 model_not_found, not 400."""
    _stub_raises(
        _api_error(openai.NotFoundError, "The model `nope` does not exist", "model_not_found")
    )
    response = client.post("/ask", json={"question": "hi", "model": "nope"})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail == "The model `nope` does not exist", detail
    # A 400 body must not quote the upstream 404, nor a stringified Python dict.
    assert "404" not in detail and "{" not in detail, detail
    print("PASS  an unknown model name returns 400 with a clean message, not a raw 502")


def test_bad_request_is_400():
    _stub_raises(_api_error(openai.BadRequestError, "Invalid value for 'temperature'"))
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 400, response.text
    print("PASS  an upstream 400 is passed through as 400")


def test_upstream_rate_limit_stays_502():
    """Only caller-fixable errors become 4xx; our quota problem is not the caller's."""
    _stub_raises(_api_error(openai.RateLimitError, "Rate limit reached"))
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 502, response.text
    print("PASS  an upstream 429 still surfaces as 502")


def _stub_ingest(chunks=3, raises=None):
    """Replace the whole chunk/embed/upsert path; neither service is touched."""

    async def fake_ingest(document_id, text, metadata=None):
        if raises is not None:
            raise raises
        return chunks

    main.rag.ingest_document = fake_ingest
    main.rag.PINECONE_API_KEY = "test-key"


def test_ingest_returns_document_id_chunks_and_status():
    _stub_ingest(chunks=3)
    response = client.post(
        "/ingest",
        json={
            "document_id": "handbook-2026",
            "text": "Retrieval-augmented generation combines retrieval with generation.",
            "metadata": {"source": "handbook.md"},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "document_id": "handbook-2026",
        "chunks_indexed": 3,
        "status": "indexed",
    }, response.text
    print("PASS  /ingest returns document_id, chunks_indexed and status")


def test_ingest_metadata_is_optional():
    _stub_ingest(chunks=1)
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "some text"})
    assert response.status_code == 200, response.text
    print("PASS  /ingest accepts a body with no metadata")


def test_ingest_empty_text_is_400():
    """Empty input is a clear 400, not the 422 a min_length constraint would give."""
    _stub_ingest()
    response = client.post("/ingest", json={"document_id": "doc-a", "text": ""})
    assert response.status_code == 400, response.text
    assert "text" in response.json()["detail"], response.text
    print("PASS  empty text is a 400 with a message naming the field")


def test_ingest_whitespace_only_text_is_400():
    _stub_ingest()
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "   \n\t  "})
    assert response.status_code == 400, response.text
    print("PASS  whitespace-only text is a 400, not an empty success")


def test_ingest_empty_document_id_is_400():
    _stub_ingest()
    response = client.post("/ingest", json={"document_id": "  ", "text": "some text"})
    assert response.status_code == 400, response.text
    assert "document_id" in response.json()["detail"], response.text
    print("PASS  a blank document_id is a 400")


def test_ingest_indexing_nothing_is_400_not_an_empty_200():
    """A 200 reading "0 chunks indexed" would look like the document was stored."""
    _stub_ingest(chunks=0)
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "..."})
    assert response.status_code == 400, response.text
    print("PASS  text that yields no chunks is a 400, not a 200 saying zero")


def test_ingest_missing_pinecone_key_is_500():
    _stub_ingest()
    main.rag.PINECONE_API_KEY = None
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "some text"})
    assert response.status_code == 500, response.text
    assert "PINECONE_API_KEY" in response.json()["detail"], response.text
    print("PASS  a missing Pinecone key is a 500 naming the variable")


def test_ingest_bad_chunk_settings_are_500_not_400():
    """A misconfigured deployment is ours to fix, so it must not read as a bad request."""
    _stub_ingest()
    original = main.rag.CHUNK_OVERLAP
    main.rag.CHUNK_OVERLAP = main.rag.CHUNK_SIZE
    try:
        response = client.post("/ingest", json={"document_id": "doc-a", "text": "some text"})
    finally:
        main.rag.CHUNK_OVERLAP = original
    assert response.status_code == 500, response.text
    assert "CHUNK_OVERLAP" in response.json()["detail"], response.text
    print("PASS  an overlap not smaller than the chunk size is a 500")


def test_ingest_upstream_rate_limit_is_502():
    _stub_ingest(raises=_api_error(openai.RateLimitError, "Rate limit reached"))
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "some text"})
    assert response.status_code == 502, response.text
    print("PASS  an upstream 429 during ingest surfaces as 502")


def test_ingest_unknown_embedding_model_is_400():
    """Same mapping as /ask: an upstream 404 model_not_found is the caller's error."""
    _stub_ingest(raises=_api_error(openai.NotFoundError, "The model does not exist"))
    response = client.post("/ingest", json={"document_id": "doc-a", "text": "some text"})
    assert response.status_code == 400, response.text
    print("PASS  an upstream 404 during ingest is passed through as 400")


def _stub_search(matches=None, raises=None):
    """Replace the embed-and-query path; neither service is touched."""

    async def fake_search(question, top_k=5):
        if raises is not None:
            raise raises
        return list(matches or [])

    main.rag.search = fake_search
    main.rag.PINECONE_API_KEY = "test-key"


_A_MATCH = {
    "id": "handbook-2026#0",
    "score": 0.87,
    "document_id": "handbook-2026",
    "chunk_index": 0,
    "source": "handbook.md",
    "text": "Retrieval-augmented generation combines retrieval with generation.",
}


def test_debug_retrieve_returns_scores_and_metadata():
    _stub_search([_A_MATCH])
    response = client.get("/debug/retrieve", params={"q": "What is RAG?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["question"] == "What is RAG?", body
    assert body["matches"] == [_A_MATCH], body
    print("PASS  /debug/retrieve returns chunks with scores and document metadata")


def test_debug_retrieve_never_calls_the_model():
    """The whole point: retrieval verifiable without generating an answer."""
    _stub_search([_A_MATCH])

    async def explode(**_kwargs):
        raise AssertionError("/debug/retrieve called the LLM")

    main.client.chat.completions.parse = explode
    assert client.get("/debug/retrieve", params={"q": "What is RAG?"}).status_code == 200
    print("PASS  /debug/retrieve does not call the LLM")


def test_debug_retrieve_defaults_to_five():
    seen = {}

    async def fake_search(question, top_k=5):
        seen["top_k"] = top_k
        return []

    main.rag.search = fake_search
    main.rag.PINECONE_API_KEY = "test-key"
    client.get("/debug/retrieve", params={"q": "What is RAG?"})
    assert seen["top_k"] == 5, seen
    client.get("/debug/retrieve", params={"q": "What is RAG?", "top_k": 2})
    assert seen["top_k"] == 2, seen
    print("PASS  /debug/retrieve asks for five matches unless told otherwise")


def test_debug_retrieve_empty_question_is_400():
    _stub_search([_A_MATCH])
    assert client.get("/debug/retrieve", params={"q": "   "}).status_code == 400
    print("PASS  a blank q is a 400")


def test_debug_retrieve_no_matches_is_200_not_404():
    """Nothing indexed yet is a real answer, and the thing this endpoint reveals."""
    _stub_search([])
    response = client.get("/debug/retrieve", params={"q": "What is RAG?"})
    assert response.status_code == 200, response.text
    assert response.json()["matches"] == [], response.text
    print("PASS  no matches is an empty 200, not an error")


def test_debug_retrieve_upstream_failure_is_502():
    _stub_search(raises=_api_error(openai.RateLimitError, "Rate limit reached"))
    response = client.get("/debug/retrieve", params={"q": "What is RAG?"})
    assert response.status_code == 502, response.text
    print("PASS  an upstream 429 during retrieval surfaces as 502")


def test_ask_puts_the_retrieved_chunks_in_the_prompt():
    """Retrieval is pointless if the text never reaches the model."""
    _stub(_fake_completion(Answer(answer="14 characters", confidence=0.9,
                                  sources_needed=False, citations=["doc3_security"])))
    client.post("/ask", json={"question": "What is the password policy?"})

    system, user = LAST_CALL["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    assert system["content"] == main.GROUNDING_SYSTEM_PROMPT
    assert A_CHUNK["text"] in user["content"], user["content"]
    # The document_id is given plainly so the model never parses the chunk id.
    assert "document_id: doc3_security" in user["content"], user["content"]
    assert "What is the password policy?" in user["content"], user["content"]
    print("PASS  /ask puts the retrieved chunk text and document_id in the prompt")


def test_ask_defaults_to_five_chunks_and_honours_top_k():
    _stub(_fake_completion(Answer(answer="42", confidence=0.9,
                                  sources_needed=False, citations=[])))
    client.post("/ask", json={"question": "q"})
    assert LAST_CALL["top_k"] == 5, LAST_CALL
    client.post("/ask", json={"question": "q", "top_k": 2})
    assert LAST_CALL["top_k"] == 2, LAST_CALL
    print("PASS  /ask retrieves five chunks unless top_k says otherwise")


def test_ask_with_no_matches_still_answers_through_the_schema():
    """An empty index must produce a refusal in the usual shape, not a 500."""
    _stub(
        _fake_completion(
            Answer(answer="I cannot answer from the documents available.",
                   confidence=0.1, sources_needed=True, citations=[])
        ),
        chunks=[],
    )
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieved_chunk_ids"] == [], body
    assert body["answer"]["sources_needed"] is True, body
    # The model still gets a well-formed prompt, just one saying there is nothing.
    assert "no matching documents" in LAST_CALL["messages"][1]["content"]
    print("PASS  no retrieved chunks still yields a refusal through the schema")


def test_ask_missing_pinecone_key_is_500():
    _stub(_fake_completion(Answer(answer="42", confidence=0.9,
                                  sources_needed=False, citations=[])))
    main.rag.PINECONE_API_KEY = None
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 500, response.text
    assert "PINECONE_API_KEY" in response.json()["detail"], response.text
    print("PASS  /ask without a Pinecone key is a 500 naming the variable")


def test_ask_retrieval_failure_is_502_and_never_reaches_the_model():
    _stub(_fake_completion(Answer(answer="42", confidence=0.9,
                                  sources_needed=False, citations=[])))
    _stub_retrieval(raises=_api_error(openai.RateLimitError, "Rate limit reached"))

    async def explode(**_kwargs):
        raise AssertionError("/ask called the model after retrieval failed")

    main.client.chat.completions.parse = explode
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 502, response.text
    print("PASS  a retrieval failure is a 502 and skips generation entirely")


def test_grounding_prompt_numbers_chunks_from_one():
    prompt = main.build_grounding_prompt("q", [A_CHUNK, dict(A_CHUNK, id="doc1#0",
                                                             document_id="doc1", text="other")])
    assert "[1] document_id: doc3_security" in prompt, prompt
    assert "[2] document_id: doc1" in prompt, prompt
    print("PASS  the grounding prompt numbers chunks from one")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall checks passed")
