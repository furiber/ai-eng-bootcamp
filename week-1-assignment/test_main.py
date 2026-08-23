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


def _stub(completion):
    """Replace the network call with a coroutine returning a canned completion."""

    async def fake_parse(**_kwargs):
        return completion

    main.client.chat.completions.parse = fake_parse


def _stub_raises(exc):
    """Replace the network call with a coroutine that raises."""

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
    _stub(_fake_completion(Answer(answer="42", confidence=0.9, sources_needed=False)))
    response = client.post("/ask", json={"question": "meaning of life?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == {"answer": "42", "confidence": 0.9, "sources_needed": False}
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
            Answer(answer="42", confidence=0.9, sources_needed=False),
            model="gpt-4o-mini-2024-07-18",
        )
    )
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["cost_usd"] is not None and body["cost_usd"] > 0
    print("PASS  a dated model id is priced via prefix match")


def test_unknown_model_costs_null_not_wrong():
    _stub(
        _fake_completion(
            Answer(answer="42", confidence=0.9, sources_needed=False),
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
    assert "Week 1 Assignment" in response.text
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall checks passed")
