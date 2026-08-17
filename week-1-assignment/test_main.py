"""Offline checks for main.py — the OpenAI call is stubbed, so nothing is billed.

Run: python test_main.py
"""

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
from main import Answer, app

client = TestClient(app)


def _fake_completion(parsed, *, refusal=None, total_tokens=42, model="gpt-4o-mini"):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=total_tokens),
        model=model,
    )


def _stub(completion):
    """Replace the network call with a coroutine returning a canned completion."""

    async def fake_parse(**_kwargs):
        return completion

    main.client.chat.completions.parse = fake_parse


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
    print("PASS  /ask returns a structured answer with usage metadata")


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall checks passed")
