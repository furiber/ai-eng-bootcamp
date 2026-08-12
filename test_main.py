"""Self-check for the /ask route. Run with `python test_main.py` (or pytest).

The OpenAI call is stubbed out, so no API key and no network access are needed.
"""

import types

import httpx
import openai
from fastapi.testclient import TestClient

import main


def _client(stub) -> TestClient:
    main.client.responses.create = stub
    main.client.api_key = "test-key"
    return TestClient(main.app)


async def _ok(model: str, input: str):
    return types.SimpleNamespace(output_text=f"answer to: {input}", model=model)


async def _api_error(model: str, input: str):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(404, request=request, json={"error": {"message": "no such model"}})
    raise openai.NotFoundError("no such model", response=response, body=None)


def test_health():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_model():
    res = _client(_ok).post("/ask", json={"question": "hi", "model": "gpt-4o"})
    assert res.status_code == 200, res.text
    assert res.json() == {"answer": "answer to: hi", "model": "gpt-4o"}


def test_ask_uses_default_model_when_omitted():
    res = _client(_ok).post("/ask", json={"question": "hi"})
    assert res.json()["model"] == main.DEFAULT_MODEL


def test_ask_rejects_empty_question():
    assert _client(_ok).post("/ask", json={"question": ""}).status_code == 422


def test_ask_passes_through_upstream_client_errors():
    res = _client(_api_error).post("/ask", json={"question": "hi", "model": "nope"})
    assert res.status_code == 404, res.text
    assert "no such model" in res.json()["detail"]


def test_index_page_is_served_at_root():
    res = TestClient(main.app).get("/")
    assert res.status_code == 200
    assert "Ask API" in res.text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
