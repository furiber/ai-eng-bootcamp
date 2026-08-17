# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An AI engineering bootcamp workspace. Two separate, unrelated codebases live side by side:

1. **Repo root** — the tracked "Ask API" project: a FastAPI service with one endpoint
   (`POST /ask`) that forwards a question to an OpenAI model, plus a static frontend
   served from the same app. This is the deployable artifact.
2. **`week 1/`** — course exercise material, **untracked by git**. Five standalone
   FastAPI apps (`serve_stage1.py` … `serve_stage5.py`) that build up the same `/ask`
   idea incrementally, plus a Streamlit runner. Changes here are not version-controlled;
   do not assume `git status` will show them.

`capstone-answers.md` describes a planned (not yet built) Spanish flashcard tutoring app.
It is a design note, not a description of the current code.

## Commands (repo root)

```bash
pip install -r requirements.txt
uvicorn main:app --reload          # http://127.0.0.1:8000/  and /docs

python test_main.py                # run all checks, prints one line per test
pytest test_main.py::test_ask_returns_answer_and_model   # single test

docker build -t ask-api .
docker run --rm -p 10000:10000 --env-file .env ask-api
```

`test_main.py` has a dual entry point: a `__main__` block that runs every `test_*`
function in order, and standard pytest collection. It stubs
`main.client.responses.create`, so no API key or network is needed.

There is no linter or formatter configured.

## Commands (`week 1/`)

Has its own `.env`, `requirements.txt`, and virtualenv. Run one stage at a time — they
all bind port 8000:

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
streamlit run demo_page.py         # UI over all five stages, http://localhost:8501
python test_all_stages.py          # spawns each server and hits the REAL OpenAI API
```

`test_all_stages.py` costs money and needs a valid key — unlike the root tests.

## Architecture notes worth knowing before editing

**Single container, no CORS.** `main.py` mounts `StaticFiles(..., html=True)` at `/`.
That mount is a catch-all and must stay the *last* route registration, or it will
shadow `/health`, `/ask`, `/docs`, and `/openapi.json`.

**The OpenAI key is read lazily.** `AsyncOpenAI` is constructed at import time with
whatever `OPENAI_API_KEY` holds, possibly `None`. The `/ask` handler checks
`client.api_key` and returns a 500 with a clear message. This is deliberate: the app
must boot and pass `/health` without a key so Render container-start failures stay
readable. Do not move the key check to import time.

**Upstream error mapping.** `openai.APIStatusError` in the 4xx range is passed through
with its original status code (bad model name, quota); everything else upstream becomes
502. Tests cover this path.

**Uses the Responses API** (`client.responses.create`, `response.output_text`), not
chat completions. `week 1/` stages use the older sync `OpenAI()` client and
`completions.parse` — the two codebases intentionally differ.

**Render deployment.** `render.yaml` is a Docker blueprint with `healthCheckPath: /health`.
`OPENAI_API_KEY` is `sync: false`, so Render prompts for it and the value never enters
the repo. The container binds `0.0.0.0:$PORT` (Render requirement).

## Gotchas the README documents

If `/` 404s while `/health` still answers, a stale uvicorn process is holding port 8000
and serving old code. Verify with `curl -s http://127.0.0.1:8000/openapi.json`, then kill
it (the README has the PowerShell one-liner). `--reload` needs `watchfiles`, supplied by
`uvicorn[standard]`; without it the polling reloader can miss edits entirely.
