# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A broken tool no longer costs a whole run. When Supabase stopped answering, the study
  planner retried the same query three times, then carried on without it -- spending its
  entire twenty-call budget on sixteen dictionary lookups for a plan it had no way to save,
  and ending in a raw `LlmCallsLimitExceededError` after five and a half minutes. A tool
  that fails outright three times now ends the run instead, and the page says which tool
  failed and what the last error was rather than showing a spinner that never resolves. The
  same run now costs four LLM calls instead of twenty and stops in about a minute. A
  Wiktionary "no such word" is an answer the agent is meant to work around, so it does not
  count as a failure; the limit is `MAX_TOOL_FAILURES`, default 3.

### Added

- A Render service, `week-3-spanish-tutor`, for the Spanish Tutor app: a Docker image with
  Python and Node (the Supabase MCP server runs on Node, installed at build time and pinned
  to 0.12.0), pinned Python requirements, and a Blueprint entry that prompts for every
  secret. On Render the app is locked behind `APP_PASSWORD` and refuses to run without it.
- `week-3-assignment/capstone_router.py`, a `spanish_tutor` router that is now the single
  thing a user talks to. It routes to `study_planner` for a learner with no record yet and
  to `deck_builder` for one with review history, asks a clarifying question rather than
  guessing when a request could go either way, and passes the specialist's report back
  unchanged. Adding a specialist means editing the router, not the UI.
- `week-3-assignment/capstone_vocabulary.sql`, a 200-word Spanish vocabulary bank across 16
  subjects and three CEFR levels, with accents preserved so dictionary lookups match. 175 of
  the 200 were confirmed against Wiktionary; none came back wrong.

- `week-3-assignment/capstone_deck_builder.py`, a single Google ADK agent for one capstone
  job: read a learner's Spanish flashcard review history, judge the CEFR level they are
  ready for, and write them a new 8-card deck. It reaches the database through the Supabase
  MCP server rather than hand-written tools, so it picks up `execute_sql` and friends at
  runtime. The loop is capped at 12 LLM calls via `RunConfig(max_llm_calls=...)` so it
  cannot run away, and every turn prints as Think / Act / Observe. `--selfcheck` verifies
  the log parsing offline, with no API key or network.
- `week-3-assignment/capstone_schema.sql`, the four tables the capstone slice needs
  (`learner`, `deck`, `card`, `review`) with row-level security on and one seeded learner,
  so the deck builder has a learner and some review history to work from on first run.

- An Ingest tab on the `week-2-assignment` Streamlit page, and a rewritten Ask tab showing
  the cited document ids, the chunk ids that were retrieved, and a refusal as its own state
  rather than an answer with a warning attached. The page holds no retrieval logic: it posts
  to the API and renders the reply, so what it shows is what the service returned. Its base
  URL comes from `ASK_API_BASE_URL` or the sidebar, and it handles no credentials -- those
  stay on the service.
- `/ask` on `week-2-assignment` now answers from retrieved documents rather than from the
  model's own knowledge. It embeds the question, retrieves the nearest chunks (five by
  default, set per request with `top_k`) and answers from those alone, refusing when the
  context does not support an answer. The response gains `retrieved_chunk_ids`, everything
  put in front of the model in rank order, and each answer gains `citations`, the
  `document_id`s the model says it used. `tokens_used` and `cost_usd` keep their Session 1
  meaning of generation figures; `latency_ms` now covers retrieval as well, since that is
  what the caller waited for.
- `GET /debug/retrieve?q=...` on `week-2-assignment`, returning the nearest indexed chunks
  with their similarity scores, document id, chunk index and source. It calls no language
  model and generates no answer, so retrieval quality can be judged on its own before
  generation is wired in. No matches is an empty `200` rather than an error: nothing indexed
  yet is a real answer, and the one this endpoint exists to surface.
- `POST /ingest` on `week-2-assignment`: chunk a document, embed each chunk and index it,
  returning the document id, the number of chunks indexed and a status. Chunks are stored
  under deterministic ids, so re-sending the same document id overwrites its chunks instead
  of duplicating them and the endpoint is safe to retry. Re-sending a document that has since
  got shorter also removes its surplus chunks, which would otherwise linger from the longer
  previous version and keep matching searches. Empty or whitespace-only text, a
  blank document id, and text that yields no chunks are each a `400` with a message naming
  the problem -- a `200` reading zero chunks indexed would look like the document was stored.
- Chunking via `RecursiveCharacterTextSplitter`, sized by `CHUNK_SIZE` and `CHUNK_OVERLAP`.
  Both are counted in characters rather than tokens. The splitter is built on first use, not
  at import: it refuses an overlap that is not smaller than the chunk size, and since both
  come from the environment, building it at import would let one bad setting stop the app
  booting and take `/health` down with it. Deferred, it is a `500` naming both variables.
- Pinecone vector-store support in `week-2-assignment`, in a new `rag.py`: embedding, upsert
  and similarity search, all configured from environment variables. Ingest and query embed
  through one function reading one model and dimension setting, because vectors from different
  models are not comparable and mixing them returns quietly wrong results rather than failing.
- A `rag.check()` health check, runnable as `python rag.py`, confirming Pinecone is configured,
  reachable and agreeing with the configured embedding size. It makes no OpenAI call, so it
  costs nothing, and it reports key lengths rather than key values. A dimension mismatch is
  reported as a failure rather than a warning: it is accepted silently at config time and
  otherwise only surfaces later as every upsert being rejected.
- A third deployable service, `week-2-assignment`, beginning as a copy of `week-1-assignment`:
  the same typed `/ask` endpoint with structured output, token, latency and cost figures, with
  its own Dockerfile, static frontend, Streamlit page and Render blueprint entry. It rebuilds
  only when files under `week-2-assignment/` change. Session 2 builds retrieval on top of it.
- A README for `week-1-assignment`, covering the response shape, the error-status mapping,
  local and container runs, the Streamlit page, and the Render blueprint. It records two
  things that are easy to get wrong: the key comes from the repository root's `.env` rather
  than one in that folder, and an unknown model is reported by the SDK as `404
  model_not_found`, not `400`.
- A static frontend for `week-1-assignment`, served at `/` from the same container as the API.
  It posts to `/ask` and shows the answer with confidence, tokens, latency and cost.
- A `Cost` metric on the Streamlit page, alongside confidence, tokens and latency. An unpriced
  model reads `unpriced` rather than `$0.000000`, which would look like a free call.
- `cost_usd` on the `week-1-assignment` `/ask` response, estimating spend from input and output
  token counts. It is `null` for a model with no price on file, rather than reporting a figure
  derived from the wrong rates.
- A second deployable service, `week-1-assignment`: a typed `/ask` endpoint returning structured
  model output, with its own Dockerfile and a Render blueprint entry. It only rebuilds when files
  under `week-1-assignment/` change.
- A Streamlit page for exercising that endpoint locally.
- Local troubleshooting notes in the README for a stale uvicorn process holding port 8000.

### Changed

- The Spanish Tutor page opens on a new-learner intake form that asks the parent for
  everything the study planner uses: email, the child's name and age, Spanish so far and
  words already known, whether they can read yet, likes, dislikes, why they are learning,
  free times and concentration span. A filled-in example sits alongside it. The planner now
  steers placement words towards the child's likes and away from their dislikes, skips words
  the parent says they know, and never sets sessions longer than the stated concentration
  span.
- The Streamlit app is one page fronted by the router, replacing the five-page sidebar. The
  three ADK teaching demos are no longer pages; they remain runnable as standalone scripts.
- Agents reach Gemini through OpenRouter when `OPENROUTER_API_KEY` is set, sidestepping the
  Google AI Studio free tier's 20-requests-per-day cap. Set `USE_OPENROUTER=0` to go back to
  the direct Google key.

- The ask endpoint is served at `/ask` rather than `/api/ask`, matching the originally specified path.

### Fixed

- A wrong or missing Langfuse key no longer takes down the Spanish Tutor page. The sidebar's
  tracing check raised on bad credentials; it now shows "not connected" and the app works.
- Dictionary lookups return short English answers ("dog", not "dog (the species Canis
  familiaris ...)"), so placement tests have answers a child can actually match. Twelve
  existing cards and placement items were shortened the same way.
- The deck builder now applies the level rule in code. On its first live run it scored a
  learner at 83%, quoted the "80% or better means promote" rule, and kept them on A1 anyway;
  the level is now computed from review history and handed to the agent as a fact.
- New decks draw their words from the 200-word vocabulary bank, so cards get short English
  answers ("bread", not "bread (food made by baking cereal dough)") and never repeat a word
  the learner already has.
- `week-3-assignment` now pins `mcp` below 2.0. The 2.x release moved `mcp.shared.session`,
  which made `McpToolset` fail to import, silently leaving the MCP demos with no tools at
  all.
- An unknown model name sent to `week-1-assignment`'s `/ask` now returns a `400` with the
  upstream message, instead of a bare `502`. OpenAI reports an unknown model as `404
  model_not_found` rather than `400`, so both upstream codes are treated as a bad request
  from the caller. Upstream auth, quota and provider failures still return `502`, since
  those are not the caller's to fix.
- Inline emoji favicon on the frontend, removing a `favicon.ico` 404 from the browser console.

### Security

- The week-3 agents can no longer reach tables outside their job. They are given only the
  Supabase `execute_sql` tool (no schema listing, migrations or branching), and every query is
  checked before it runs: the deck builder may touch only `deck` and `card`, the study planner
  only its seven tables, and anything else is refused.

## [0.1.0] - 2026-08-13

### Added

- `POST /api/ask` endpoint that sends a question to an OpenAI model and returns the answer, with an optional per-request `model` override.
- `GET /health` liveness endpoint.
- Static single-page frontend served at `/` that calls the API from the browser.
- Interactive API docs (Swagger UI at `/docs`, ReDoc at `/redoc`) via FastAPI's generated OpenAPI schema.
- Dockerfile and `render.yaml` blueprint for deploying the container to Render.
- `test_main.py` self-check covering the endpoint's success, validation, and upstream-error paths.
