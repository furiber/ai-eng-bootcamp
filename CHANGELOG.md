# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The ask endpoint is served at `/ask` rather than `/api/ask`, matching the originally specified path.

### Added

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

### Fixed

- Inline emoji favicon on the frontend, removing a `favicon.ico` 404 from the browser console.

## [0.1.0] - 2026-08-13

### Added

- `POST /api/ask` endpoint that sends a question to an OpenAI model and returns the answer, with an optional per-request `model` override.
- `GET /health` liveness endpoint.
- Static single-page frontend served at `/` that calls the API from the browser.
- Interactive API docs (Swagger UI at `/docs`, ReDoc at `/redoc`) via FastAPI's generated OpenAPI schema.
- Dockerfile and `render.yaml` blueprint for deploying the container to Render.
- `test_main.py` self-check covering the endpoint's success, validation, and upstream-error paths.
