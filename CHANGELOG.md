# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-13

### Added

- `POST /api/ask` endpoint that sends a question to an OpenAI model and returns the answer, with an optional per-request `model` override.
- `GET /health` liveness endpoint.
- Static single-page frontend served at `/` that calls the API from the browser.
- Interactive API docs (Swagger UI at `/docs`, ReDoc at `/redoc`) via FastAPI's generated OpenAPI schema.
- Dockerfile and `render.yaml` blueprint for deploying the container to Render.
- `test_main.py` self-check covering the endpoint's success, validation, and upstream-error paths.
