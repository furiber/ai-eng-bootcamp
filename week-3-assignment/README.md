# ADK Multi-Agent Systems

Three progressive demos showing multi-agent system design using [Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/).

| Demo | What it shows | Protocol |
|------|--------------|----------|
| **Demo 1** — Routing | Router agent delegates to billing, technical, and escalation specialists | Local tools |
| **Demo 2** — MCP | Agent queries a live Supabase database; tools are auto-discovered at runtime | MCP |
| **Demo 3** — Full System | Combines routing + MCP + A2A with a remote shipping agent | MCP + A2A |
| **Spanish Tutor (UI)** | One page, one router, two specialists — the capstone interface | MCP |
| **Capstone Agent** | Single agent that builds a level-appropriate Spanish flashcard deck | MCP |

## Prerequisites

- **Python 3.12+** (required — earlier versions have asyncio incompatibilities with MCP)
- **Node.js / npm** (needed by the Supabase MCP server, launched via `npx`)
- A **Google API key** for Gemini models → [Get one here](https://aistudio.google.com/apikey)
- A **Supabase project** with a Personal Access Token → [Generate here](https://supabase.com/dashboard/account/tokens) *(Demos 2 & 3 only)*

## Setup

### 1. Create and activate a virtual environment

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
uv pip install -e .
```

**With plain pip:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -e .
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GOOGLE_API_KEY=your_google_api_key_here
SUPABASE_ACCESS_TOKEN=your_personal_access_token_here
SUPABASE_PROJECT_REF=your_project_ref_here
```

## Running the demos

### Demo 1 — Multi-Agent Routing (local tools only)

```bash
python demo1_routing.py
```

### Demo 2 — MCP + Supabase

Requires `SUPABASE_ACCESS_TOKEN` and `SUPABASE_PROJECT_REF` in `.env`.

```bash
python demo2_mcp.py
```

### Demo 3 — Full System (Routing + MCP + A2A)

Start the shipping agent in one terminal, then run the demo in another:

```bash
# Terminal 1 — start the A2A shipping agent
uvicorn shipping_agent:app --port 8001

# Terminal 2 — run the demo
python demo3_full_system.py
```

### Capstone Agent — Deck Builder

A single ADK agent for one multi-step capstone job: *look at a learner's review history,
work out what level they are ready for, and write them a new 8-card deck.*

It is deliberately **not** a multi-agent system. There is one specialist and one job, so a
router would add a hop and a hand-off for nothing. Add sub-agents when a second genuinely
different specialist shows up (a pronunciation coach, say).

**One-time setup.** Open the Supabase SQL Editor for the project in `SUPABASE_PROJECT_REF`
and run [`capstone_schema.sql`](capstone_schema.sql). It creates four tables — `learner`,
`deck`, `card`, `review` — and seeds one learner (`maria@example.com`) with six A1 cards
and enough review history for the agent to have something to judge.

```bash
python capstone_deck_builder.py

# Offline check of the Think/Act/Observe log parsing — no API key or network needed
python capstone_deck_builder.py --selfcheck
```

Every turn of the loop prints one line:

```
  [ 1] ACT      execute_sql({"query": "select id, name from learner where email = ..."})
  [ 1] OBSERVE  execute_sql -> {'rows': [{'id': 1, 'name': 'Maria Lopez'}]}
  [ 1] THINK    Found the learner. Now reading her review history.
  [ 2] ACT      execute_sql({"query": "select d.level, count(*) ..."})
```

`ACT` is a tool call, `OBSERVE` is what came back, `THINK` is the model's narration in
between, and `FINAL` is the answer. The step counter increments on each tool call.

The loop is capped by `MAX_STEPS = 12` in `capstone_deck_builder.py`, passed to ADK as
`RunConfig(max_llm_calls=...)`. Hit the cap and the run raises instead of looping forever.
ADK's own default is 500.

> **Free-tier quota.** Gemini's free tier allows 20 `generate_content` requests per day per
> model. One deck build spends several. A `429 RESOURCE_EXHAUSTED` means the daily quota is
> gone, not that the agent is broken.

### Spanish Tutor — the capstone UI

One page. Everything goes through the router in `capstone_router.py`, which reads the
request and hands it to one specialist. You never choose the agent.

```
User
  │
spanish_tutor            routes only, never answers
  ├── study_planner      new learner, no record yet
  └── deck_builder       learner with review history
```

```bash
streamlit run streamlit_app.py       # → http://localhost:8501
python capstone_router.py "Build the next deck for maria@example.com."
```

Adding a third specialist means adding it to `sub_agents` and naming it in the router's
instruction — the UI does not change.

The three ADK teaching demos are no longer pages in the app. They are still runnable on
their own:

```bash
python demo1_routing.py
python demo2_mcp.py
uvicorn shipping_agent:app --port 8001   # then, in another terminal:
python demo3_full_system.py
```

## Deploying to Render

The service is defined as `week-3-spanish-tutor` in the repo-root `render.yaml`, built from
this folder's `Dockerfile`. The image carries Node as well as Python because the agents reach
Supabase through the Supabase MCP server, which runs on Node.

1. Merge to the branch Render deploys from.
2. In Render: **New → Blueprint**, pick this repo. Render prompts for each secret:
   `OPENROUTER_API_KEY`, `SUPABASE_ACCESS_TOKEN`, `SUPABASE_PROJECT_REF`, `APP_PASSWORD`,
   `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`. Paste values **without quotes**.
3. Open the service URL and enter `APP_PASSWORD`.

`APP_PASSWORD` is required: on Render (detected by `RENDER_EXTERNAL_URL`) the app stays locked
without it. Every request runs agents holding a Supabase personal access token — which reaches
every project on that Supabase account — and spends OpenRouter credit.

Test the image locally before pushing:

```bash
docker build -t spanish-tutor:local .
docker run --rm -p 10000:10000 --env-file .env -e APP_PASSWORD=pick-one spanish-tutor:local
# → http://localhost:10000
```

`docker --env-file` keeps quotes literally, unlike python-dotenv. If `.env` has
`KEY="value"`, the container sees the quotes and that key fails (Langfuse shows "not
connected").

Free instances sleep after 15 minutes idle, so the first visit afterwards takes about a minute.

## Architecture

```
User Query
    │
    ▼
┌───────────────────────┐
│     Router Agent       │
├───────┬───────┬───────┤
│       │       │       │
▼       ▼       ▼       │
Billing  Tech  Shipping │
│       │       │       │
▼       ▼       ▼       │
MCP    Local    A2A     │
Server  Tools  Protocol │
│               │       │
▼               ▼       │
Supabase     Remote     │
  DB         Agent      │
└───────────────────────┘
```

