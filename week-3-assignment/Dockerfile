# Spanish Tutor (week-3 capstone) for Render.
#
# Python runs the app; Node runs the Supabase MCP server the agents talk to. Without Node every
# database call fails, which is why this is not a plain Python image.

FROM node:22-slim AS node

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Borrow Node from the official image rather than Debian's older apt package.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 # Installed at build time so a cold start never downloads it. Keep in step with
 # SUPABASE_MCP_VERSION in agent_core.py.
 && npm install -g @supabase/mcp-server-supabase@0.12.0 \
 && npm cache clean --force

WORKDIR /app

# Requirements first so dependency installs stay cached across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT (default 10000) and requires binding to 0.0.0.0.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
