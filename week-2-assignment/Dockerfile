FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements first so dependency installs stay cached across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime. streamlit_app.py is a local dev UI and
# the test files are not run in the image, so none of them are copied.
COPY main.py .
COPY env_setup.py .
COPY rag.py .
COPY static ./static

# Render sets PORT (default 10000) and requires binding to 0.0.0.0.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
