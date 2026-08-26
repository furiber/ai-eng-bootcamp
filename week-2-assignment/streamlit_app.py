"""Local UI over the deployed /ask and /ingest endpoints.

The API is the source of truth. This page holds no retrieval logic of its own --
it posts JSON and renders what comes back, so anything it shows is something the
service actually returned.

    streamlit run streamlit_app.py

The API base URL comes from the ASK_API_BASE_URL environment variable, or from
the sidebar box if that is unset. No key of any kind is read or sent: the
service holds its own credentials, and this page never sees them.

    ASK_API_BASE_URL=https://your-service.onrender.com streamlit run streamlit_app.py
"""

import os

import httpx
import streamlit as st

# Priced server-side; anything else comes back with cost_usd null.
MODELS = ["gpt-4o-mini", "gpt-4o"]
# Render's free tier sleeps, so a first call can spend most of a minute waking.
TIMEOUT_SECONDS = 120.0
DEFAULT_BASE_URL = os.getenv("ASK_API_BASE_URL", "http://127.0.0.1:8000")


def call(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    """POST JSON to the API. Returns (status, body); status 0 means it never landed."""
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}{path}", json=payload, timeout=TIMEOUT_SECONDS
        )
    except httpx.ConnectError:
        return 0, f"Cannot reach {base_url}. Is the service running?"
    except httpx.HTTPError as exc:
        return 0, str(exc)

    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def show_error(status: int, body: dict | str) -> None:
    """One place for both failure shapes: unreachable, and a non-200 from the API."""
    if status == 0:
        st.error(body)
        return
    detail = body.get("detail", body) if isinstance(body, dict) else body
    st.error(f"HTTP {status} — {detail}")


st.set_page_config(page_title="Northwind RAG", layout="centered")
st.title("Northwind document assistant")
st.caption("Ask questions answered only from the ingested documents, or add a document.")

base_url = st.sidebar.text_input("API base URL", DEFAULT_BASE_URL)
st.sidebar.caption(
    "From ASK_API_BASE_URL if set. No credentials are entered here — "
    "the service holds its own."
)

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ask_tab:
    question = st.text_area("Question", "What is the mileage rate for travel over 50 miles?")
    model = st.selectbox("Model", MODELS)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Calling /ask…"):
            status, body = call(
                base_url,
                "/ask",
                {"question": question.strip(), "model": model, "top_k": top_k},
            )

        if status != 200:
            show_error(status, body)
        else:
            answer = body["answer"]

            # A refusal is a correct outcome, not a failure, so it is shown as
            # its own state rather than as an answer with a warning stuck on it.
            if answer["sources_needed"]:
                st.warning("**Refused — the documents do not cover this.**")
                st.write(answer["answer"])
            else:
                st.success(answer["answer"])

            if answer["citations"]:
                st.markdown("**Cited documents:** " + ", ".join(f"`{c}`" for c in answer["citations"]))
            else:
                st.markdown("**Cited documents:** none")

            cost = body.get("cost_usd")
            confidence_col, tokens_col, latency_col, cost_col = st.columns(4)
            confidence_col.metric("Confidence", f"{answer['confidence']:.0%}")
            tokens_col.metric("Tokens", body["tokens_used"])
            latency_col.metric("Latency", f"{body['latency_ms']} ms")
            # cost_usd is null when the model has no price on file -- say so
            # rather than rendering a misleading $0.000000.
            cost_col.metric("Cost", f"${cost:.6f}" if cost is not None else "unpriced")

            if cost is None:
                st.caption(f"No price on file for `{body['model']}`, so cost is not estimated.")

            # Everything the model was shown, against what it said it used.
            # The two differing is the point: retrieval is broad, citation is not.
            st.caption(
                f"Retrieved {len(body['retrieved_chunk_ids'])} chunks: "
                + ", ".join(f"`{c}`" for c in body["retrieved_chunk_ids"])
            )

            with st.expander("Raw JSON"):
                st.json(body)

with ingest_tab:
    document_id = st.text_input("document_id", "demo-doc", help="Re-using an id overwrites that document.")
    source = st.text_input("source (optional)", "pasted-in-streamlit")
    text = st.text_area("Text", height=200, placeholder="Paste the document text here…")

    if st.button("Ingest", type="primary", disabled=not text.strip()):
        with st.spinner("Calling /ingest…"):
            payload = {"document_id": document_id.strip(), "text": text}
            if source.strip():
                payload["metadata"] = {"source": source.strip()}
            status, body = call(base_url, "/ingest", payload)

        if status != 200:
            show_error(status, body)
        else:
            st.success(
                f"Indexed `{body['document_id']}` as {body['chunks_indexed']} "
                f"chunk{'s' if body['chunks_indexed'] != 1 else ''}."
            )
            st.caption(
                "Chunk ids are deterministic, so sending the same document_id again "
                "overwrites these chunks rather than adding a second copy."
            )
            with st.expander("Raw JSON"):
                st.json(body)
