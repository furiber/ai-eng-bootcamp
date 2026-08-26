"""Local UI for the assignment's /ask endpoint.

Two terminals — one for the API, one for this page:

    uvicorn main:app --reload          # terminal 1
    streamlit run streamlit_app.py     # terminal 2

Then open http://localhost:8501.
"""

import httpx
import streamlit as st

from main import DEFAULT_MODEL

MODELS = [DEFAULT_MODEL, "gpt-4o"]
TIMEOUT_SECONDS = 120.0


def ask(base_url: str, question: str, model: str) -> tuple[int, dict | str]:
    """POST to /ask. Returns (status, body); status 0 means the call never landed."""
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/ask",
            json={"question": question, "model": model},
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.ConnectError:
        return 0, f"Cannot reach {base_url} — is `uvicorn main:app --reload` running?"
    except httpx.HTTPError as exc:
        return 0, str(exc)

    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


st.set_page_config(page_title="Week 2 Assignment — Ask", layout="centered")
st.title("Week 2 Assignment — `/ask`")
st.caption("Type a question, get a structured answer back with its token and latency cost.")

base_url = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")
model = st.sidebar.selectbox("Model", MODELS)

question = st.text_area("Question", "What is Retrieval-Augmented Generation in one sentence?")

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Calling /ask…"):
        status, body = ask(base_url, question.strip(), model)

    if status == 0:
        st.error(body)
    elif status != 200:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        st.error(f"HTTP {status} — {detail}")
    else:
        answer = body["answer"]
        st.success(answer["answer"])

        cost = body.get("cost_usd")

        confidence_col, tokens_col, latency_col, cost_col = st.columns(4)
        confidence_col.metric("Confidence", f"{answer['confidence']:.0%}")
        tokens_col.metric("Tokens used", body["tokens_used"])
        latency_col.metric("Latency", f"{body['latency_ms']} ms")
        # cost_usd is null when the model has no price on file — say so rather
        # than rendering a misleading $0.000000.
        cost_col.metric("Cost", f"${cost:.6f}" if cost is not None else "unpriced")

        if cost is None:
            st.caption(f"No price on file for `{body['model']}`, so cost is not estimated.")

        if answer["sources_needed"]:
            st.warning("The model flagged this answer as needing sources.")

        with st.expander("Raw JSON"):
            st.json(body)
