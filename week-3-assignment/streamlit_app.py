"""Spanish Tutor -- the capstone UI.

One page, one interface: everything goes through the router in `capstone_router.py`, which
picks the specialist. There is deliberately no page-per-agent. Adding a specialist changes
the router, not this file.

The three ADK teaching demos are not here any more -- they are still runnable on their own:
    python demo1_routing.py
    python demo2_mcp.py
    python demo3_full_system.py     (needs: uvicorn shipping_agent:app --port 8001)

Run:
    streamlit run streamlit_app.py
"""

import asyncio
import concurrent.futures
import hmac
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

langfuse = get_client()
GoogleADKInstrumentor().instrument()

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import capstone_router

# redact() strips credentials from anything shown in the browser. Importing agent_core does
# not spawn the MCP subprocess -- that happens in supabase_toolset(), called per run.
from agent_core import USE_OPENROUTER, redact

MAX_STEPS = capstone_router.MAX_STEPS


# --- Runner ---

def run_agent_sync(agent, message, timeout=180, max_steps=MAX_STEPS):
    """Run an ADK agent synchronously with trace capture, capped at max_steps LLM calls.

    Everything returned has been through redact(), so a credential cannot reach the browser
    even if an agent is talked into printing one.
    """

    async def _run():
        service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="capstone_ui", session_service=service)
        session = await service.create_session(app_name="capstone_ui", user_id="user1")
        content = types.Content(role="user", parts=[types.Part(text=message)])
        trace, final = [], "(no response)"
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=max_steps),
        ):
            author = getattr(event, "author", "unknown")
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                fr = getattr(part, "function_response", None)
                text = getattr(part, "text", None)
                if fc:
                    args = {k: redact(str(v)) for k, v in (dict(fc.args) if fc.args else {}).items()}
                    trace.append({"author": author, "type": "tool_call", "tool": fc.name, "args": args})
                elif fr:
                    result = redact(str(fr.response))[:800] if fr.response else ""
                    trace.append({"author": author, "type": "tool_response", "tool": fr.name, "result": result})
                elif text:
                    trace.append({"author": author, "type": "text", "text": redact(text)})
                    if event.is_final_response():
                        final = redact(text)
        langfuse.flush()
        return final, trace

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result(timeout=timeout)


def render_trace(trace):
    """Show the run as Think / Act / Observe, and call out each hand-off by the router."""
    if not trace:
        return
    seen = []
    for i, step in enumerate(trace, 1):
        author = step.get("author", "unknown")
        if author not in seen:
            seen.append(author)
            if len(seen) > 1:
                st.info(f"Routed to **{author}**")
        if step["type"] == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in step.get("args", {}).items())
            st.warning(f"**{i}. ACT** — `{author}` called **{step['tool']}**({args[:200]})")
        elif step["type"] == "tool_response":
            st.success(f"**{i}. OBSERVE** — `{author}` got a result from **{step['tool']}**")
            if step.get("result"):
                st.code(step["result"][:500], language="json")
        elif step["type"] == "text" and step.get("text", "").strip():
            st.markdown(f"**{i}. THINK** — `{author}`: {step['text'][:300]}")


# --- Page ---

st.set_page_config(page_title="Spanish Tutor", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;}</style>", unsafe_allow_html=True)

# --- Password gate ---
# Every request runs agents holding a Supabase access token and spending OpenRouter credit, so a
# public URL must not be open. Render sets RENDER_EXTERNAL_URL on every web service: when it is
# present the app refuses to run at all without APP_PASSWORD. Locally the gate is off unless
# APP_PASSWORD is set.
# ponytail: one shared password, no rate limit on guesses. Put a real login (st.login / OIDC) in
# front of it before anyone outside the project has the URL.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if os.getenv("RENDER_EXTERNAL_URL") and not APP_PASSWORD:
    st.error("APP_PASSWORD is not set for this deployment, so the app is locked.")
    st.stop()
if APP_PASSWORD and not st.session_state.get("authenticated"):
    st.header("Spanish Tutor")
    attempt = st.text_input("Password", type="password")
    if attempt:
        if hmac.compare_digest(attempt.encode(), APP_PASSWORD.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()

with st.sidebar:
    st.title("Spanish Tutor")
    st.caption("One router, two specialists")

    st.markdown("### Status")
    # The model can be reached two ways; only the one actually in use needs a key.
    if USE_OPENROUTER:
        st.success("Model — OpenRouter")
    elif os.getenv("GOOGLE_API_KEY"):
        st.success("Model — Google AI Studio")
        st.caption("Free tier: 20 requests/day")
    else:
        st.error("Model — no OPENROUTER_API_KEY or GOOGLE_API_KEY")

    if os.getenv("SUPABASE_ACCESS_TOKEN") and os.getenv("SUPABASE_PROJECT_REF"):
        st.success("Supabase")
    else:
        st.error("Supabase — not configured")

    # auth_check() raises on bad credentials rather than returning False. Tracing is optional, so
    # a wrong or quoted key must show a warning, not take the whole page down.
    try:
        langfuse_ok = langfuse.auth_check()
    except Exception:
        langfuse_ok = False
    if langfuse_ok:
        st.success("Langfuse tracing")
    else:
        st.warning("Langfuse — not connected")

    st.markdown("---")
    # Kept narrow on purpose: st.code does not wrap, and the sidebar clips long lines.
    st.markdown(
        "```\n"
        "User\n"
        " |\n"
        "spanish_tutor\n"
        " |  (routes only)\n"
        " +- study_planner\n"
        " |    new learner\n"
        " +- deck_builder\n"
        "      has history\n"
        "```"
    )
    st.caption(f"Loop capped at {MAX_STEPS} LLM calls.")

st.header("Spanish Tutor")
st.markdown(
    "Ask for anything a tutor would handle. The **router** reads the request and hands it to "
    "one specialist — you never pick the agent yourself."
)

# What a parent is asked for. Email and age are the two the study planner refuses to guess;
# the rest shape the level, the word choice and the practice schedule it recommends.
INTAKE_FORM = """New learner intake

Parent email:
Child's name:
Child's age:
Spanish so far (none / a few words / some lessons / can hold a simple conversation):
Words or phrases they already know:
Can they read and write yet (in English)?:
Things they love (hobbies, animals, sports, games, food, shows):
Things they dislike or find boring:
Why they are learning (school, family, holiday, fun):
Days and times they are free to practise:
How long they can concentrate in one go:
Anything else we should know (learning needs, other languages spoken at home):
"""

EXAMPLES = {
    "New learner — blank intake form": INTAKE_FORM,
    "New learner — filled-in example": """New learner intake

Parent email: ana.garcia@example.com
Child's name: Diego
Child's age: 9
Spanish so far (none / a few words / some lessons / can hold a simple conversation): a few words
Words or phrases they already know: hola, gracias, adiós
Can they read and write yet (in English)?: yes, confidently
Things they love (hobbies, animals, sports, games, food, shows): football, dogs, Minecraft, pizza
Things they dislike or find boring: long worksheets, anything about clothes
Why they are learning (school, family, holiday, fun): visiting cousins in Spain next summer
Days and times they are free to practise: Monday and Wednesday after school, Saturday morning
How long they can concentrate in one go: about 15 minutes
Anything else we should know (learning needs, other languages spoken at home): English only at home
""",
    "Next deck for an existing learner": "Build the next deck for maria@example.com.",
    "Something a tutor should decline": "What is the capital of France?",
}

choice = st.selectbox("Start from", list(EXAMPLES) + ["(write my own)"])
default = EXAMPLES.get(choice, "")
st.caption(
    "New learner? Fill in the form — parent email and child's age are required. "
    "Existing learner? Just give their email and ask for the next deck."
)
request = st.text_area("Request", value=default, height=340, key=f"req_{choice}")

if st.button("Send", type="primary", disabled=not request.strip()):
    with st.spinner("Routing…"):
        try:
            answer, trace = run_agent_sync(capstone_router.build_agent(), request.strip())
        except concurrent.futures.TimeoutError:
            st.error("The run timed out. The step cap or a slow tool call is the usual cause.")
            st.stop()
        except Exception as exc:  # surfaced rather than swallowed: quota and auth land here
            st.error(f"{type(exc).__name__}: {redact(str(exc))[:600]}")
            st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    with st.expander("Think / Act / Observe trace", expanded=True):
        render_trace(trace)
