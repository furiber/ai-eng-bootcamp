"""Spanish Tutor: the one agent a user talks to.

Everything in the capstone is reached through here. The router does no work of its own --
it reads the request, picks the specialist whose description fits, and hands over. That is
the whole pattern from `demo1_routing.py`: sub-agents carry a `name` and a `description`,
and the router's instruction says when to choose each.

    User
      |
      v
  spanish_tutor  (this file -- routes, never answers directly)
      |
      +--> study_planner   (study_plan_agent.py)      new learner, no record yet
      +--> deck_builder    (capstone_deck_builder.py) learner with review history

Adding a third specialist means adding it to `sub_agents` and naming it in the instruction.
Nothing else changes -- the specialists do not know the router exists.

Run: python capstone_router.py "your request"
"""

import asyncio
import sys

from google.adk.agents import Agent

import capstone_deck_builder
import study_plan_agent
from agent_core import MODEL, UNTRUSTED_DATA_RULES, redact, run

APP_NAME = "capstone_router"

# The router itself spends one call to decide, then the chosen specialist runs its own job
# inside the same budget. So this is the larger of the two specialists' limits plus slack.
MAX_STEPS = 20

INSTRUCTION = """
You are the front desk of a Spanish tutoring service. You do not tutor and you do not touch
the database. You read what the user wants and hand it to exactly one specialist.

CHOOSING
- `study_planner` -- the request is about a NEW learner: an intake note from a parent, a
  description of a child, someone with no record in the system yet, or an explicit ask for a
  study plan or a placement test.
- `deck_builder` -- the request is about an EXISTING learner: an email address, a named
  learner who has been studying, an ask for more cards, the next deck, or what to practise
  next.

- A message headed "New learner intake", or laid out as a form about a child (age, likes,
  availability), always goes to `study_planner` -- even though it contains an email address.
  The email there is the parent's contact, not a sign of an existing learner.

RULES
- Never answer a Spanish-tutoring question yourself, and never write SQL. Delegate.
- If the intake form arrives with its fields still empty, do not delegate: ask the parent to
  fill in at least the parent email and the child's age.
- If the request could go either way, ask ONE short clarifying question: does this learner
  already have a record? Do not guess.
- If the request is not about Spanish tutoring at all, say so plainly and do not delegate.
""" + UNTRUSTED_DATA_RULES + """
DONE
You are done when a specialist has finished and you have passed its report back unchanged.
Do not summarise it away -- the specialist's report is the answer.
"""


def build_agent() -> Agent:
    """The router plus its specialists.

    Each specialist is constructed fresh here rather than imported as a singleton: building
    one spawns an npx subprocess for the Supabase MCP server, and an ADK sub-agent may only
    be parented once.
    """
    return Agent(
        name="spanish_tutor",
        model=MODEL,
        description="Front desk for the Spanish tutor: routes a request to the right specialist.",
        instruction=INSTRUCTION,
        sub_agents=[
            study_plan_agent.build_agent(),
            capstone_deck_builder.build_agent(),
        ],
    )


async def ask(request: str) -> tuple[str, list[str]]:
    """Run one request through the router. Returns the RAW answer and the raw ACT lines."""
    return await run(build_agent(), request.strip(), app_name=APP_NAME, max_steps=MAX_STEPS)


async def main() -> None:
    request = " ".join(sys.argv[1:]) or "Build the next deck for maria@example.com."
    print(f"\n=== Spanish Tutor (max {MAX_STEPS} steps) ===")
    print(f"Request: {request}\n")
    answer, _ = await ask(request)
    print(f"\n--- Answer ---\n{redact(answer)}\n")


if __name__ == "__main__":
    asyncio.run(main())
