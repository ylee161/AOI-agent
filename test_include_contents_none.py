"""
§3B smoke test — verify intra-activation tool visibility with include_contents="none".

The fix plan sets include_contents="none" on most agents to stop replaying the
whole session history. Before flipping any production agent we must confirm that
"none" only strips CROSS-AGENT / CROSS-TURN history — NOT the agent's own in-flight
tool calls and responses within a single activation.

Test design:
  - tool_a()      -> returns a secret token "RESULT_A_<nonce>"
  - tool_b(value) -> records what the LLM passed back in; the test asserts the
                     LLM passed the exact token returned by tool_a.

If include_contents="none" stripped the in-flight tool response, the LLM would
NOT see the token from tool_a when deciding how to call tool_b, and the assertion
would fail. If it passes, multi-tool agents (ablation steps, evaluator, coder)
are safe to flip.
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "google", "LiteLLM", "litellm"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from mle_star_agent.config import MODEL  # deepseek-v4-flash — cheap, fine for a wiring test

APP_NAME = "test_inc_none"
USER_ID = "tester"
SESSION_ID = "inc_none_session"

NONCE = uuid.uuid4().hex[:8]
SECRET_TOKEN = f"RESULT_A_{NONCE}"

# Captures what the LLM actually passed into tool_b.
_captured = {"value": None, "tool_a_called": False, "tool_b_called": False}


def tool_a() -> str:
    """Call this FIRST. Returns a secret token you must pass to tool_b."""
    _captured["tool_a_called"] = True
    return SECRET_TOKEN


def tool_b(value: str) -> str:
    """Call this SECOND with the exact token that tool_a returned."""
    _captured["tool_b_called"] = True
    _captured["value"] = value
    return "OK"


agent = LlmAgent(
    name="two_tool_agent",
    model=MODEL,
    include_contents="none",  # <-- the flag under test
    instruction=(
        "You are a wiring-test agent. Do exactly this, in order:\n"
        "STEP 1: Call tool_a(). It returns a secret token string.\n"
        "STEP 2: Call tool_b(value=<the exact token tool_a returned>).\n"
        "STEP 3: Reply with the single word DONE.\n"
        "Do not invent the token — use the exact string tool_a returned."
    ),
    tools=[tool_a, tool_b],
)


async def main() -> bool:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state={}
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text="Run the wiring test now.")],
    )

    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    print(f"  --> tool call: {part.function_call.name}({dict(part.function_call.args or {})})")
                if getattr(part, "function_response", None):
                    print(f"  <-- tool result: {part.function_response.response}")
                if getattr(part, "text", None) and part.text.strip():
                    print(f"  [text] {part.text.strip()[:120]}")

    print("\n" + "=" * 60)
    print(f"  secret token from tool_a : {SECRET_TOKEN}")
    print(f"  value seen by tool_b     : {_captured['value']}")
    print(f"  tool_a called            : {_captured['tool_a_called']}")
    print(f"  tool_b called            : {_captured['tool_b_called']}")
    print("=" * 60)

    ok = (
        _captured["tool_a_called"]
        and _captured["tool_b_called"]
        and _captured["value"] == SECRET_TOKEN
    )
    if ok:
        print("RESULT: PASS — in-flight tool response IS visible with include_contents='none'.")
        print("        Multi-tool agents are safe to flip to include_contents='none'.")
    else:
        print("RESULT: FAIL — the LLM did not carry tool_a's result into tool_b.")
        print("        include_contents='none' stripped intra-activation tool visibility.")
        print("        DO NOT flip multi-tool agents; revisit the plan (§3B / §9 risk).")
    return ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
