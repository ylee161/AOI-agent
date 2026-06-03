"""Step 8.2 - Phase 1 isolated test using ADK Runner."""
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from mle_star_agent.config import (
    CKPT_CANDIDATE_SCRIPTS,
    CKPT_CANDIDATE_SCORES,
    CKPT_L0,
    CKPT_PHASE2_INIT,
    CKPT_DATA_SPLIT,
)
from mle_star_agent.phases.phase1_init import phase1_init

APP_NAME  = "test_phase1"
USER_ID   = "tester"
SESSION_ID = "phase1_test_session"

CHECKPOINTS = {
    "candidate_scripts.json": CKPT_CANDIDATE_SCRIPTS,
    "candidate_scores.json":  CKPT_CANDIDATE_SCORES,
    "L0.json":                CKPT_L0,
    "phase2_init.json":       CKPT_PHASE2_INIT,
}

print("=" * 60)
print("STEP 8.2 - PHASE 1 ISOLATED TEST")
print("=" * 60)

# Verify data_split checkpoint exists (required by Phase 1)
if not CKPT_DATA_SPLIT.exists():
    print(f"\nERROR: {CKPT_DATA_SPLIT} not found.")
    print("Run test_data_split.py first (step 8.1).")
    sys.exit(1)

print(f"\ndata_split.json found — Phase 1 will skip re-splitting.")
print("Starting Phase 1 agents via ADK Runner...\n")
print("-" * 60)


async def run_phase1():
    session_service = InMemorySessionService()

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={},
    )

    runner = Runner(
        agent=phase1_init,
        app_name=APP_NAME,
        session_service=session_service,
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text="Run Phase 1: ensure data split, generate candidate scripts, evaluate them, and select L0.")],
    )

    event_count = 0
    agent_turns = []

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
    ):
        event_count += 1

        # Print agent responses as they arrive
        if hasattr(event, "author") and event.author:
            if hasattr(event, "content") and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        agent_turns.append(event.author)
                        print(f"\n[{event.author}]")
                        # Print first 500 chars to avoid flooding terminal
                        text = part.text.strip()
                        if len(text) > 500:
                            print(text[:500] + f"\n... ({len(text)-500} more chars)")
                        else:
                            print(text)

        # Show tool calls
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    print(f"  --> tool call: {part.function_call.name}()")
                if hasattr(part, "function_response") and part.function_response:
                    resp = str(part.function_response.response)
                    if len(resp) > 200:
                        resp = resp[:200] + "..."
                    print(f"  <-- tool result: {resp}")

    print("\n" + "-" * 60)
    print(f"Phase 1 complete. Total events: {event_count}")
    return agent_turns


async def main():
    agent_turns = await run_phase1()

    print("\n" + "=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)

    all_ok = True
    for name, path in CHECKPOINTS.items():
        exists = path.exists()
        status = "PASS" if exists else "MISSING"
        size = f"{path.stat().st_size / 1024:.1f} KB" if exists else "—"
        print(f"  [{status}] {name:<30} {size}")
        if not exists:
            all_ok = False

    # Detailed check of key checkpoints
    if CKPT_CANDIDATE_SCRIPTS.exists():
        with open(CKPT_CANDIDATE_SCRIPTS) as f:
            data = json.load(f)
        scripts = data.get("scripts", [])
        print(f"\n  Candidate scripts saved: {len(scripts)}")
        for s in scripts:
            print(f"    - {s.get('name', '?')} | {s.get('architecture', '?')}")

    if CKPT_CANDIDATE_SCORES.exists():
        with open(CKPT_CANDIDATE_SCORES) as f:
            data = json.load(f)
        scores = data.get("scores", data)
        print(f"\n  Candidate scores:")
        if isinstance(scores, list):
            for s in scores:
                m = s.get('metrics') or {}
                print(f"    - {s.get('name','?')}: ng_recall={m.get('ng_recall','?')} miss_rate={m.get('miss_rate','?')} f1={m.get('f1','?')}")
        else:
            print(f"    {json.dumps(scores, indent=4)[:300]}")

    if CKPT_L0.exists():
        with open(CKPT_L0) as f:
            data = json.load(f)
        print(f"\n  L0 best score: {data.get('L0_score', '?')}")
        print(f"  L0 script name: {data.get('best_candidate_name', '?')}")

    if CKPT_PHASE2_INIT.exists():
        with open(CKPT_PHASE2_INIT) as f:
            data = json.load(f)
        keys = list(data.keys())
        print(f"\n  phase2_init.json keys: {keys}")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: PASS — all Phase 1 checkpoints present")
    else:
        print("RESULT: FAIL — some checkpoints missing (see above)")
    print("=" * 60)


MAX_RETRIES = 5
RETRY_DELAY = 30  # seconds between retries on 503

for attempt in range(1, MAX_RETRIES + 1):
    try:
        asyncio.run(main())
        break
    except Exception as e:
        err = str(e)
        if "503" in err or "UNAVAILABLE" in err or "high demand" in err:
            if attempt < MAX_RETRIES:
                print(f"\n[RETRY {attempt}/{MAX_RETRIES}] Gemini 503 — waiting {RETRY_DELAY}s before retry...")
                import time; time.sleep(RETRY_DELAY)
            else:
                print(f"\n[FAILED] Gemini still unavailable after {MAX_RETRIES} attempts.")
                print("Try again in a few minutes.")
                sys.exit(1)
        else:
            raise
