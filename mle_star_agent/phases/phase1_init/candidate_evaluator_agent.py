import logging
import time

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import (
    check_validation_cache_tool,
    code_validator_tool,
    store_validation_cache_tool,
)
from mle_star_agent.shared import code_runner, metric_guard
from mle_star_agent.shared.callbacks import count_tokens_callback, rate_limit_retry_callback
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.metrics_parser import metrics_to_dict, parse_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# How many candidate slots to create.  Must match baseline_coder_agent output.
# ---------------------------------------------------------------------------
NUM_SLOTS = 3


# ---------------------------------------------------------------------------
# Per-slot FunctionTool factory
# ---------------------------------------------------------------------------

def _make_read_script_fn(slot_index: int):
    """Return a FunctionTool that reads the script text for one slot from state."""

    def read_script_fn(tool_context) -> str:
        scripts = tool_context.state.get("candidate_scripts", [])
        if slot_index >= len(scripts):
            return f"ERROR: no script at slot {slot_index}."
        text = scripts[slot_index].get("script", "")
        if not text:
            return f"ERROR: script key missing at slot {slot_index}."
        return text

    read_script_fn.__name__ = f"read_candidate_script_{slot_index}_fn"
    read_script_fn.__doc__ = (
        f"Read and return the raw script text for candidate slot {slot_index} from state. "
        "Call this first, then pass the returned text to code_validator_agent."
    )
    return read_script_fn


def _make_update_script_fn(slot_index: int):
    """Return a FunctionTool that writes a corrected script back to state for one slot."""

    def update_candidate_script_fn(tool_context, script: str) -> str:
        scripts = list(tool_context.state.get("candidate_scripts", []))
        if slot_index >= len(scripts):
            return f"ERROR: slot {slot_index} out of range — cannot update script."
        entry = dict(scripts[slot_index])
        entry["script"] = script
        scripts[slot_index] = entry
        tool_context.state["candidate_scripts"] = scripts
        return f"Slot {slot_index}: script updated with validator-corrected version."

    update_candidate_script_fn.__name__ = f"update_candidate_script_{slot_index}_fn"
    update_candidate_script_fn.__doc__ = (
        f"Write a corrected script text back to state['candidate_scripts'][{slot_index}] "
        "before re-running it."
    )
    return update_candidate_script_fn


def _make_run_slot_fn(slot_index: int):
    """Return a FunctionTool function bound to one candidate slot."""

    def run_slot_fn(tool_context) -> str:
        # Skip if result already written (resume / checkpoint-gate support)
        existing = tool_context.state.get(f"candidate_result_{slot_index}")
        if existing is not None:
            return f"Slot {slot_index}: already evaluated — skipping."

        scripts = tool_context.state.get("candidate_scripts", [])
        if slot_index >= len(scripts):
            tool_context.state[f"candidate_result_{slot_index}"] = {
                "slot": slot_index,
                "status": "skipped",
                "reason": "no script at this slot index",
                "metrics": None,
            }
            return f"Slot {slot_index}: no candidate script present — marked skipped."

        candidate = scripts[slot_index]
        script_text = candidate.get("script", "")
        script_name = candidate.get("name", f"candidate_{slot_index}")
        architecture = candidate.get("architecture", "")

        logger.info("Running candidate slot %d: %s", slot_index, script_name)
        result = code_runner.run_script(script_text, timeout=config.TIMEOUT_SECONDS)

        metrics = parse_metrics(result.stdout)
        # Persistence-boundary guard: drop degenerate runs (dummy/empty split,
        # implausibly fast, no separability) so they never enter candidate_scores.
        metrics = metric_guard.guard_metrics(
            metrics, result.duration_ms, context=f"phase1 candidate slot {slot_index}"
        )

        result_dict = {
            "slot": slot_index,
            "name": script_name,
            "architecture": architecture,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ms": round(result.duration_ms, 1),
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-1000:],
            "metrics": metrics_to_dict(metrics) if metrics else None,
            "status": "success" if (result.returncode == 0 and metrics is not None) else "failed",
        }
        tool_context.state[f"candidate_result_{slot_index}"] = result_dict

        # Rate-limit delay between sequential LLM-heavy steps
        time.sleep(config.RATE_LIMIT_DELAY_SECONDS)

        if metrics:
            return (
                f"Slot {slot_index} ({script_name}): "
                f"accuracy={metrics.accuracy:.3f}, ng_recall={metrics.ng_recall:.3f}, "
                f"miss_rate={metrics.miss_rate:.3f}, overkill_rate={metrics.overkill_rate:.3f}, "
                f"f1={metrics.f1:.3f}, threshold={metrics.threshold}"
            )
        return (
            f"Slot {slot_index} ({script_name}): FAILED "
            f"(rc={result.returncode}, timed_out={result.timed_out}). "
            f"stderr: {result.stderr[-300:]}"
        )

    run_slot_fn.__name__ = f"run_candidate_slot_{slot_index}_fn"
    run_slot_fn.__doc__ = f"Run candidate script at slot {slot_index} and write result to state."
    return run_slot_fn


# ---------------------------------------------------------------------------
# Entry-gate FunctionTool — checked BEFORE any slot agent runs
# ---------------------------------------------------------------------------

def check_candidate_scores_checkpoint_fn(tool_context) -> str:
    """
    If candidate_scores.json already exists, restore scores AND pre-populate all
    candidate_result_N keys so the slot agents' skip-guard fires immediately.
    Returns CHECKPOINT_FOUND or CHECKPOINT_NOT_FOUND.
    """
    if not checkpoint_exists(config.CKPT_CANDIDATE_SCORES):
        return "CHECKPOINT_NOT_FOUND: proceed to evaluate candidates."

    data = load_checkpoint(config.CKPT_CANDIDATE_SCORES)
    scores = data.get("scores", [])
    tool_context.state["candidate_scores"] = scores

    # Pre-populate candidate_result_N so per-slot guards fire and skip execution
    for entry in scores:
        slot = entry.get("slot")
        if slot is not None:
            tool_context.state[f"candidate_result_{slot}"] = entry

    return (
        f"CHECKPOINT_FOUND: loaded {len(scores)} candidate score(s) from checkpoint. "
        "All candidate_result_N keys pre-populated — slot agents will skip execution."
    )


_checkpoint_gate_tool = FunctionTool(func=check_candidate_scores_checkpoint_fn)

_GATE_INSTRUCTION = """You are the Candidate Scores Checkpoint Gate.

Call `check_candidate_scores_checkpoint_fn` immediately.
- If it returns CHECKPOINT_FOUND: report how many scores were loaded and stop. Do nothing else.
- If it returns CHECKPOINT_NOT_FOUND: report that evaluation will proceed and stop.

Do not call any other tools.
"""

candidate_checkpoint_gate = LlmAgent(
    name="candidate_checkpoint_gate",
    model=config.MODEL,
    description="Checks for an existing candidate_scores.json checkpoint before any script is executed.",
    instruction=_GATE_INSTRUCTION,
    tools=[_checkpoint_gate_tool],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)

# ---------------------------------------------------------------------------
# Aggregator FunctionTool
# ---------------------------------------------------------------------------

def consolidate_candidate_scores_fn(tool_context) -> str:
    """
    Gather all candidate_result_N keys from state, build the scores list, save
    checkpoint, and return a JSON summary so the aggregator LLM can see the data.
    (Checkpoint-found path is handled by the entry gate above.)
    """
    import json as _json

    scores = []
    for i in range(NUM_SLOTS):
        result = tool_context.state.get(f"candidate_result_{i}")
        if result is not None:
            scores.append(result)

    tool_context.state["candidate_scores"] = scores
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(config.CKPT_CANDIDATE_SCORES, {"scores": scores})

    successful = sum(1 for s in scores if s.get("status") == "success")

    # Build a compact summary the LLM can read directly from this tool's return value
    summary_rows = []
    for s in scores:
        m = s.get("metrics") or {}
        summary_rows.append({
            "name": s.get("name"),
            "status": s.get("status"),
            "accuracy": m.get("accuracy"),
            "ng_recall": m.get("ng_recall"),
            "miss_rate": m.get("miss_rate"),
            "overkill_rate": m.get("overkill_rate"),
            "f1": m.get("f1"),
            "threshold": m.get("threshold"),
        })

    return (
        f"Aggregated {len(scores)} candidate result(s) ({successful} successful). "
        f"Saved to candidate_scores.json.\n\n"
        f"SCORES:\n{_json.dumps(summary_rows, indent=2)}"
    )


_consolidate_tool = FunctionTool(func=consolidate_candidate_scores_fn)

# ---------------------------------------------------------------------------
# Per-slot LlmAgents
# ---------------------------------------------------------------------------

def _make_slot_agent(slot_index: int) -> LlmAgent:
    read_fn = _make_read_script_fn(slot_index)
    read_tool = FunctionTool(func=read_fn)
    run_fn = _make_run_slot_fn(slot_index)
    run_tool = FunctionTool(func=run_fn)
    update_fn = _make_update_script_fn(slot_index)
    update_tool = FunctionTool(func=update_fn)

    instruction = f"""You are Candidate Evaluator Slot {slot_index}.

Your job is to validate, then execute the candidate script at slot index {slot_index}.

## Steps (follow in order)

### STEP 1 — Read the script
Call `read_candidate_script_{slot_index}_fn`. It returns the full script text from state.

### STEP 2 — Validate
First call `check_validation_cache_fn` with the script text from Step 1.
- If it returns "CACHE_HIT: VALIDATED": the script was already validated by the coder — skip
  `code_validator_agent`. Call `update_candidate_script_{slot_index}_fn` only if the validator
  previously corrected it (it won't have in this path), then proceed to STEP 3.
- If it returns "CACHE_HIT: VALIDATION_FAILED": the script is known-broken — skip
  `code_validator_agent`. Call `run_candidate_slot_{slot_index}_fn` to record the failure, then stop.
- If it returns "CACHE_MISS": call `code_validator_agent` with the script text.
  - If it returns "VALIDATED_SCRIPT:": extract the corrected script.
    Call `store_validation_cache_fn` with the corrected script and status "VALIDATED".
    If the validator changed the script, call `update_candidate_script_{slot_index}_fn`
    with the corrected text so it is written back to state before running.
  - If it returns "VALIDATION_FAILED": call `store_validation_cache_fn` with the original
    script and status "VALIDATION_FAILED". Call `run_candidate_slot_{slot_index}_fn` to
    record the failure, then stop.

### STEP 3 — Run
Call `run_candidate_slot_{slot_index}_fn`. This reads the (possibly updated) script from
state, executes it, parses METRICS, and writes `state["candidate_result_{slot_index}"]`.

### STEP 4 — Report
Report the result: key metrics if successful, or the error summary if failed.

Do not loop or retry beyond what is described above.
"""
    return LlmAgent(
        name=f"candidate_slot_{slot_index}_agent",
        model=config.MODEL,
        description=f"Validates then runs candidate script at slot {slot_index}.",
        instruction=instruction,
        tools=[read_tool, run_tool, update_tool, code_validator_tool, check_validation_cache_tool, store_validation_cache_tool],
        include_contents="none",
        after_model_callback=count_tokens_callback,
        on_model_error_callback=rate_limit_retry_callback,
    )


_slot_agents = [_make_slot_agent(i) for i in range(NUM_SLOTS)]

# ---------------------------------------------------------------------------
# candidate_sequential_evaluator: run one slot at a time to avoid CPU/API contention
# ---------------------------------------------------------------------------

candidate_sequential_evaluator = SequentialAgent(
    name="candidate_sequential_evaluator",
    description="Evaluates candidate scripts one at a time; each slot writes to its own state key.",
    sub_agents=_slot_agents,
)

# ---------------------------------------------------------------------------
# candidate_aggregator_agent: consolidates results → candidate_scores
# ---------------------------------------------------------------------------

_AGGREGATOR_INSTRUCTION = """You are the Candidate Aggregator Agent.

Steps:
1. Call `consolidate_candidate_scores_fn`. It gathers all results, saves the checkpoint,
   and returns a SCORES: block containing a JSON summary of every candidate.
2. Using the SCORES data returned by the tool (not from state directly), produce a
   brief summary table: Name | Status | ng_recall | miss_rate | overkill_rate | f1 | threshold
3. Identify the best candidate: lowest miss_rate first, then highest ng_recall, then highest f1.
4. State the recommended best candidate by name.

Do not write any new state keys — `consolidate_candidate_scores_fn` handles all state writes.
"""

candidate_aggregator_agent = LlmAgent(
    name="candidate_aggregator_agent",
    model=config.MODEL_PRO,
    description="Consolidates per-slot evaluation results into state['candidate_scores'] and saves checkpoint.",
    instruction=_AGGREGATOR_INSTRUCTION,
    tools=[_consolidate_tool],
    include_contents="none",
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)

# ---------------------------------------------------------------------------
# candidate_evaluator_agent: top-level SequentialAgent
# ---------------------------------------------------------------------------

candidate_evaluator_agent = SequentialAgent(
    name="candidate_evaluator_agent",
    description=(
        "Evaluates all candidate training scripts sequentially, then aggregates "
        "scores. Writes state['candidate_scores'] and checkpoints/candidate_scores.json. "
        "Entry gate skips all execution if candidate_scores.json already exists."
    ),
    sub_agents=[candidate_checkpoint_gate, candidate_sequential_evaluator, candidate_aggregator_agent],
)
