import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.shared.callbacks import (
    _make_bypass_response,
    count_tokens_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared import metric_guard
from mle_star_agent.shared.selection_metrics import (
    AVERAGED_EVALUATION_KEY,
    select_best_record,
    selection_metrics_for_record,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FunctionTools
# ---------------------------------------------------------------------------


def _find_candidate_metrics(candidate_name: str | None = None, score: float | None = None) -> dict:
    if not checkpoint_exists(config.CKPT_CANDIDATE_SCORES):
        return {}
    data = load_checkpoint(config.CKPT_CANDIDATE_SCORES)
    for entry in data.get("scores", []):
        metrics = entry.get("metrics") or {}
        if candidate_name and entry.get("name") == candidate_name:
            return metrics
        if score is not None and abs(float(metrics.get("ng_recall", -1.0)) - score) < 1e-9:
            return metrics
    return {}


def _miss_rate_from_score(score: float) -> float:
    return max(0.0, 1.0 - float(score or 0.0))


def _metric_value(
    resume: dict, recovered: dict, resume_key: str, recovered_key: str, default: float
) -> float:
    value = resume.get(resume_key)
    if value is None:
        value = recovered.get(recovered_key, default)
    if value is None:
        value = default
    return float(value)


def _select_best_successful_candidate(scores: list[dict]) -> dict | None:
    successful = [
        s for s in scores
        if s.get("status") == "success" and s.get("metrics") is not None
    ]
    valid = []
    for candidate in successful:
        metrics = selection_metrics_for_record(candidate)
        try:
            ng_recall = float(metrics.get("ng_recall", 0.0) or 0.0)
        except (TypeError, ValueError):
            ng_recall = 0.0
        if ng_recall <= 0.0:
            continue
        if metric_guard.is_degenerate_solution(metrics):
            continue
        if metric_guard.has_corner_threshold(metrics):
            continue
        valid.append(candidate)
    best = select_best_record(valid)
    if best is None:
        return None

    best = dict(best)
    best["selection_metrics"] = selection_metrics_for_record(best)
    return best


def _l0_candidate_name() -> str | None:
    if not checkpoint_exists(config.CKPT_L0):
        return None
    data = load_checkpoint(config.CKPT_L0)
    return data.get("best_candidate_name")

def check_and_load_phase2_init_fn(tool_context) -> str:
    """
    Restore Phase 2 state from checkpoints.

    Priority:
    1. If best_pipeline.json exists, it is the authoritative mid-Phase-2 resume
       snapshot (written by evaluator_agent after every inner iteration).  Load
       loop counters from it; load the L0 base info from L0.json.
    2. Otherwise, if L0.json + phase2_init.json exist, load the initial values
       (outer=0, inner=0) — this means Phase 2 has not yet run any inner iteration.
    3. Otherwise return CHECKPOINT_NOT_FOUND.
    """
    if not (checkpoint_exists(config.CKPT_L0) and checkpoint_exists(config.CKPT_PHASE2_INIT)):
        return "CHECKPOINT_NOT_FOUND: generate L0 and Phase 2 init state now."

    l0 = load_checkpoint(config.CKPT_L0)
    tool_context.state["L0_script"] = l0["L0_script"]
    tool_context.state["L0_score"] = l0["L0_score"]

    # Prefer best_pipeline.json (updated live during Phase 2) over the frozen
    # phase2_init.json so that mid-Phase-2 restarts resume from the correct counters.
    if checkpoint_exists(config.CKPT_BEST_PIPELINE):
        resume = load_checkpoint(config.CKPT_BEST_PIPELINE)
        source = "best_pipeline.json"
    else:
        resume = load_checkpoint(config.CKPT_PHASE2_INIT)
        source = "phase2_init.json"

    current_best_score = float(resume.get("current_best_score", 0.0))
    recovered_metrics = _find_candidate_metrics(_l0_candidate_name(), current_best_score)
    best_overkill_rate = _metric_value(
        resume, recovered_metrics, "best_overkill_rate", "overkill_rate", 1.0
    )
    if best_overkill_rate == 1.0 and recovered_metrics.get("overkill_rate") is not None:
        best_overkill_rate = recovered_metrics["overkill_rate"]
    best_miss_rate = resume.get("best_miss_rate")
    if best_miss_rate is None:
        best_miss_rate = recovered_metrics.get("miss_rate", _miss_rate_from_score(current_best_score))

    tool_context.state["current_best_score"]    = current_best_score
    tool_context.state["best_overkill_rate"]    = float(best_overkill_rate)
    tool_context.state["best_miss_rate"]        = float(best_miss_rate)
    tool_context.state["best_accuracy"]         = _metric_value(
        resume, recovered_metrics, "best_accuracy", "accuracy", 0.0
    )
    tool_context.state["best_f1"]               = _metric_value(
        resume, recovered_metrics, "best_f1", "f1", 0.0
    )
    tool_context.state["best_selection_evaluation"] = resume.get("best_selection_evaluation")
    tool_context.state["best_pipeline_script"]  = resume.get("best_pipeline_script", l0["L0_script"])
    tool_context.state["refinement_population"] = resume.get("refinement_population", [])
    tool_context.state["no_improve_count"]      = resume.get("no_improve_count", 0)
    tool_context.state["outer_iteration"]       = resume.get("outer_iteration", 0)
    tool_context.state["inner_iteration"]       = resume.get("inner_iteration", 0)
    tool_context.state["stop_outer_loop"]       = resume.get("stop_outer_loop", False)
    tool_context.state["ensemble_iteration"]    = resume.get("ensemble_iteration", 0)
    tool_context.state["ensemble_best_score"]   = resume.get("ensemble_best_score", current_best_score)
    tool_context.state["ensemble_best_overkill"] = resume.get("ensemble_best_overkill", best_overkill_rate)
    tool_context.state["ensemble_best_accuracy"] = resume.get(
        "ensemble_best_accuracy", tool_context.state["best_accuracy"]
    )
    tool_context.state["ensemble_best_f1"] = resume.get(
        "ensemble_best_f1", tool_context.state["best_f1"]
    )
    tool_context.state["token_count"]           = resume.get("token_count", 0)

    return (
        f"CHECKPOINT_FOUND: L0 loaded (score={l0['L0_score']:.4f}). "
        f"Phase 2 loop state restored from {source} — "
        f"outer_iteration={tool_context.state['outer_iteration']}, "
        f"inner_iteration={tool_context.state['inner_iteration']}, "
        f"no_improve_count={tool_context.state['no_improve_count']}. "
        "Skip selection."
    )


def initialize_phase2_fn(tool_context) -> str:
    """
    Deterministically select the best candidate from state['candidate_scores'],
    write all Phase 2 initialisation state keys, and save L0.json + phase2_init.json.

    Selection uses averaged selection metrics when present, falling back to the
    original single-run metrics only when the averaged evaluation is incomplete.
    Falls back to the first available script if no candidate succeeded.
    """
    scores = tool_context.state.get("candidate_scores", [])
    scripts = tool_context.state.get("candidate_scripts", [])

    # Build a name→script lookup
    script_map = {s.get("name"): s.get("script", "") for s in scripts}

    best = _select_best_successful_candidate(scores)
    if best is not None:
        best_candidate_name = best["name"]
        selection_metrics = best["selection_metrics"]
        l0_score   = float(selection_metrics.get("ng_recall",    0.0))
        l0_overkill = float(selection_metrics.get("overkill_rate", 1.0))
        l0_miss_rate = float(selection_metrics.get("miss_rate", _miss_rate_from_score(l0_score)))
        l0_script  = script_map.get(best_candidate_name, "")
        logger.info(
            "Selected L0='%s' (ng_recall=%.4f, overkill=%.4f)",
            best_candidate_name, l0_score, l0_overkill,
        )
    else:
        tool_context.state["stop_outer_loop"] = True
        tool_context.state["phase1_stop_reason"] = "no_valid_phase1_candidate"
        logger.error(
            "No valid Phase 1 candidates found; refusing to initialize Phase 2."
        )
        return (
            "NO_VALID_PHASE1_CANDIDATE: all Phase 1 candidates failed, produced "
            "L0_score=0.0, selected forbidden corner thresholds, or collapsed to "
            "degenerate all-G/all-NG outputs. Refusing to write L0.json or "
            "phase2_init.json; regenerate Phase 1 candidates or stop for a new strategy."
        )

    if not l0_script:
        return f"ERROR: script for candidate '{best_candidate_name}' is empty."

    # Write L0 state
    tool_context.state["L0_script"] = l0_script
    tool_context.state["L0_score"] = l0_score

    # Initialise all Phase 2 loop state — preserve Phase 1 token_count
    existing_tokens = tool_context.state.get("token_count", 0) or 0
    tool_context.state["current_best_score"]     = l0_score
    tool_context.state["best_overkill_rate"]     = l0_overkill
    tool_context.state["best_miss_rate"]         = l0_miss_rate
    tool_context.state["best_accuracy"]          = float(selection_metrics.get("accuracy", 0.0))
    tool_context.state["best_f1"]                = float(selection_metrics.get("f1", 0.0))
    tool_context.state["best_selection_evaluation"] = best.get(AVERAGED_EVALUATION_KEY)
    tool_context.state["best_pipeline_script"]   = l0_script
    tool_context.state["no_improve_count"]       = 0
    tool_context.state["outer_iteration"]        = 0
    tool_context.state["inner_iteration"]        = 0
    tool_context.state["stop_outer_loop"]        = False
    tool_context.state["ensemble_iteration"]     = 0
    tool_context.state["ensemble_best_score"]    = l0_score
    tool_context.state["ensemble_best_overkill"] = l0_overkill
    tool_context.state["ensemble_best_accuracy"] = tool_context.state["best_accuracy"]
    tool_context.state["ensemble_best_f1"]       = tool_context.state["best_f1"]
    tool_context.state["token_count"]            = existing_tokens

    # Persist checkpoints
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    save_checkpoint(config.CKPT_L0, {
        "L0_script": l0_script,
        "L0_score": l0_score,
        "L0_overkill_rate": l0_overkill,
        "L0_miss_rate": l0_miss_rate,
        "L0_accuracy": tool_context.state["best_accuracy"],
        "L0_f1": tool_context.state["best_f1"],
        AVERAGED_EVALUATION_KEY: best.get(AVERAGED_EVALUATION_KEY),
        "selection_metrics": selection_metrics,
        "best_candidate_name": best_candidate_name,
    })

    phase2_init_data = {
        "current_best_score":     l0_score,
        "best_overkill_rate":     l0_overkill,
        "best_miss_rate":         l0_miss_rate,
        "best_accuracy":          tool_context.state["best_accuracy"],
        "best_f1":                tool_context.state["best_f1"],
        "best_selection_evaluation": tool_context.state["best_selection_evaluation"],
        "best_pipeline_script":   l0_script,
        "no_improve_count":       0,
        "outer_iteration":        0,
        "inner_iteration":        0,
        "stop_outer_loop":        False,
        "ensemble_iteration":     0,
        "ensemble_best_score":    l0_score,
        "ensemble_best_overkill": l0_overkill,
        "ensemble_best_accuracy": tool_context.state["best_accuracy"],
        "ensemble_best_f1":       tool_context.state["best_f1"],
        "token_count":            existing_tokens,
    }
    save_checkpoint(config.CKPT_PHASE2_INIT, phase2_init_data)

    return (
        f"Phase 2 initialised. L0='{best_candidate_name}', "
        f"l0_score={l0_score:.4f}. "
        "Saved L0.json and phase2_init.json."
    )


_check_load_tool = FunctionTool(func=check_and_load_phase2_init_fn)
_init_phase2_tool = FunctionTool(func=initialize_phase2_fn)


def _bypass_merger(callback_context, llm_request):
    try:
        result1 = check_and_load_phase2_init_fn(callback_context)
        if result1.startswith("CHECKPOINT_FOUND"):
            return _make_bypass_response(result1)
        result2 = initialize_phase2_fn(callback_context)
        return _make_bypass_response(f"{result1}\n\n{result2}")
    except Exception as exc:
        logger.warning("bypass_merger: exception — falling back to LLM: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Merger agent instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Merger Agent. Your role is to select the best candidate pipeline as L0 and initialise all Phase 2 loop state.

---
## STEP 1 — Check for existing checkpoints

Call `check_and_load_phase2_init_fn`.
- If it returns a string starting with "CHECKPOINT_FOUND": all state is restored — your job is done. Report what was loaded and stop.
- If it returns "CHECKPOINT_NOT_FOUND": proceed to STEP 2.

---
## STEP 2 — Initialise Phase 2 state

Call `initialize_phase2_fn` with no arguments. It will:
- Read `state["candidate_scores"]` and `state["candidate_scripts"]` directly
- Select the best candidate deterministically using averaged selection metrics when available
- Write all Phase 2 state keys and save L0.json + phase2_init.json

---
## STEP 3 — Report

After `initialize_phase2_fn` returns, produce a brief summary:
- The selected L0 candidate name and its key metrics (miss_rate, ng_recall, overkill_rate, f1, threshold)
- Whether averaged selection metrics were used
- Confirmation that L0.json and phase2_init.json were saved
- The preserved token_count carried over from Phase 1
"""

# ---------------------------------------------------------------------------
# Merger agent
# ---------------------------------------------------------------------------

merger_agent = LlmAgent(
    name="merger_agent",
    model=config.MODEL_PRO,
    description=(
        "Selects the best candidate as L0, initialises all Phase 2 loop counters "
        "and flags, and saves L0.json + phase2_init.json checkpoints."
    ),
    instruction=_INSTRUCTION,
    tools=[_check_load_tool, _init_phase2_tool],
    include_contents="none",
    before_model_callback=_bypass_merger,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
