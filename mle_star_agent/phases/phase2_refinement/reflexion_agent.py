import json
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.shared.callbacks import (
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)

logger = logging.getLogger(__name__)


_KNOWN_TARGET_COMPONENTS = [
    "architecture/model_capacity",
    "augmentation",
    "weighted_loss",
    "stereo_fusion",
    "calibration",
    "threshold_sweep",
    "preprocessing/lot_normalization",
    "optimizer/lr-schedule",
]


def _entry_target_component(entry: dict) -> str:
    fingerprint = entry.get("strategy_fingerprint") or {}
    target = (
        entry.get("target_component")
        or fingerprint.get("target_component")
        or "unknown"
    )
    return str(target or "unknown").strip().lower() or "unknown"


def _entry_mechanism_class(entry: dict) -> str:
    fingerprint = entry.get("strategy_fingerprint") or {}
    mechanism = fingerprint.get("mechanism_class") or "unknown"
    return str(mechanism or "unknown").strip().lower() or "unknown"


def _entry_result(entry: dict) -> dict:
    result = entry.get("result") or {}
    return {
        "outer": entry.get("outer"),
        "inner": entry.get("inner"),
        "ng_recall": result.get("ng_recall"),
        "overkill": result.get("overkill", result.get("overkill_rate")),
        "improved": result.get("improved"),
        "failure_reason": entry.get("failure_reason"),
    }


def load_and_reflect_fn(tool_context) -> str:
    """
    Group tried_approaches by target_component and mechanism_class, then save the
    structured reflection context to state["reflexion_memo"].
    """
    tried = [
        entry
        for entry in (tool_context.state.get("tried_approaches", []) or [])
        if isinstance(entry, dict)
    ]
    try:
        max_history = int(config.REFLEXION_MAX_HISTORY)
    except (TypeError, ValueError):
        max_history = 30
    if max_history > 0:
        tried = tried[-max_history:]

    grouped: dict[str, dict[str, list[dict]]] = {}
    tried_targets = set()
    for entry in tried:
        target = _entry_target_component(entry)
        mechanism = _entry_mechanism_class(entry)
        tried_targets.add(target)
        grouped.setdefault(target, {}).setdefault(mechanism, []).append(_entry_result(entry))

    never_tried = [
        target for target in _KNOWN_TARGET_COMPONENTS if target.lower() not in tried_targets
    ]
    memo = {
        "history_count": len(tried),
        "grouped_by_target_component": grouped,
        "never_tried_target_components": never_tried,
    }
    tool_context.state["reflexion_memo"] = memo
    return json.dumps(grouped, indent=2, default=str)


def save_reflexion_memo_fn(tool_context, memo_text: str) -> str:
    """Save the LLM-authored Reflexion narrative for the next planner turn."""
    tool_context.state["reflexion_memo_text"] = str(memo_text or "").strip()
    return "REFLEXION_MEMO_SAVED"


_load_and_reflect_tool = FunctionTool(func=load_and_reflect_fn)
_save_reflexion_memo_tool = FunctionTool(func=save_reflexion_memo_fn)


_INSTRUCTION = """You are the Reflexion Agent for Phase 2 Refinement.

Call `load_and_reflect_fn` first. Read the grouped tried_approaches by
target_component and mechanism_class.

Identify 2-3 dominant failure patterns, name any target_component that has never
been tried, and write a memo under 400 tokens. The memo must flag specific
mechanism_class strings to avoid when they repeatedly failed or regressed core
metrics. Do not propose code.

Call `save_reflexion_memo_fn` with the memo text. After the tool returns, output
exactly the returned status string.
"""


reflexion_agent = LlmAgent(
    name="reflexion_agent",
    model=config.MODEL,
    description=(
        "Summarizes tried_approaches into a compact self-reflection memo for the "
        "next refinement planner turn."
    ),
    instruction=_INSTRUCTION,
    tools=[_load_and_reflect_tool, _save_reflexion_memo_tool],
    include_contents="none",
    before_model_callback=log_context_size_callback,
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
