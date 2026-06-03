import json
import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from mle_star_agent import config

logger = logging.getLogger(__name__)


def log_context_size_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """
    before_model_callback that logs the serialized input size of each LLM call.

    This is the direct measure of context pressure the include_contents="none"
    fix is meant to reduce: with the fix, per-call input size for a given agent
    should stay roughly flat across loop iterations instead of growing monotonically.
    Does NOT rely on usage_metadata.prompt_token_count (not confirmed available via
    DeepSeek/LiteLLM) — it measures the request payload we send.
    """
    try:
        contents = llm_request.contents or []
        size = len(json.dumps([c.model_dump() for c in contents], default=str))
    except Exception:  # never let instrumentation break a real call
        return None

    agent_name = getattr(callback_context, "agent_name", None) or "unknown"
    logger.info(
        "CONTEXT_SIZE agent=%s turns=%d input_chars=%d",
        agent_name,
        len(contents),
        size,
    )
    return None


def count_tokens_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """after_model_callback that accumulates token usage into state['token_count']."""
    usage = llm_response.usage_metadata
    if usage and usage.total_token_count:
        current = callback_context.state.get("token_count", 0) or 0
        callback_context.state["token_count"] = current + usage.total_token_count
        logger.debug(
            "Token count updated: +%d → %d",
            usage.total_token_count,
            callback_context.state["token_count"],
        )
    return None


def rate_limit_retry_callback(
    callback_context: CallbackContext,  # noqa: ARG001
    llm_request: LlmRequest,  # noqa: ARG001
    error: Exception,
    **kwargs,
) -> Optional[LlmResponse]:
    """on_model_error_callback that logs transient 503/429 errors.

    Important ADK 2.1.0 semantics: an on_model_error_callback can only *substitute*
    a fallback LlmResponse; returning None re-raises the original error. It cannot
    re-invoke the model, so this hook cannot itself "retry" the call. The actual
    retry-with-backoff for transient rate-limit/server errors is handled one layer
    down by litellm's num_retries (set on MODEL / MODEL_PRO in config.py). By the
    time this callback fires, those retries have already been exhausted, so we log
    for visibility and return None to surface the failure to the caller.
    """
    msg = str(error).lower()
    transient = any(k in msg for k in (
        "503", "429", "resource_exhausted", "quota",
        "rate limit", "too many requests", "overloaded",
    ))
    if transient:
        logger.warning(
            "Transient model error after litellm retries were exhausted: %s", error
        )
    return None


def lite_mode_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """
    DEPRECATED (Tier 0): before_model_callback that downgraded reasoning agents to
    the flash model once cumulative tokens exceeded TOKEN_LITE_THRESHOLD. This fired
    exactly when context was largest and made the agents "dumb." Replaced by
    budget_stop_callback. Kept for reference / backward-compat imports; no longer wired.
    """
    if config.LITE_ABLATION_MODE:
        tokens = callback_context.state.get("token_count", 0) or 0
        if tokens > config.TOKEN_LITE_THRESHOLD:
            # Override model id dynamically before request is sent
            llm_request.config.model = config.MODEL.model


def budget_stop_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,  # noqa: ARG001
) -> None:
    """
    before_model_callback that winds the run down — rather than downgrading the model —
    once cumulative tokens exceed TOKEN_LITE_THRESHOLD (Tier 0 replacement for
    lite_mode_callback). Reasoning quality is preserved: agents always run on MODEL_PRO.

    Mechanism (why not escalate alone): setting actions.escalate in a before_model_callback
    only exits the *nearest* LoopAgent, does nothing for agents inside a SequentialAgent
    (e.g. the ablation variant steps), and is stripped by RetryLoopAgent for Phase 2/3.
    So we ALSO set state["stop_outer_loop"]=True, which ablation_flag_checker polls at the
    top of every outer iteration and turns into a proper escalate — this is the pipeline's
    proven stop path and works regardless of container type. Net effect: no NEW outer
    refinement cycles start past the soft budget; the current call/inner loop finishes,
    then Phase 2 winds down and Phase 3/4 still produce a submission from the best pipeline.

    The evaluator's TOKEN_BUDGET (10M) check remains the absolute inner-loop backstop.
    token_count is reset per retry cycle (see retry_loop_agent), so this does not bleed
    across attempts.
    """
    tokens = callback_context.state.get("token_count", 0) or 0
    if tokens > config.TOKEN_LITE_THRESHOLD:
        callback_context.actions.escalate = True
        # Proven, container-agnostic stop path (polled by ablation_flag_checker).
        callback_context.state["stop_outer_loop"] = True
        logger.warning(
            "Budget stop: token_count=%d exceeded TOKEN_LITE_THRESHOLD=%d — "
            "set stop_outer_loop=True to wind down refinement (no model downgrade).",
            tokens,
            config.TOKEN_LITE_THRESHOLD,
        )

