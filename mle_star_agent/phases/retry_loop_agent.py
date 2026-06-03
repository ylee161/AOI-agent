"""
RetryLoopAgent — custom BaseAgent that runs Phase 2 → Phase 3 → Phase 4 in a
retry loop without the escalation-bubbling problem of nested LoopAgents.

Why a custom BaseAgent and not LoopAgent:
  ADK's LoopAgent yields every event (including ones with escalate=True) to its
  parent before processing them internally. This means any escalation from Phase 2's
  inner loop would propagate upward and prematurely kill the retry cycle before
  Phase 3, Phase 4, or the acceptance check could run.

  This agent instead runs the three phases as a plain Python while-loop and strips
  the escalate flag from events emitted by Phase 2 and Phase 3 before yielding them
  upward. Phase 4 events pass through unchanged.
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from typing_extensions import override

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing

from mle_star_agent import config
from mle_star_agent.phases.phase2_refinement.ablation_agent import NUM_ABLATION_VARIANTS
from mle_star_agent.shared.checkpoint_io import save_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _archive_glob(pattern: str, archive_dir: Path) -> None:
    """Move matching checkpoints into a retry archive. Logs errors but does not raise."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(pattern):
        try:
            src = Path(p)
            dest = archive_dir / src.name
            if dest.exists():
                dest = archive_dir / f"{src.stem}.{int(os.path.getmtime(src))}{src.suffix}"
            os.replace(src, dest)
            logger.debug("Archived checkpoint: %s -> %s", src, dest)
        except OSError as e:
            # Log but continue — an archive failure is recoverable: Phase 4
            # will reload the old result, the acceptance check will fail again,
            # and the retry budget will be exhausted gracefully rather than
            # crashing the entire pipeline session.
            logger.error("Could not archive checkpoint %s: %s", p, e)


def _candidate_metrics_for_score(score: float) -> dict:
    try:
        if not config.CKPT_CANDIDATE_SCORES.exists():
            return {}
        data = json.loads(config.CKPT_CANDIDATE_SCORES.read_text(encoding="utf-8"))
        for entry in data.get("scores", []):
            metrics = entry.get("metrics") or {}
            if abs(float(metrics.get("ng_recall", -1.0)) - score) < 1e-9:
                return metrics
    except Exception as e:
        logger.warning("Could not recover candidate metrics: %s", e)
    return {}


def _miss_rate_from_score(score: float) -> float:
    return max(0.0, 1.0 - float(score or 0.0))


def _state_or_recovered(state, recovered: dict, state_key: str, recovered_key: str, default: float) -> float:
    value = state.get(state_key)
    if value is None:
        value = recovered.get(recovered_key, default)
    if value is None:
        value = default
    return float(value)


def _reset_for_retry(state, attempt: int) -> None:
    """
    Clear Phase 2/3/4 checkpoints and reset all loop-state keys so the next
    attempt starts fresh while keeping the best pipeline script found so far.
    """
    ckpt_dir = str(config.CHECKPOINT_DIR)
    archive_dir = config.CHECKPOINT_DIR / "retry_archives" / f"attempt_{attempt}"

    # ── Checkpoint cleanup ────────────────────────────────────────────────────
    for pattern in [
        f"{ckpt_dir}/submission.json",      # Phase 4 — MUST delete or Phase 4 skips re-run
        f"{ckpt_dir}/ensemble.json",        # Phase 3
        f"{ckpt_dir}/ensemble_*.json",      # Phase 3 attempt files
        f"{ckpt_dir}/ablation_*.json",      # Phase 2 ablation + per-variant files
        f"{ckpt_dir}/refinement_*.json",    # Phase 2 refinement attempt files
        f"{ckpt_dir}/error_analysis_*.json",  # Phase 2 deterministic FP/FN evidence
        f"{ckpt_dir}/error_analysis_report_*.json",  # Phase 2 interpreted FP/FN reports
        f"{ckpt_dir}/diagnosis_*.json",     # Phase 2 diagnosis files
    ]:
        _archive_glob(pattern, archive_dir)

    # Rewrite best_pipeline.json with reset counters but preserve the current
    # best script so Phase 2 uses it as its new starting point.
    best_script   = state.get("best_pipeline_script", "")
    best_score    = float(state.get("current_best_score", 0.0))
    recovered = _candidate_metrics_for_score(best_score)
    best_overkill = _state_or_recovered(
        state, recovered, "best_overkill_rate", "overkill_rate", 1.0
    )
    if best_overkill == 1.0 and recovered.get("overkill_rate") is not None:
        best_overkill = float(recovered["overkill_rate"])
    best_miss = _state_or_recovered(
        state, recovered, "best_miss_rate", "miss_rate", _miss_rate_from_score(best_score)
    )
    best_accuracy = _state_or_recovered(state, recovered, "best_accuracy", "accuracy", 0.0)
    best_f1 = _state_or_recovered(state, recovered, "best_f1", "f1", 0.0)
    save_checkpoint(config.CKPT_BEST_PIPELINE, {
        "outer_iteration":    0,
        "inner_iteration":    0,
        "no_improve_count":   0,
        "current_best_score": best_score,
        "best_overkill_rate": best_overkill,
        "best_miss_rate":     best_miss,
        "best_accuracy":      best_accuracy,
        "best_f1":            best_f1,
        "best_pipeline_script": best_script,
        "ensemble_best_score": best_score,
        "ensemble_best_overkill": best_overkill,
        "ensemble_best_accuracy": best_accuracy,
        "ensemble_best_f1":    best_f1,
        "refinement_population": state.get("refinement_population", []),
        "token_count":        0,
        "stop_outer_loop":    False,
    })

    # ── Token counter — reset so flash downgrade doesn't bleed into next attempt
    state["token_count"] = 0

    # ── Phase 2 loop state ────────────────────────────────────────────────────
    state["outer_iteration"]  = 0
    state["inner_iteration"]  = 0
    state["no_improve_count"] = 0
    state["stop_outer_loop"]  = False
    state["best_overkill_rate"] = best_overkill   # carry forward — evaluator uses this as baseline
    state["best_miss_rate"] = best_miss
    state["best_accuracy"] = best_accuracy
    state["best_f1"] = best_f1
    state["ablation_results"] = []
    for i in range(NUM_ABLATION_VARIANTS):
        state[f"ablation_result_{i}"] = None
        state[f"ablation_script_{i}"] = ""
    state["diagnosis_report"] = ""
    state["refinement_plan"]  = ""
    state["current_script"]   = ""   # written by evaluator_agent.save_validated_script_fn
    state["error_analysis_report"]   = None
    state["latest_error_analysis"]   = None
    state["latest_error_analysis_path"] = ""
    # tried_approaches is intentionally NOT reset — the planner reads it to avoid
    # repeating failed strategies across retry cycles.
    state["latest_calibration_stats"] = {}
    state["latest_epoch_logs"]       = []
    state["diagnosis_brief"]         = {}

    # ── Phase 3 loop state ────────────────────────────────────────────────────
    state["ensemble_iteration"]    = 0
    state["ensemble_script"]       = ""
    state["ensemble_no_improve_count"] = 0  # must reset or Phase 3 exits immediately on retry
    # Seed from Phase 2 best so Phase 3 only accepts genuine improvements.
    state["ensemble_best_score"]   = best_score
    state["ensemble_best_overkill"] = best_overkill  # Phase 3 inherits Phase 2's overkill baseline
    state["ensemble_best_accuracy"] = best_accuracy
    state["ensemble_best_f1"] = best_f1

    # ── Phase 4 result state ──────────────────────────────────────────────────
    state["final_metrics"] = {}
    state["submission_result"] = {}
    state["submission_script"] = ""
    state["submission_source"] = ""

    logger.info(
        "Retry %d: checkpoints archived and state reset. "
        "best_score=%.4f, best_script=%s",
        attempt,
        best_score,
        "present" if best_script else "MISSING",
    )


def _append_attempt_log(attempt: int, pass_fail: dict, metrics: dict) -> None:
    """Append this attempt's result to checkpoints/submission_attempts.json."""
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if config.CKPT_SUBMISSION_ATTEMPTS.exists():
            existing = json.loads(
                config.CKPT_SUBMISSION_ATTEMPTS.read_text(encoding="utf-8")
            )
        existing.append({
            "attempt": attempt,
            "pass_fail": pass_fail,
            "metrics": metrics,
        })
        config.CKPT_SUBMISSION_ATTEMPTS.write_text(
            json.dumps(existing, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("Could not update submission_attempts.json: %s", e)


# ---------------------------------------------------------------------------
# RetryLoopAgent
# ---------------------------------------------------------------------------

class RetryLoopAgent(BaseAgent):
    """
    Runs phase2_refinement → phase3_ensemble → phase4_submission in a loop.

    Exits when the submission passes the relaxed minimum acceptance criteria
    or SUBMISSION_RETRY_MAX retry cycles are exhausted.

    sub_agents must be exactly:
        [phase2_refinement, phase3_ensemble, phase4_submission]
    """

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if len(self.sub_agents) != 3:
            raise ValueError(
                "RetryLoopAgent requires exactly 3 sub_agents: "
                "[phase2_refinement, phase3_ensemble, phase4_submission]"
            )

        phase2, phase3, phase4 = self.sub_agents
        state = ctx.session.state
        max_attempts = config.SUBMISSION_RETRY_MAX + 1  # 1 initial + N retries

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "RetryLoopAgent: attempt %d of %d — starting Phase 2 → 3 → 4",
                attempt,
                max_attempts,
            )

            # ── Phase 2 ───────────────────────────────────────────────────────
            # Phase 2 uses FunctionTool escalation to exit its inner and outer
            # loops. Strip the escalate flag before yielding upward so those
            # internal signals don't terminate this retry loop.
            async with Aclosing(phase2.run_async(ctx)) as agen:
                async for event in agen:
                    if event.actions.escalate:
                        yield event.model_copy(
                            update={
                                "actions": event.actions.model_copy(
                                    update={"escalate": None}
                                )
                            }
                        )
                    else:
                        yield event

            # ── Phase 3 ───────────────────────────────────────────────────────
            # Same treatment — strip Phase 3's internal escalation signals.
            async with Aclosing(phase3.run_async(ctx)) as agen:
                async for event in agen:
                    if event.actions.escalate:
                        yield event.model_copy(
                            update={
                                "actions": event.actions.model_copy(
                                    update={"escalate": None}
                                )
                            }
                        )
                    else:
                        yield event

            # ── Phase 4 ───────────────────────────────────────────────────────
            # Pass all Phase 4 events through unchanged — Phase 4 shouldn't
            # escalate, but we don't want to mask unexpected signals from it.
            async with Aclosing(phase4.run_async(ctx)) as agen:
                async for event in agen:
                    yield event

            # ── Check acceptance ──────────────────────────────────────────────
            submission_result = state.get("submission_result", {}) or {}
            pass_fail = submission_result.get("pass_fail", {}) or {}
            relaxed_minimum_pass = pass_fail.get("relaxed_minimum_pass", False)
            metrics = submission_result.get("metrics", {}) or {}
            reasons = pass_fail.get("reasons", [])

            _append_attempt_log(attempt, pass_fail, metrics)

            if relaxed_minimum_pass:
                logger.info(
                    "RetryLoopAgent: PASSED on attempt %d — pipeline complete.",
                    attempt,
                )
                return

            logger.warning(
                "RetryLoopAgent: attempt %d FAILED. "
                "Reasons: %s | ng_recall=%s miss_rate=%s accuracy=%s",
                attempt,
                reasons,
                metrics.get("ng_recall"),
                metrics.get("miss_rate"),
                metrics.get("accuracy"),
            )

            if attempt == max_attempts:
                logger.warning(
                    "RetryLoopAgent: retry budget exhausted after %d attempt(s). "
                    "Finishing with best available result.",
                    attempt,
                )
                return

            # Reset Phase 2/3/4 state and checkpoints for the next cycle.
            _reset_for_retry(state, attempt)
