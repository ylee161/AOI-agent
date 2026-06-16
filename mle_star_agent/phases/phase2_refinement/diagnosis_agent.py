import json as _json
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from mle_star_agent import config
from mle_star_agent.shared.callbacks import (
    _make_bypass_response,
    count_tokens_callback,
    log_context_size_callback,
    rate_limit_retry_callback,
)
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from mle_star_agent.shared.checkpoint_lineage import lineage_matches, stable_json_sha256
from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief

logger = logging.getLogger(__name__)


def _diagnosis_lineage(state: dict) -> dict:
    return {
        "ablation_results_sha256": stable_json_sha256(state.get("ablation_results", [])),
        "ablation_result_count": len(state.get("ablation_results", []) or []),
        "error_analysis_report_sha256": stable_json_sha256(state.get("error_analysis_report")),
    }


def _is_diagnosis_checkpoint_current(data: dict, state: dict) -> bool:
    return lineage_matches(data.get("lineage"), _diagnosis_lineage(state))

# ---------------------------------------------------------------------------
# Checkpoint gate FunctionTool
# ---------------------------------------------------------------------------


def check_diagnosis_checkpoint_fn(tool_context) -> str:
    """
    If diagnosis_{{N}}.json exists for the current outer_iteration, load it into
    state["diagnosis_report"] and return CHECKPOINT_FOUND.
    Otherwise return CHECKPOINT_NOT_FOUND.
    """
    n = int(tool_context.state.get("outer_iteration", 0))
    ckpt_path = config.ckpt_diagnosis(n)

    if not checkpoint_exists(ckpt_path):
        return f"CHECKPOINT_NOT_FOUND: no diagnosis checkpoint for outer_iteration={n}."

    data = load_checkpoint(ckpt_path)
    report = data.get("diagnosis_report", {})
    if not _is_diagnosis_checkpoint_current(data, tool_context.state):
        logger.warning("diagnosis_%d.json lineage mismatch — recomputing.", n)
        return (
            f"CHECKPOINT_STALE: diagnosis_{n}.json lineage does not match current "
            "ablation results. Proceeding to re-diagnose."
        )

    # Reject stale checkpoints written when ablation data was unavailable.
    # If ablation_ranking is empty but ablation_results in state has entries,
    # the diagnosis was computed without evidence and must be recomputed.
    ranking = report.get("ablation_ranking", [])
    ablation_results = tool_context.state.get("ablation_results", [])
    if not ranking and ablation_results:
        logger.warning(
            "diagnosis_%d.json has empty ablation_ranking but current ablation_results "
            "has %d entries — checkpoint is stale, recomputing.",
            n, len(ablation_results),
        )
        return (
            f"CHECKPOINT_STALE: diagnosis_{n}.json was written without ablation evidence "
            f"but {len(ablation_results)} ablation result(s) are now available. "
            "Proceeding to re-diagnose."
        )

    tool_context.state["diagnosis_report"] = report
    target = report.get("target_component", "unknown")
    return (
        f"CHECKPOINT_FOUND: diagnosis_{n}.json loaded. "
        f"Target component: '{target}'. Skipping re-diagnosis."
    )


_check_ckpt_tool = FunctionTool(func=check_diagnosis_checkpoint_fn)


def _bypass_diagnosis_ckpt(callback_context, llm_request):
    """Skip agent entirely when a current diagnosis checkpoint exists."""
    try:
        result = check_diagnosis_checkpoint_fn(callback_context)
        if result.startswith("CHECKPOINT_FOUND"):
            return _make_bypass_response(result)
        return None  # checkpoint absent or stale → LLM must run scorer + reason + write
    except Exception as exc:
        logger.warning("bypass_diagnosis_ckpt: exception — falling back to LLM: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Write-report FunctionTool
# ---------------------------------------------------------------------------


def write_diagnosis_report_fn(
    tool_context,
    target_component: str,
    impact_summary: str,
    recommended_changes: str,
    ablation_ranking: str,
    prediction_contract: str = "",
) -> str:
    """
    Persist the diagnosis report to state["diagnosis_report"] and save
    checkpoints/diagnosis_{{N}}.json.

    Args:
        target_component: Name of the pipeline component to improve next
            (e.g. "stereo_fusion", "weighted_loss", "threshold_sweep", "augmentation",
             or "model_architecture" / "training_schedule" for non-ablated targets).
        impact_summary: One-paragraph summary of which ablation variant caused the
            biggest performance drop and why.
        recommended_changes: Concrete changes to make to the target component in the
            next refinement cycle (what to try, why it should help).
        ablation_ranking: JSON-serialisable string listing all variants ranked by
            impact (worst first), e.g. '[{"name": "no_stereo_fusion", "delta_miss_rate": 0.12}, ...]'
        prediction_contract: JSON-serialisable string with falsifiable expected
            metric constraints, e.g.
            '{"expected_overkill_rate_max": 0.25, "expected_ng_recall_min": 0.93}'.
    """
    n = int(tool_context.state.get("outer_iteration", 0))

    try:
        ranking = _json.loads(ablation_ranking)
    except Exception:
        ranking = ablation_ranking  # keep as string if not valid JSON

    brief = tool_context.state.get("diagnosis_brief") or {}
    report = {
        "outer_iteration": n,
        "target_component": target_component,
        "impact_summary": impact_summary,
        "recommended_changes": recommended_changes,
        "ablation_ranking": ranking,
        "failure_classification": brief.get("failure_classification"),
    }
    if prediction_contract:
        try:
            prediction = _json.loads(prediction_contract)
        except Exception:
            prediction = {"raw": prediction_contract}
        report["prediction"] = prediction

    error_analysis_evidence = tool_context.state.get("error_analysis_report")
    if isinstance(error_analysis_evidence, dict):
        report["error_analysis_evidence"] = {
            "dominant_failure": error_analysis_evidence.get("dominant_failure"),
            "recommended_target_component": error_analysis_evidence.get(
                "recommended_target_component"
            ),
            "evidence_available": error_analysis_evidence.get("evidence_available"),
            "fp_count": error_analysis_evidence.get("fp_count"),
            "fn_count": error_analysis_evidence.get("fn_count"),
            "metrics_consistency": error_analysis_evidence.get("metrics_consistency"),
        }

    tool_context.state["diagnosis_report"] = report

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.ckpt_diagnosis(n)
    save_checkpoint(ckpt_path, {
        "lineage": _diagnosis_lineage(tool_context.state),
        "diagnosis_report": report,
    })

    logger.info(
        "Diagnosis report written for outer_iteration=%d, target='%s'",
        n,
        target_component,
    )
    return (
        f"Diagnosis report saved to diagnosis_{n}.json. "
        f"Target component: '{target_component}'."
    )


_write_report_tool = FunctionTool(func=write_diagnosis_report_fn)

# ---------------------------------------------------------------------------
# Deterministic pre-scorer FunctionTool (P3)
# ---------------------------------------------------------------------------

def compute_diagnosis_scores_fn(tool_context) -> str:
    """Deterministic scorer: computes ablation deltas and classifies failure mode
    in Python, saving LLM tokens. Call BEFORE writing the diagnosis report."""
    ablation_results = tool_context.state.get("ablation_results", [])
    ng_recall = float(tool_context.state.get("current_best_score", 0.0) or 0.0)
    saved_miss_rate = tool_context.state.get("best_miss_rate")
    baseline_metrics = {
        "accuracy": float(tool_context.state.get("best_accuracy", 0.0) or 0.0),
        "ng_recall": ng_recall,
        "miss_rate": float(saved_miss_rate) if saved_miss_rate is not None else max(0.0, 1.0 - ng_recall),
        "overkill_rate": float(tool_context.state.get("best_overkill_rate", 1.0) or 1.0),
        "f1": float(tool_context.state.get("best_f1", 0.0) or 0.0),
    }
    calibration_stats = tool_context.state.get("latest_calibration_stats")
    error_analysis = tool_context.state.get("latest_error_analysis")
    threshold_curve = tool_context.state.get("latest_threshold_curve")
    input_modality = tool_context.state.get("input_modality", "stereo")

    brief = generate_diagnosis_brief(
        ablation_results, baseline_metrics, calibration_stats, error_analysis, threshold_curve,
        input_modality=input_modality,
    )
    tool_context.state["diagnosis_brief"] = brief
    return _json.dumps(brief, indent=2, default=str)

_compute_scores_tool = FunctionTool(func=compute_diagnosis_scores_fn)

# ---------------------------------------------------------------------------
# Diagnosis agent instruction
# ---------------------------------------------------------------------------

_INSTRUCTION = """You are the Diagnosis Agent for Phase 2 Refinement.

Your role is to use the deterministic scorer to analyse ablation results, then write
a structured diagnosis report that guides the Refinement Coder Agent.

---
## STEP 1 — Check for existing checkpoint

Call `check_diagnosis_checkpoint_fn` immediately.
- If it returns CHECKPOINT_FOUND: report the loaded target component and stop.
- If it returns CHECKPOINT_NOT_FOUND or CHECKPOINT_STALE: proceed to STEP 2.

---
## STEP 2 — Compute deterministic diagnosis scores

Call `compute_diagnosis_scores_fn`. This tool performs all arithmetic in Python:
- Computes ablation deltas for each variant vs baseline
- Ranks variants by acceptance-distance impact
- Classifies the failure mode using this decision tree:

  overkill > 0.40?
    G_prob_mean > 0.50 → threshold_collapse → target asymmetric loss + constrained threshold
    G_prob_mean 0.20–0.50 → g_ng_overlap → target stereo difference features
    G_prob_mean < 0.20 → lot_shift_noise → target lot-level normalization
    No calibration stats → g_ng_overlap (low confidence) → improve features + add calibration output

  miss_rate > 0.03, overkill OK → low_capacity_miss → increase model capacity
  acceptance_distance < 0.5 → near_acceptance → calibration + threshold fine-tuning

The tool returns a JSON brief with: failure_classification, ablation_ranking,
recommended_target, recommended_action.

---
## STEP 3 — Review and refine

Read the returned brief. You do NOT need to recompute any deltas — the scorer
already did that. Your job is to:
1. Verify the recommended_target makes sense given the ablation ranking
2. Write 3-5 concrete implementation bullet points for the refinement coder
3. If the failure_classification confidence is "low", consider whether a different
   target would be more effective based on the ablation ranking

---
## STEP 4 — Write the diagnosis report

Call `write_diagnosis_report_fn` with ALL five arguments:

1. `target_component` — use the scorer's recommended_target unless you have
   strong evidence to override it
2. `impact_summary` — one paragraph: which variant matters most and why.
   Include the failure_mode from the scorer.
3. `recommended_changes` — 3-5 concrete bullet points as a single string
4. `ablation_ranking` — JSON array string from the scorer's ablation_ranking
5. `prediction_contract` — JSON object string with falsifiable expected metrics.
   Include at least:
   - `expected_overkill_rate_max`
   - `expected_ng_recall_min`
   - `expected_miss_rate_max`
   - `failure_if`

ANCHOR the prediction to the CURRENT BEST metrics in the brief, not to the acceptance
targets. One refinement step on this dataset typically moves a metric by 0.02-0.10;
predicting near-target numbers (e.g. recall >= 0.93 when the best is 0.60) makes the
contract fail every iteration and tells the planner nothing. Set:
- `expected_ng_recall_min` ≈ current best ng_recall (a floor the change must not break),
- `expected_miss_rate_max` ≈ current best miss_rate + 0.02,
- `expected_overkill_rate_max` ≈ what the targeted change should plausibly reach.
Example with best ng_recall=0.60, overkill=0.35:
`{"expected_overkill_rate_max":0.30,"expected_ng_recall_min":0.58,
"expected_miss_rate_max":0.44,"failure_if":"recall drops below 0.58 or overkill stays above 0.30"}`.

---
## STEP 5 — Summarise

After the tool call, provide a brief summary:
- Failure mode classification and confidence
- Target component and why it was chosen
- Top recommended change
"""

# ---------------------------------------------------------------------------
# Diagnosis agent
# ---------------------------------------------------------------------------

diagnosis_agent = LlmAgent(
    name="diagnosis_agent",
    # Flash (thinking disabled): the deterministic scorer (compute_diagnosis_scores_fn
    # -> generate_diagnosis_brief) already computes all ablation deltas, the variant
    # ranking, and the full failure-mode decision tree in Python. The LLM here only
    # reads that pre-computed brief, verifies the recommended target, writes a few
    # concrete bullets, and emits a quantitative prediction_contract — a read-and-route
    # task that does not need the 16k MODEL_PRO thinking budget.
    model=config.MODEL,
    description=(
        "Analyses ablation results to identify the most impactful pipeline component "
        "and writes state['diagnosis_report'] with a targeted refinement plan. "
        "Saves checkpoints/diagnosis_{{N}}.json."
    ),
    instruction=_INSTRUCTION,
    tools=[_check_ckpt_tool, _compute_scores_tool, _write_report_tool],
    include_contents="none",
    before_model_callback=[_bypass_diagnosis_ckpt, log_context_size_callback],
    after_model_callback=count_tokens_callback,
    on_model_error_callback=rate_limit_retry_callback,
)
