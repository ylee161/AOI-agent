"""Dynamic ideation injection for Phase 2 refinement (Fix #1).

When the inner refinement loop stagnates below relaxed acceptance, the evaluator
has historically just kicked the outer loop back to a fresh ablation/diagnosis
cycle — but the *idea pool* the planner draws from stays frozen at whatever
A_retriever pulled once in Phase 1. This module breaks that stall by fetching a
small batch of FRESH small-data vision-classification technique ideas from arXiv,
keyed to the CURRENT failure mode, and overwriting
``state["retrieved_technique_hints"]`` so the planner (planner_agent.py, which
already reads that key as advisory context) sees new options on the next cycle.

Design notes
------------
* This is a ``FunctionTool``-backed helper, NOT a new LoopAgent member. It is
  called programmatically from ``evaluator_agent.evaluate_and_update_fn`` at the
  stagnation-restart branch via :func:`trigger_ideation`. The ``FunctionTool``
  wrapper (:data:`ideator_tool`) exists so the same logic is also invocable as an
  ADK tool if ever wired into an agent.
* The ideation core (:func:`generate_technique_hints`) receives ONLY the failure
  mode string and the metric snapshot (overkill_rate, miss_rate, ng_recall) — it
  never sees the full state — so the search query is built from the minimal,
  decision-relevant signal and nothing else.
* arXiv is queried via the public Atom API over ``httpx`` (the library already
  used by phase1_init/retriever_agent.py). No API key is needed. On any
  network/parse failure it degrades gracefully to a curated, failure-mode-keyed
  fallback list so the loop is never blocked and the key is always populated.
* The hints are ADVISORY only. Anything the planner derives from them still has
  to pass the KNOWN_FAILED_STRATEGY_FINGERPRINTS gate in
  ``save_strategy_candidates_fn`` — this module deliberately does not pre-vet fit.
"""

import logging
import re
import xml.etree.ElementTree as ET

import httpx
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_SEARCH_TIMEOUT = 20.0
_MAX_RESULTS = 5      # arXiv entries fetched per trigger
_MIN_HINTS = 3        # spec: populate 3-5 hints
_MAX_HINTS = 5
_HINT_MAX_CHARS = 240

# Per-failure-mode arXiv search terms. The failure_mode strings come from
# shared/diagnosis_scorer.classify_failure_mode (full_freeze_underfit,
# low_capacity_miss, g_ng_overlap, threshold_collapse, preprocessing_lot_shift,
# near_acceptance). Each maps to the small-data technique vocabulary most likely
# to surface a NOVEL lever for that specific stall.
_FAILURE_MODE_TERMS = {
    "full_freeze_underfit": (
        "partial fine-tuning transfer learning small dataset underfitting "
        "layer unfreezing"
    ),
    "low_capacity_miss": (
        "recall sensitivity rare defect detection imbalanced small data "
        "cost sensitive learning"
    ),
    "g_ng_overlap": (
        "fine-grained feature separability metric learning small data "
        "contrastive representation defect"
    ),
    "threshold_collapse": (
        "probability calibration decision threshold uncertainty small data "
        "binary classification"
    ),
    "preprocessing_lot_shift": (
        "domain shift normalization batch effect industrial inspection "
        "covariate shift correction"
    ),
    "near_acceptance": (
        "calibration regularization small data fine-tuning generalization "
        "binary classification"
    ),
}

# Generic anchor terms always folded into the query so even an unknown
# failure_mode still pulls relevant small-data vision-classification work.
_BASE_TERMS = "small data image classification overfitting regularization augmentation"

# Curated fallback hints, keyed by failure_mode, used only when the live arXiv
# fetch fails or returns nothing. They are honest small-data techniques aligned
# with the project's small-data-safe priorities and deliberately avoid the banned
# fingerprints (no larger backbone, no global-feature-difference-only, no
# two-independent-backbone stereo, no threshold-only fixes).
_FALLBACK_HINTS = {
    "full_freeze_underfit": [
        "Partial unfreeze of only the final ResNet block (layer4) with a small "
        "dropout head, AdamW, and nonzero weight decay to escape full-freeze underfit",
        "Discriminative layer-wise learning rates (lower for early layers, higher "
        "for the head) when partially unfreezing a pretrained backbone on small data",
        "MixUp / CutMix on the unfrozen-backbone setup to add regularization while "
        "increasing effective capacity",
        "Stochastic depth / drop-path on the unfrozen block to regularize the added "
        "capacity on a few-hundred-sample set",
    ],
    "low_capacity_miss": [
        "Asymmetric / cost-sensitive loss that up-weights NG (defect) misses to lift "
        "recall without a larger backbone",
        "Focal loss (alpha tuned toward the defect class, gamma=2) to focus learning "
        "on hard, easily-missed defects",
        "Class-balanced oversampling of NG samples in the DataLoader to raise "
        "sensitivity on the minority defect class",
        "Hard-negative-aware mining of near-miss defect crops to concentrate capacity "
        "on the recall-critical region",
    ],
    "g_ng_overlap": [
        "Supervised contrastive pretraining of the head to pull G and NG embeddings "
        "apart before the binary classifier",
        "Local/patch or ROI evidence aggregation (MIL-style) to expose sub-resolution "
        "defects that wash out in a global feature",
        "Metric-learning triplet head on frozen features to widen the G/NG margin on "
        "small data",
        "Spatial-attention pooling on backbone features to localize the defect signal "
        "instead of averaging it away",
    ],
    "threshold_collapse": [
        "Temperature scaling on logits (post-hoc Platt scaling) to recover a usable "
        "operating point after probability collapse",
        "Label smoothing during training to reduce overconfidence and restore a "
        "separable probability range",
        "Isotonic-regression calibration on validation probabilities for a "
        "non-parametric fix to a collapsed threshold curve",
        "Deep-ensemble or MC-dropout probability averaging to widen the G/NG "
        "probability gap before thresholding",
    ],
    "preprocessing_lot_shift": [
        "Per-lot mean/std normalization computed from the train split only, to remove "
        "lot-to-lot brightness/contrast shift",
        "CLAHE contrast equalization applied identically across the stereo pair before "
        "fusion to stabilize lot variation",
        "Histogram matching each image to a reference-lot distribution to suppress "
        "covariate shift across lots",
        "Test-time augmentation averaging to reduce sensitivity to per-lot capture "
        "variation",
    ],
    "near_acceptance": [
        "Test-time augmentation (TTA) averaging to squeeze the last points of accuracy "
        "near the acceptance boundary",
        "Stochastic Weight Averaging (SWA) over the final epochs to flatten the minimum "
        "and improve small-data generalization",
        "Temperature scaling to fine-tune the operating point once metrics are close to "
        "the relaxed target",
        "Light AOI-safe augmentation (geometric transforms applied identically to L and "
        "R) to regularize the near-accepted pipeline",
    ],
}

# Used when failure_mode is empty/unknown and the live fetch also fails.
_GENERIC_FALLBACK_HINTS = [
    "Partial-unfreeze of the final backbone block with a small regularized head "
    "(AdamW + weight decay) for small-data fine-tuning",
    "Asymmetric / focal loss to trade overkill against defect recall without adding "
    "model capacity",
    "Temperature scaling or isotonic calibration to fix the probability operating "
    "point post-training",
    "MixUp / CutMix and AOI-safe geometric augmentation to regularize a "
    "few-hundred-sample binary defect set",
]


def _clean(text: str) -> str:
    """Collapse arXiv title/summary whitespace into a single compact line."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _split_failure_mode_payload(failure_mode: str) -> tuple[str, str]:
    """Return the canonical failure_mode plus any seek-help query emphasis."""
    raw = (failure_mode or "").strip()
    marker = "| planner roadblock:"
    if marker not in raw:
        return raw.lower(), ""
    mode, emphasis = raw.split(marker, 1)
    return mode.strip().lower(), _clean(emphasis)


def _build_query(failure_mode: str, overkill_rate: float, miss_rate: float) -> str:
    """Build the arXiv search_query from the failure mode + metric emphasis.

    The metric snapshot nudges the query toward the dominant failure axis: a high
    overkill rate biases toward false-positive/precision work, a high miss rate
    biases toward recall/sensitivity work.
    """
    mode, query_emphasis = _split_failure_mode_payload(failure_mode)
    mode_terms = _FAILURE_MODE_TERMS.get(mode, "")
    query_emphasis = re.sub(r"[^A-Za-z0-9_\- ]+", " ", query_emphasis)

    metric_terms = ""
    if overkill_rate >= miss_rate and overkill_rate > 0.0:
        metric_terms = "false positive reduction precision"
    elif miss_rate > 0.0:
        metric_terms = "recall sensitivity miss rate reduction"

    raw = " ".join(t for t in (_BASE_TERMS, mode_terms, metric_terms, query_emphasis) if t)
    # arXiv `all:` field with '+'-joined terms (an OR-ish keyword bag).
    return "all:" + "+".join(raw.split())


def _fetch_arxiv_hints(query: str) -> list[str]:
    """Query the arXiv Atom API and distill entries into hint strings.

    Returns [] on any network/parse failure so the caller can fall back. Never
    raises — this runs inside the evaluator's hot path and must not break the loop.
    """
    try:
        resp = httpx.get(
            _ARXIV_API,
            params={
                "search_query": query,
                "start": 0,
                "max_results": _MAX_RESULTS,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=_SEARCH_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:  # network, HTTP, or XML parse failure
        logger.warning("ideator: arXiv fetch failed for %r: %s", query, exc)
        return []

    hints: list[str] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = _clean(entry.findtext(f"{_ATOM_NS}title", default=""))
        summary = _clean(entry.findtext(f"{_ATOM_NS}summary", default=""))
        if not title:
            continue
        # First sentence of the abstract is usually the technique's thesis.
        first_sentence = re.split(r"(?<=[.!?])\s+", summary)[0] if summary else ""
        hint = f"{title} — {first_sentence}".strip(" —") if first_sentence else title
        if len(hint) > _HINT_MAX_CHARS:
            hint = hint[: _HINT_MAX_CHARS - 1].rstrip() + "…"
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) >= _MAX_HINTS:
            break
    return hints


def generate_technique_hints(
    failure_mode: str,
    overkill_rate: float,
    miss_rate: float,
    ng_recall: float,
) -> list[str]:
    """Ideation core: failure_mode + metric snapshot -> 3-5 technique hint strings.

    Receives ONLY the failure mode and the metric snapshot — never the full state.
    Tries a live, failure-mode-keyed arXiv search first; falls back to a curated
    failure-mode list if the fetch yields too few results. Always returns between
    :data:`_MIN_HINTS` and :data:`_MAX_HINTS` non-empty, de-duplicated strings.
    """
    mode, _query_emphasis = _split_failure_mode_payload(failure_mode)
    query = _build_query(failure_mode, overkill_rate, miss_rate)
    logger.info(
        "ideator: failure_mode=%r overkill=%.3f miss=%.3f ng_recall=%.3f query=%r",
        mode or "(unknown)", overkill_rate, miss_rate, ng_recall, query,
    )

    hints = _fetch_arxiv_hints(query)
    source = "arxiv"

    if len(hints) < _MIN_HINTS:
        fallback = _FALLBACK_HINTS.get(mode, _GENERIC_FALLBACK_HINTS)
        source = "arxiv+fallback" if hints else "fallback"
        for h in fallback:
            if len(hints) >= _MIN_HINTS:
                break
            if h not in hints:
                hints.append(h)

    hints = hints[:_MAX_HINTS]
    logger.info("ideator: returning %d hint(s) via %s", len(hints), source)
    return hints


def ideate_technique_hints_fn(tool_context) -> str:
    """Refresh ``state["retrieved_technique_hints"]`` with fresh, failure-mode-keyed
    technique ideas pulled from arXiv.

    Reads ONLY the current failure_mode and the metric snapshot (overkill_rate,
    miss_rate, ng_recall) out of state, queries arXiv for novel small-data vision
    classification techniques matching that failure mode, and OVERWRITES
    ``state["retrieved_technique_hints"]`` with a list of 3-5 hint strings. The
    Phase 2 planner already reads that key as advisory context.
    """
    state = tool_context.state

    # failure_mode lives under failure_classification in diagnosis_brief (primary)
    # or diagnosis_report (mirror). Read only that one field.
    failure_mode = ""
    for source_key in ("diagnosis_brief", "diagnosis_report"):
        blob = state.get(source_key) or {}
        if isinstance(blob, dict):
            classification = blob.get("failure_classification") or {}
            if isinstance(classification, dict):
                failure_mode = (classification.get("failure_mode") or "").strip()
                if failure_mode:
                    break

    # Metric snapshot: ng_recall == current_best_score; miss_rate falls back to
    # 1 - ng_recall when not separately persisted (same convention as loop_guard).
    ng_recall = float(state.get("current_best_score", 0.0) or 0.0)
    overkill_rate = float(state.get("best_overkill_rate", 1.0) or 1.0)
    saved_miss = state.get("best_miss_rate")
    miss_rate = float(saved_miss) if saved_miss is not None else max(0.0, 1.0 - ng_recall)

    hints = generate_technique_hints(failure_mode, overkill_rate, miss_rate, ng_recall)
    state["retrieved_technique_hints"] = hints

    return (
        f"IDEATION_INJECTED: refreshed state['retrieved_technique_hints'] with "
        f"{len(hints)} technique hint(s) for failure_mode="
        f"{failure_mode or '(unknown)'} (overkill={overkill_rate:.3f}, "
        f"miss_rate={miss_rate:.3f}, ng_recall={ng_recall:.3f}).\n- "
        + "\n- ".join(hints)
    )


def _read_failure_mode_and_metrics(state) -> tuple[str, float, float, float]:
    """Read failure_mode plus metric snapshot from state, tolerating empty state."""
    if not hasattr(state, "get"):
        return "", 0.0, 1.0, 1.0

    failure_mode = ""
    for source_key in ("diagnosis_brief", "diagnosis_report"):
        blob = state.get(source_key) or {}
        if isinstance(blob, dict):
            classification = blob.get("failure_classification") or {}
            if isinstance(classification, dict):
                failure_mode = (classification.get("failure_mode") or "").strip()
                if failure_mode:
                    break

    ng_recall = float(state.get("current_best_score", 0.0) or 0.0)
    overkill_rate = float(state.get("best_overkill_rate", 1.0) or 1.0)
    saved_miss = state.get("best_miss_rate")
    miss_rate = float(saved_miss) if saved_miss is not None else max(0.0, 1.0 - ng_recall)
    return failure_mode, overkill_rate, miss_rate, ng_recall


def seek_help_fn(tool_context, problem: str) -> str:
    """Return fresh on-demand technique ideas without clobbering planner hints.

    This is a read-oriented planner helper. It keys ideation to the current
    failure_mode and metric snapshot, folds the planner's specific roadblock into
    the arXiv query emphasis, stores the most recent response under
    ``state["seek_help_hints"]``, and deliberately leaves
    ``state["retrieved_technique_hints"]`` untouched.
    """
    try:
        state = getattr(tool_context, "state", None)
        failure_mode, overkill_rate, miss_rate, ng_recall = _read_failure_mode_and_metrics(state)
        problem_text = _clean(str(problem or ""))
        failure_mode_with_emphasis = failure_mode
        if problem_text:
            failure_mode_with_emphasis = (
                f"{failure_mode or 'unknown'} | planner roadblock: {problem_text}"
            )

        hints = generate_technique_hints(
            failure_mode_with_emphasis,
            overkill_rate,
            miss_rate,
            ng_recall,
        )

        if hasattr(state, "__setitem__"):
            state["seek_help_hints"] = hints

        return (
            f"SEEK_HELP_HINTS: {len(hints)} fresh on-demand hint(s) for "
            f"failure_mode={failure_mode or '(unknown)'} (overkill={overkill_rate:.3f}, "
            f"miss_rate={miss_rate:.3f}, ng_recall={ng_recall:.3f}). "
            "These are advisory only; any derived strategy must still pass the "
            "failed-fingerprint gate.\n- "
            + "\n- ".join(hints)
        )
    except Exception as exc:  # pragma: no cover - defensive; tool must not break planning
        logger.warning("ideator: seek_help failed: %s", exc)
        return f"SEEK_HELP_UNAVAILABLE: {type(exc).__name__}: {exc}"


# FunctionTool wrapper (satisfies the "FunctionTool-backed helper" contract and
# allows the same logic to be wired into an ADK agent if ever needed).
ideator_tool = FunctionTool(func=ideate_technique_hints_fn)
seek_help_tool = FunctionTool(func=seek_help_fn)


def trigger_ideation(tool_context) -> str:
    """Programmatic entry point for the evaluator.

    ``evaluate_and_update_fn`` is itself a FunctionTool body, so it cannot invoke
    another tool through the LLM — it calls this directly. Thin wrapper over
    :func:`ideate_technique_hints_fn` that never raises, so a transient ideation
    failure can never break the evaluator's loop-management path.
    """
    try:
        return ideate_technique_hints_fn(tool_context)
    except Exception as exc:  # pragma: no cover - defensive; loop must not break
        logger.warning("ideator: trigger_ideation failed: %s", exc)
        return f"IDEATION_SKIPPED: ideation failed ({type(exc).__name__}: {exc})."


__all__ = [
    "ideator_tool",
    "seek_help_tool",
    "seek_help_fn",
    "ideate_technique_hints_fn",
    "generate_technique_hints",
    "trigger_ideation",
]
