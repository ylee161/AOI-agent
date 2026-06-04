"""Cross-branch code FUSION operator for Phase 2 refinement (MCGS Evolution/Fusion).

Normal refinement is a single chain: take the current best script and apply one
targeted change. The Pareto-style archive in ``state["refinement_population"]``
(maintained by ``evaluator_agent._update_refinement_population``) already holds a
handful of useful *divergent* branches — each its own script + metrics — but until
now it was only ever read as a metrics SUMMARY. Nothing fused code ACROSS members.

This module adds that. When the inner loop stagnates (the SAME signal the ideator
uses, :func:`loop_guard.should_restart_inner_for_stagnation`) AND the archive holds
at least two members, we arm a fusion attempt: the Coder is handed the TOP 2
population scripts (full text, on demand via a FunctionTool — never broadcast into
every turn) and instructed to keep one as the BASE and transplant ONLY the block
where the other (the DONOR) is measurably better. The result is one runnable
candidate that flows through the EXACT same validate -> evaluate path as any other
refinement attempt (predictive-abort, multiseed, acceptance scoring, population
update, persistent-KB write all apply unchanged).

Design split (to avoid an import cycle):
  * This module holds the dependency-light pure helpers (eligibility, top-member
    selection, pair fingerprint, directive text). It imports only ``loop_guard``.
  * The *consumption* of an armed fusion (which needs the planner's known-failed /
    dedup gate helpers) lives in ``planner_agent.ensure_selected_strategy_fn``,
    calling into the helpers here.
  * The *arming* is wired into ``evaluator_agent`` at the stagnation-restart branch.
"""

import hashlib
import json

from mle_star_agent.shared import loop_guard

# Marker that prefixes every fusion directive written to
# ``state["selected_refinement_strategy"]``. The coder keys on this to decide
# whether to call ``load_fusion_scripts_fn``; the fingerprint target_component
# mirrors it.
FUSION_DIRECTIVE_MARKER = "cross_branch_fusion"
FUSION_TARGET_COMPONENT = "fusion"


def _script_sha(member: dict) -> str:
    """Return a stable short hash for a population member.

    Prefers the precomputed ``script_sha256`` (written by the evaluator); falls
    back to hashing the script text so a hand-built member still fingerprints.
    """
    sha = (member.get("script_sha256") or "").strip()
    if not sha:
        sha = hashlib.sha256(
            (member.get("script", "") or "").encode("utf-8", errors="replace")
        ).hexdigest()
    return sha


def top_fusion_members(state: dict, k: int = 2) -> list[dict]:
    """Return the top ``k`` archive members (already sorted best-first by the
    evaluator: ascending overkill, then miss_rate, then higher accuracy/recall)."""
    population = [
        p for p in (state.get("refinement_population", []) or []) if isinstance(p, dict)
    ]
    return population[:k]


def fusion_fingerprint(member_a: dict, member_b: dict) -> dict:
    """Pair-specific fingerprint so the SAME two scripts are not re-fused.

    The mechanism class encodes the (order-independent) identity of the two
    members, so ``tried_approaches`` dedup and the KNOWN_FAILED gate operate per
    fusion PAIR rather than over a single generic "fusion" bucket.
    """
    a, b = sorted((_script_sha(member_a)[:8], _script_sha(member_b)[:8]))
    return {
        "target_component": FUSION_TARGET_COMPONENT,
        "mechanism_class": f"fusion_{a}_{b}",
    }


def should_trigger_fusion(state: dict) -> bool:
    """Fusion is only meaningful with >=2 diverse archived branches at a genuine
    inner-loop stagnation. Reuses the exact stagnation signal the ideator uses."""
    population = state.get("refinement_population", []) or []
    if len(population) < 2:
        return False
    return loop_guard.should_restart_inner_for_stagnation(state)


def arm_fusion_if_eligible(state: dict, target_outer: int) -> str:
    """Arm a fusion attempt for ``target_outer`` if eligible, at most once per outer.

    Called from the evaluator's stagnation-restart branch BEFORE the loop counters
    are reset (``should_trigger_fusion`` re-checks the stagnation signal, which
    depends on the ``no_improve_count`` the branch is about to clear). Sets
    ``state["pending_fusion"]`` for the strategy gate to consume next cycle, and a
    per-outer guard flag so the same outer iteration cannot arm fusion twice.
    """
    if not should_trigger_fusion(state):
        return "FUSION_NOT_ARMED: need >=2 population members at inner stagnation."
    flag = f"fusion_attempted_outer_{target_outer}"
    if state.get(flag):
        return "FUSION_NOT_ARMED: fusion already attempted for this outer iteration."
    state["pending_fusion"] = True
    state[flag] = True
    return (
        f"FUSION_ARMED: top-2 population members will be fused on outer={target_outer} "
        "(strategy gate will override the next refinement strategy with a fusion directive)."
    )


def _modality_clause(input_modality: str) -> str:
    if input_modality == "mono":
        return (
            "INPUT_MODALITY=mono: the dataset has a SINGLE image per sample (key `img`). "
            "NEVER introduce stereo handling — no `img_l`/`img_r` pair, no L/R difference, "
            "no Siamese/9-channel stereo input. Even if a source script contains stereo "
            "code, the fused script MUST stay mono throughout."
        )
    return (
        "INPUT_MODALITY=stereo: keep the existing stereo handling (both _L and _R images); "
        "do not drop a branch to a single image."
    )


def build_fusion_directive(state: dict, members: list[dict]) -> str:
    """Build the fusion directive written to ``state["selected_refinement_strategy"]``.

    Respects ``state["input_modality"]`` exactly and instructs the coder to produce
    ONE coherent script (one model, one training loop) — with an architecture-agnostic
    fallback when the two bases cannot be fused cleanly.
    """
    base_m = (members[0].get("metrics") or {}) if members else {}
    donor_m = (members[1].get("metrics") or {}) if len(members) > 1 else {}
    modality_clause = _modality_clause(state.get("input_modality", "stereo"))

    def _m(metrics: dict) -> str:
        return (
            f"overkill={metrics.get('overkill_rate')}, miss={metrics.get('miss_rate')}, "
            f"ng_recall={metrics.get('ng_recall')}, accuracy={metrics.get('accuracy')}"
        )

    return (
        f"{FUSION_DIRECTIVE_MARKER}: merge the WINNING code blocks of the top-2 "
        "refinement population members into ONE unified, runnable candidate.\n"
        "- Call `load_fusion_scripts_fn` FIRST to fetch both full scripts on demand "
        "(FUSION_SCRIPT_0 = BASE, FUSION_SCRIPT_1 = DONOR). Do not reconstruct them "
        "from memory.\n"
        f"- BASE = FUSION_SCRIPT_0 ({_m(base_m)}). Keep it as the skeleton: ONE model, "
        "ONE training loop. Preserve its architecture and data pipeline.\n"
        f"- DONOR = FUSION_SCRIPT_1 ({_m(donor_m)}). Identify the SINGLE block where the "
        "donor is measurably better — its loss / augmentation / calibration / threshold "
        "logic — and transplant ONLY that block onto the base.\n"
        "- Produce ONE coherent script. Do NOT stitch two incompatible backbones "
        "together. If the two bases are architecturally incompatible to fuse cleanly, "
        "FALL BACK to transplanting only the donor's data/loss/calibration/threshold "
        "block (architecture-agnostic) onto the base, leaving the base model untouched.\n"
        f"- {modality_clause}\n"
        "- The fused script must run end-to-end and print the standard METRICS line plus "
        "PROBE_METRICS / EPOCH_LOG / CALIBRATION_STATS / THRESHOLD_CURVE / PREDICTIONS, "
        "exactly like any normal refinement candidate."
    )


def render_fusion_scripts(state: dict, members: list[dict]) -> str:
    """Render the top-2 full scripts + metrics for the coder's on-demand fetch tool.

    Prepends the modality clause so the mono/stereo contract travels with the
    scripts and the coder cannot silently introduce stereo code for mono input.
    """
    modality_clause = _modality_clause(state.get("input_modality", "stereo"))
    parts = [modality_clause]
    for i, member in enumerate(members):
        label = "BASE" if i == 0 else "DONOR"
        metrics = member.get("metrics") or {}
        parts.append(
            f"=== FUSION_SCRIPT_{i} ({label}) sha={_script_sha(member)[:12]} "
            f"metrics={json.dumps(metrics, default=str)} ===\n"
            f"{member.get('script', '')}"
        )
    return "\n\n".join(parts)


__all__ = [
    "FUSION_DIRECTIVE_MARKER",
    "FUSION_TARGET_COMPONENT",
    "top_fusion_members",
    "fusion_fingerprint",
    "should_trigger_fusion",
    "arm_fusion_if_eligible",
    "build_fusion_directive",
    "render_fusion_scripts",
]
