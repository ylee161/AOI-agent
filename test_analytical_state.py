"""Tests for the deterministic Phase 2 analytical-state ladder.

Verifies the priority ladder (P0 miss_rate outranks P2 overkill), the relaxed ->
final tier distinction, and that empty / None state never raises.
"""

from mle_star_agent import config
from mle_star_agent.shared.analytical_state import (
    analytical_state_line,
    compute_analytical_state,
)


def test_miss_rate_above_target_wins_over_overkill():
    # (a) miss_rate above target -> P0 label even when overkill is ALSO bad.
    state = {
        "current_best_score": 0.95,   # ng_recall
        "best_miss_rate": 0.041,      # > 0.03 relaxed (P0)
        "best_overkill_rate": 0.30,   # > 0.08 relaxed (P2) — but P0 outranks
        "best_accuracy": 0.90,
    }
    result = compute_analytical_state(state)
    assert result["label"] == "OPTIMIZING_P0_MISS_RATE"
    assert "miss_rate=0.041" in result["detail"]
    assert "(P0)" in result["detail"]


def test_miss_ok_overkill_above_target():
    # (b) miss within relaxed budget but overkill above target -> P2 mitigation.
    state = {
        "current_best_score": 0.98,
        "best_miss_rate": 0.02,       # <= 0.03 relaxed
        "best_overkill_rate": 0.12,   # > 0.08 relaxed (P2)
        "best_accuracy": 0.95,
    }
    result = compute_analytical_state(state)
    assert result["label"] == "MITIGATING_P2_OVERKILL"
    assert "overkill_rate=0.120" in result["detail"]


def test_all_relaxed_met_pushes_to_final():
    # (c) all relaxed met (but not all final) -> push toward final tier.
    state = {
        "current_best_score": 0.98,   # >= 0.97 relaxed, < 1.00 final
        "best_miss_rate": 0.02,       # <= 0.03 relaxed, > 0.00 final
        "best_overkill_rate": 0.06,   # <= 0.08 relaxed, > 0.05 final
        "best_accuracy": 0.94,        # >= 0.92 relaxed, < 0.97 final
    }
    result = compute_analytical_state(state)
    assert result["label"] == "PUSHING_TO_FINAL_TARGET"
    assert "final" in result["detail"]


def test_all_final_met_holds():
    # (d) all final targets met -> hold.
    state = {
        "current_best_score": 1.00,   # ng_recall final
        "best_miss_rate": 0.00,       # final
        "best_overkill_rate": 0.04,   # <= 0.05 final
        "best_accuracy": 0.98,        # >= 0.97 final
    }
    result = compute_analytical_state(state)
    assert result["label"] == "HOLDING_ALL_TARGETS_MET"


def test_empty_and_none_state_never_raise():
    # (e) empty / None / malformed state never raises; defaults to P0.
    assert compute_analytical_state({})["label"] == "OPTIMIZING_P0_MISS_RATE"
    assert compute_analytical_state(None)["label"] == "OPTIMIZING_P0_MISS_RATE"
    assert compute_analytical_state(
        {"best_miss_rate": None, "current_best_score": None}
    )["label"] == "OPTIMIZING_P0_MISS_RATE"
    # Non-mapping inputs are tolerated too.
    assert compute_analytical_state(["not", "a", "mapping"])["label"] == "OPTIMIZING_P0_MISS_RATE"


def test_miss_rate_falls_back_to_ng_recall():
    # best_miss_rate absent -> miss_rate = max(0, 1 - ng_recall).
    # ng_recall=0.95 -> miss=0.05 > 0.03 relaxed -> P0.
    state = {"current_best_score": 0.95}
    result = compute_analytical_state(state)
    assert result["label"] == "OPTIMIZING_P0_MISS_RATE"


def test_ng_recall_gap_when_miss_and_overkill_ok():
    state = {
        "current_best_score": 0.95,   # < 0.97 relaxed (P1)
        "best_miss_rate": 0.01,       # <= 0.03 relaxed
        "best_overkill_rate": 0.05,   # <= 0.08 relaxed
        "best_accuracy": 0.95,
    }
    result = compute_analytical_state(state)
    assert result["label"] == "LIFTING_NG_RECALL"


def test_accuracy_gap_when_others_ok():
    state = {
        "current_best_score": 0.98,   # >= 0.97 relaxed
        "best_miss_rate": 0.01,       # <= 0.03 relaxed
        "best_overkill_rate": 0.05,   # <= 0.08 relaxed
        "best_accuracy": 0.90,        # < 0.92 relaxed (P4)
    }
    result = compute_analytical_state(state)
    assert result["label"] == "RAISING_ACCURACY"


def test_analytical_state_line_prefers_cached_snapshot():
    state = {"analytical_state": {"label": "CACHED_LABEL", "detail": "cached detail"}}
    line = analytical_state_line(state)
    assert line.startswith("ANALYTICAL_STATE: CACHED_LABEL — cached detail.")
    assert "Keep this as the active objective." in line


def test_analytical_state_line_computes_when_uncached():
    state = {"best_miss_rate": 0.041, "current_best_score": 0.95}
    line = analytical_state_line(state)
    assert "OPTIMIZING_P0_MISS_RATE" in line
    assert line.startswith("ANALYTICAL_STATE:")


def test_priority_thresholds_match_config():
    # Guardrail: the ladder reads the live config thresholds, not hardcoded copies.
    assert config.MISS_RATE_RELAXED_MAX == 0.03
    assert config.OVERKILL_RELAXED_MAX == 0.08
