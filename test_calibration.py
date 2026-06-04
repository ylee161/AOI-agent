"""Tests for isotonic calibration + fine-grained (0.01-step) threshold search.

Mission: isotonic-calibration-fine-threshold.

The pipeline labels are binary G/NG (no defect sub-classes), so the threshold is a
single float. These tests cover the harness contract and the two ideas the prompt
now asks generated scripts to apply:

  (a) isotonic calibration maps scores into [0, 1],
  (b) a 0.01-step sweep resolves a strictly better operating point than 0.05 can,
  (c) metrics_parser parses the (calibrated) threshold back as a plain float, and
  (d) acceptance_scoring.passes_relaxed_acceptance is unchanged.
"""

import json

from sklearn.isotonic import IsotonicRegression

from mle_star_agent.shared.acceptance_scoring import passes_relaxed_acceptance
from mle_star_agent.shared.metrics_parser import parse_metrics


# Threshold grids over the SAME 0.10–0.90 range; only the step differs.
_COARSE_GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # step 0.05
_FINE_GRID = [round(0.10 + 0.01 * i, 2) for i in range(81)]    # step 0.01


def _confusion(data, t):
    """data: list of (probability, label) with label 1=NG, 0=G. Returns (miss, overkill)."""
    fn = sum(1 for p, y in data if y == 1 and p < t)
    tp = sum(1 for p, y in data if y == 1 and p >= t)
    fp = sum(1 for p, y in data if y == 0 and p >= t)
    tn = sum(1 for p, y in data if y == 0 and p < t)
    ng, g = tp + fn, tn + fp
    miss = fn / ng if ng else 0.0
    overkill = fp / g if g else 0.0
    return miss, overkill


def _best_threshold(data, grid, overkill_budget):
    """Pick the threshold minimising miss_rate subject to overkill <= budget.

    Falls back to the min-overkill threshold when none satisfy the budget.
    Returns (threshold, miss_rate, overkill_rate).
    """
    scored = [(t, *_confusion(data, t)) for t in grid]
    feasible = [c for c in scored if c[2] <= overkill_budget]
    pool = feasible if feasible else scored
    return min(pool, key=lambda c: (c[1], c[2], c[0]))


def _metrics_stdout(threshold, cm=None) -> str:
    cm = cm or {"tp": 100, "fn": 2, "tn": 100, "fp": 5}
    payload = {
        "accuracy": 0.966, "ng_recall": 0.980, "miss_rate": 0.0196,
        "overkill_rate": 0.0476, "f1": 0.95, "avg_latency_ms": 12.0,
        "threshold": threshold,
        "ng_count": cm["tp"] + cm["fn"], "g_count": cm["tn"] + cm["fp"],
        "tp": cm["tp"], "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"],
        "roc_auc": 0.99, "prob_gap": 0.41,
    }
    return "training log\nMETRICS: " + json.dumps(payload) + "\n"


# ── (a) isotonic calibration produces probabilities in [0, 1] ────────────────

def test_isotonic_calibration_outputs_in_unit_interval():
    raw_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]  # 1=NG, 0=G

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, labels)

    # Include values OUTSIDE the training range to exercise out_of_bounds="clip".
    calibrated = iso.transform([-0.5, 0.0, 0.05, 0.37, 0.99, 1.5])
    assert all(0.0 <= v <= 1.0 for v in calibrated)
    # Monotonic non-decreasing, as isotonic regression guarantees.
    assert list(calibrated) == sorted(calibrated)


# ── (b) fine-step (0.01) sweep resolves finer than 0.05 ──────────────────────

def test_fine_sweep_resolves_better_operating_point_than_coarse():
    # Crafted CALIBRATED-probability scenario, no-FP budget (overkill must be 0):
    #   one NG at 0.39 and one G at 0.38 sit between the 0.05 grid points 0.35/0.40.
    #   A 0.39 threshold (only on the fine grid) catches the NG while excluding the G;
    #   the coarse grid must choose 0.40 (misses the NG) to keep overkill at 0.
    data = [
        (0.39, 1), (0.55, 1), (0.60, 1), (0.70, 1), (0.80, 1),  # NG
        (0.38, 0), (0.05, 0), (0.06, 0), (0.07, 0), (0.08, 0),  # G
    ]
    budget = 0.0

    coarse_t, coarse_miss, _ = _best_threshold(data, _COARSE_GRID, budget)
    fine_t, fine_miss, fine_overkill = _best_threshold(data, _FINE_GRID, budget)

    # The fine grid can land on 0.39; the coarse grid cannot.
    assert 0.39 in _FINE_GRID and 0.39 not in _COARSE_GRID
    # And that finer resolution yields a strictly lower miss_rate at the same budget.
    assert fine_t == 0.39
    assert fine_miss < coarse_miss
    assert fine_miss == 0.0 and fine_overkill == 0.0


# ── (c) metrics_parser parses the calibrated threshold as a plain float ──────

def test_parser_reads_calibrated_threshold_as_float():
    metrics = parse_metrics(_metrics_stdout(0.37))
    assert metrics is not None
    assert isinstance(metrics.threshold, float)
    assert metrics.threshold == 0.37
    # No structural change: there is no per-class field on the dataclass.
    assert not hasattr(metrics, "per_class_threshold")


# ── (d) acceptance_scoring.passes_relaxed_acceptance is unchanged ────────────

def test_passes_relaxed_acceptance_unchanged():
    passing = parse_metrics(_metrics_stdout(0.37))
    assert passes_relaxed_acceptance(passing) is True

    # Fails P0 (miss_rate well above the 0.03 relaxed budget).
    failing = parse_metrics(_metrics_stdout(0.37, cm={"tp": 80, "fn": 22, "tn": 90, "fp": 15}))
    assert passes_relaxed_acceptance(failing) is False
