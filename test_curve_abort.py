"""Tests for power-law learning-curve extrapolation ("curve abort").

The debug micro-run now emits a SHORT per-epoch curve (config.CURVE_ABORT_DEBUG_EPOCHS
epochs on 5% data) instead of one noisy epoch. The evaluator fits a saturating
power-law y = a + b * t^(-c) to that val_ng_recall curve and prunes the full run
ONLY when the projected final is CONFIDENTLY worse than the best NG recall so far
AND the fit is trustworthy. Everything else falls through to the full run.

The evaluator additionally projects the per-epoch val_overkill curve and prunes
ONLY when BOTH the ng_recall projection AND the overkill projection are
confidently worse than the best (AND logic): a doomed overkill trajectory with a
healthy ng_recall trajectory (or vice-versa) still gets the full run, since a
single bad axis is often threshold-fixable.

Covered here:
  - rising-then-plateau curve -> projects HIGH (no abort),
  - monotonically-bad curve   -> projects LOW with a good fit (abort),
  - noisy / 2-point curve      -> no usable projection (no abort),
  - the config gates are conservative,
  - the debug micro-run is patched to the multi-epoch cap,
  - end-to-end wiring: a doomed (both-bad) curve prunes; a healthy/early one does not,
  - AND logic: bad-overkill-only and bad-ng_recall-only curves both fall through,
  - a too-short overkill series falls through (safe-by-default).
"""
import importlib
import json
import tempfile
from pathlib import Path
from unittest import mock

from mle_star_agent import config
from mle_star_agent.shared import code_runner
from mle_star_agent.shared.curve_extrapolation import project_power_law


# --- pure extrapolation -----------------------------------------------------------

def test_rising_then_plateau_projects_high():
    """A curve climbing toward a high plateau projects a high final value."""
    curve = [0.55, 0.78, 0.90, 0.95, 0.965, 0.97]
    projected, fit = project_power_law(curve)
    assert projected is not None
    assert fit >= 0.5, f"a clean saturating curve should fit well, got {fit:.3f}"
    # The asymptote sits at/above the plateau the curve is approaching.
    assert projected >= max(curve) - 1e-6
    assert projected >= 0.9


def test_monotonically_bad_projects_low_with_good_fit():
    """A curve that decays toward a low floor projects LOW and fits well enough
    to clear CURVE_ABORT_MIN_FIT — this is the case that legitimately aborts."""
    curve = [0.45, 0.30, 0.22, 0.18, 0.16, 0.15]
    projected, fit = project_power_law(curve)
    assert projected is not None
    assert fit >= config.CURVE_ABORT_MIN_FIT, f"decaying curve should fit, got {fit:.3f}"
    assert projected <= min(curve) + 1e-6
    assert projected < 0.5


def test_two_point_curve_returns_no_projection():
    """Fewer than CURVE_ABORT_MIN_EPOCHS points -> no projection (never abort)."""
    projected, fit = project_power_law([0.4, 0.9])
    assert projected is None
    assert fit == 0.0


def test_noisy_curve_has_low_fit_quality():
    """An erratic, non-power-law curve must not earn a trustworthy fit, so the
    caller's `fit_quality >= CURVE_ABORT_MIN_FIT` gate keeps it from aborting."""
    curve = [0.5, 0.95, 0.30, 0.92, 0.28, 0.88]
    projected, fit = project_power_law(curve)
    assert fit < config.CURVE_ABORT_MIN_FIT, f"noisy curve should fit poorly, got {fit:.3f}"


def test_flat_curve_is_degenerate():
    """A perfectly flat curve has no curvature to fit -> degenerate, no projection."""
    projected, fit = project_power_law([0.8, 0.8, 0.8, 0.8])
    assert projected is None
    assert fit == 0.0


def test_extrapolation_never_raises_on_bad_input():
    """Pure helper: garbage in -> (None, 0.0), never an exception."""
    for bad in ([], [float("nan"), 0.1, 0.2, 0.3], ["x", "y", "z"], [None, None, None]):
        projected, fit = project_power_law(bad)
        assert projected is None and fit == 0.0


def test_curve_abort_config_is_conservative():
    """Gates must be safe-by-default: a real multi-epoch curve and a high fit bar."""
    assert config.CURVE_ABORT_MIN_EPOCHS == 3
    assert config.CURVE_ABORT_DEBUG_EPOCHS == 4
    assert config.CURVE_ABORT_MARGIN == 0.05
    assert config.CURVE_ABORT_MIN_FIT == 0.70
    assert config.CURVE_ABORT_OVERKILL_MARGIN == 0.10
    # overkill is noisier on 5%-data micro-runs, so its margin must be looser
    # (>=) than ng_recall's — never reuse CURVE_ABORT_MARGIN for overkill.
    assert config.CURVE_ABORT_OVERKILL_MARGIN >= config.CURVE_ABORT_MARGIN
    # the debug curve must actually have enough points to fit
    assert config.CURVE_ABORT_DEBUG_EPOCHS >= config.CURVE_ABORT_MIN_EPOCHS


def test_debug_micro_run_emits_multi_epoch_curve():
    """The debug rewrite now caps epochs to the (multi-epoch) curve cap, not 1."""
    cap = config.CURVE_ABORT_DEBUG_EPOCHS
    patched = code_runner.apply_debug_patches("num_epochs = 50\n")
    assert f"num_epochs = {cap}" in patched
    assert cap > 1, "a single epoch cannot form an extrapolatable curve"


# --- end-to-end wiring in the evaluator -------------------------------------------

_MODULE = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.evaluator_agent"
)


class _FakeActions:
    escalate = False


class _FakeContext:
    """Minimal tool_context with an established baseline (best ng_recall = 0.90)."""

    def __init__(self):
        self.state = dict({
            "current_script": "print('stub')",
            "outer_iteration": 0,
            "inner_iteration": 0,
            "current_best_score": 0.90,
            "best_miss_rate": 0.10,        # => best ng_recall = 0.90
            "best_overkill_rate": 0.05,
            "best_accuracy": 0.90,
            "best_f1": 0.90,
            "no_improve_count": 0,
            "refinement_plan": {
                "target_component": "threshold_sweep",
                "changes_summary": "baseline",
            },
        })
        self.actions = _FakeActions()


def _metrics_line(tp, tn, fp, fn, prob_gap=0.2):
    return (
        'METRICS: {"tp": %d, "tn": %d, "fp": %d, "fn": %d, '
        '"threshold": 0.5, "avg_latency_ms": 1, "prob_gap": %s}'
        % (tp, tn, fp, fn, prob_gap)
    )


def _epoch_lines(ng_recalls, overkill=0.05, overkills=None):
    """Render EPOCH_LOG lines with the given per-epoch val_ng_recall series.

    ``overkills``, when provided, is a per-epoch val_overkill series the same
    length as ``ng_recalls``; otherwise the constant ``overkill`` is emitted for
    every epoch (a flat curve, which the power-law fit treats as degenerate)."""
    series = overkills if overkills is not None else [overkill] * len(ng_recalls)
    out = []
    for i, (ng, ov) in enumerate(zip(ng_recalls, series), start=1):
        out.append(
            'EPOCH_LOG: {"epoch": %d, "train_loss": 0.1, "val_loss": 0.1, '
            '"val_ng_recall": %s, "val_overkill": %s}' % (i, ng, ov)
        )
    return "\n".join(out)


def _run_evaluator(debug_stdout, debug_duration_ms=5000.0):
    """Run evaluate_and_update_fn with run_script's FIRST (debug) call returning
    ``debug_stdout``; later calls return a benign near-target full-run result.
    call_count distinguishes a prune (1 call) from a real full run (>= 2)."""
    debug_result = mock.Mock(
        returncode=0, timed_out=False, duration_ms=debug_duration_ms,
        stdout=debug_stdout, stderr="",
    )
    full_result = mock.Mock(
        returncode=0, timed_out=False, duration_ms=60000.0,
        stdout=_metrics_line(98, 93, 7, 2), stderr="",
    )
    calls = {"n": 0}

    def _side_effect(*args, **kwargs):
        calls["n"] += 1
        return debug_result if calls["n"] == 1 else full_result

    run_script = mock.Mock(side_effect=_side_effect)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        with (
            mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
            mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"),
            mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
            mock.patch.object(_MODULE.code_runner, "run_script", run_script),
        ):
            context = _FakeContext()
            message = _MODULE.evaluate_and_update_fn(context)
            ckpt = checkpoint_dir / "refinement_0_0.json"
            saved = json.loads(ckpt.read_text()) if ckpt.exists() else None
            return message, run_script.call_count, saved, context


def test_doomed_curve_is_pruned():
    """A debug curve decaying to a low NG-recall floor (well below best 0.90)
    AND a rising overkill curve (well above best 0.05) — both with clean fits —
    prunes the full run. Both trajectories must project bad for the AND gate."""
    stdout = (
        _metrics_line(tp=70, tn=95, fp=5, fn=30)  # not egregious -> survives DEBUG_PREDICT
        + "\n"
        + _epoch_lines(
            [0.45, 0.30, 0.22, 0.18],
            overkills=[0.18, 0.30, 0.38, 0.42],  # rising toward ~0.45 >> best 0.05
        )
    )
    message, call_count, saved, context = _run_evaluator(stdout)
    assert call_count == 1, "full run must NOT execute after a curve abort"
    assert "CURVE ABORT PRUNED" in message
    assert saved is not None
    assert saved["failure_reason"] == "curve_abort_projected_low_utility"
    assert saved["full_run_executed"] is False
    assert saved["full_run_reason"] == "curve_abort_projected_low"
    assert saved["projected_ng_recall"] < 0.90
    assert saved["curve_fit_quality"] >= config.CURVE_ABORT_MIN_FIT
    assert saved["projected_overkill"] > 0.05 + config.CURVE_ABORT_OVERKILL_MARGIN
    assert saved["overkill_curve_fit_quality"] >= config.CURVE_ABORT_MIN_FIT
    assert context.state["inner_iteration"] == 1
    assert context.state["no_improve_count"] == 1


def test_bad_overkill_but_good_ng_recall_is_not_pruned():
    """AND logic: a debug curve with a doomed overkill trajectory (rising well
    above best 0.05) but a HEALTHY ng_recall trajectory (rising to a high
    plateau) must NOT prune — overkill alone is threshold-fixable, so the
    candidate falls through to the full run."""
    stdout = (
        _metrics_line(tp=70, tn=95, fp=5, fn=30)
        + "\n"
        + _epoch_lines(
            [0.80, 0.90, 0.95, 0.97],            # healthy ng_recall -> projects high
            overkills=[0.18, 0.30, 0.38, 0.42],  # doomed overkill -> projects high
        )
    )
    message, call_count, saved, _ = _run_evaluator(stdout)
    assert call_count >= 2, "bad overkill alone must NOT prune (full run proceeds)"
    assert "CURVE ABORT PRUNED" not in message


def test_bad_ng_recall_but_good_overkill_is_not_pruned():
    """AND logic: a doomed ng_recall trajectory with a HEALTHY (low, flat)
    overkill trajectory must NOT prune — only one of the two projections is bad,
    so the run still gets its full execution."""
    stdout = (
        _metrics_line(tp=70, tn=95, fp=5, fn=30)
        + "\n"
        + _epoch_lines([0.45, 0.30, 0.22, 0.18], overkill=0.05)  # flat, healthy overkill
    )
    message, call_count, saved, _ = _run_evaluator(stdout)
    assert call_count >= 2, "bad ng_recall with good overkill must NOT prune under AND"
    assert "CURVE ABORT PRUNED" not in message


def test_overkill_too_few_points_falls_through():
    """Both ng_recall and overkill are doomed, but the overkill series has only 2
    points (< CURVE_ABORT_MIN_EPOCHS). Missing ng_recall on those same epochs is
    irrelevant here — the overkill projection returns None, so the AND gate can't
    fire and the candidate falls through to the full run (safe-by-default)."""
    # Render a 4-epoch ng_recall curve but emit val_overkill on only the first 2.
    lines = [
        'EPOCH_LOG: {"epoch": 1, "val_ng_recall": 0.45, "val_overkill": 0.20}',
        'EPOCH_LOG: {"epoch": 2, "val_ng_recall": 0.30, "val_overkill": 0.40}',
        'EPOCH_LOG: {"epoch": 3, "val_ng_recall": 0.22}',
        'EPOCH_LOG: {"epoch": 4, "val_ng_recall": 0.18}',
    ]
    stdout = _metrics_line(tp=70, tn=95, fp=5, fn=30) + "\n" + "\n".join(lines)
    message, call_count, saved, _ = _run_evaluator(stdout)
    assert call_count >= 2, "too-short overkill curve must fall through to full run"
    assert "CURVE ABORT PRUNED" not in message


def test_healthy_curve_is_not_pruned():
    """A debug curve climbing to a high plateau projects high -> full run proceeds."""
    stdout = (
        _metrics_line(tp=98, tn=93, fp=7, fn=2)
        + "\n"
        + _epoch_lines([0.80, 0.90, 0.95, 0.97])
    )
    message, call_count, saved, _ = _run_evaluator(stdout)
    assert call_count >= 2, "a healthy candidate must proceed to the full run"
    assert "CURVE ABORT PRUNED" not in message


def test_too_few_epochs_is_not_pruned():
    """Only 2 epoch points (< CURVE_ABORT_MIN_EPOCHS) -> no projection, no abort."""
    stdout = (
        _metrics_line(tp=70, tn=95, fp=5, fn=30)
        + "\n"
        + _epoch_lines([0.30, 0.20])
    )
    message, call_count, saved, _ = _run_evaluator(stdout)
    assert call_count >= 2, "too-short curve must fall through to the full run"
    assert "CURVE ABORT PRUNED" not in message


def test_no_baseline_never_curve_aborts():
    """With no established baseline (best ng_recall ~0), even a low projection
    can't be 'confidently worse', so nothing is pruned by the curve logic."""
    stdout = (
        _metrics_line(tp=70, tn=95, fp=5, fn=30)
        + "\n"
        + _epoch_lines([0.45, 0.30, 0.22, 0.18])
    )
    debug_result = mock.Mock(
        returncode=0, timed_out=False, duration_ms=5000.0, stdout=stdout, stderr="",
    )
    full_result = mock.Mock(
        returncode=0, timed_out=False, duration_ms=60000.0,
        stdout=_metrics_line(98, 93, 7, 2), stderr="",
    )
    calls = {"n": 0}

    def _side_effect(*args, **kwargs):
        calls["n"] += 1
        return debug_result if calls["n"] == 1 else full_result

    run_script = mock.Mock(side_effect=_side_effect)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        with (
            mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
            mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"),
            mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
            mock.patch.object(_MODULE.code_runner, "run_script", run_script),
        ):
            context = _FakeContext()
            context.state["current_best_score"] = 0.0
            context.state["best_miss_rate"] = 1.0  # best ng_recall = 0.0
            message = _MODULE.evaluate_and_update_fn(context)
    assert run_script.call_count >= 2, "no-baseline candidate must get the full run"
    assert "CURVE ABORT PRUNED" not in message


if __name__ == "__main__":
    test_rising_then_plateau_projects_high()
    test_monotonically_bad_projects_low_with_good_fit()
    test_two_point_curve_returns_no_projection()
    test_noisy_curve_has_low_fit_quality()
    test_flat_curve_is_degenerate()
    test_extrapolation_never_raises_on_bad_input()
    test_curve_abort_config_is_conservative()
    test_debug_micro_run_emits_multi_epoch_curve()
    test_doomed_curve_is_pruned()
    test_bad_overkill_but_good_ng_recall_is_not_pruned()
    test_bad_ng_recall_but_good_overkill_is_not_pruned()
    test_overkill_too_few_points_falls_through()
    test_healthy_curve_is_not_pruned()
    test_too_few_epochs_is_not_pruned()
    test_no_baseline_never_curve_aborts()
    print("all tests passed")
