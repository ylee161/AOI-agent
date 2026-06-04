"""Tests for code_runner debug_mode (KompeteAI Accelerated Debugger).

A broken script must fail fast under debug_mode rather than timing out, and the
debug patching must not mutate the caller's original script string.

Also covers the KompeteAI predictive early-abort: the evaluator parses the METRICS
the cheap micro-run already prints and prunes EGREGIOUS variants before paying for
the 45-60 min full run, while letting borderline / metric-less runs fall through.
"""
import importlib
import json
import tempfile
import time
from pathlib import Path
from unittest import mock

from mle_star_agent.shared import code_runner
from mle_star_agent import config


BROKEN_SCRIPT = """
import torch

def main(:          # <- deliberate syntax error
    epochs = 50
    print("should never get here")

main()
"""


def test_broken_script_fails_fast_in_debug_mode():
    """A syntax-error script exits in well under 30s with returncode != 0."""
    start = time.monotonic()
    result = code_runner.run_script(BROKEN_SCRIPT, debug_mode=True)
    elapsed = time.monotonic() - start

    assert result.returncode != 0, "broken script should not exit cleanly"
    assert not result.timed_out, "broken script should fail outright, not time out"
    assert elapsed < 30, f"debug run took {elapsed:.1f}s, expected < 30s"


def test_debug_timeout_capped_at_config_value():
    """debug_mode caps the effective timeout at DEBUG_CHECK_TIMEOUT_SECONDS."""
    assert config.DEBUG_CHECK_TIMEOUT_SECONDS == 120


def test_debug_patches_do_not_mutate_original_script():
    """apply_debug_patches returns a new string; the input is untouched."""
    cap = config.CURVE_ABORT_DEBUG_EPOCHS
    original = "num_epochs = 50\nloader = DataLoader(train_ds, batch_size=32)\n"
    snapshot = original
    patched = code_runner.apply_debug_patches(original)

    assert original == snapshot, "original script string must not be mutated"
    assert f"num_epochs = {cap}" in patched, "epoch value should be forced to the debug cap"
    assert "__aoi_cap5(train_ds)" in patched, "DataLoader arg should be capped"
    assert "num_epochs = 50" not in patched


def test_epoch_regex_preserves_variable_name():
    """Epoch rewrite keeps the LHS name and only changes the integer literal."""
    cap = config.CURVE_ABORT_DEBUG_EPOCHS
    patched = code_runner.apply_debug_patches("EPOCHS = 20\nmax_epochs=7\n")
    assert f"EPOCHS = {cap}" in patched
    assert f"max_epochs={cap}" in patched


def test_dry_run_ternary_epochs_are_capped():
    """The mandated `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20` form (which the
    coder agents emit verbatim) must have its full-run `else` literal capped — the
    debug run does not set DRY_RUN, so otherwise it would run the full 20 epochs."""
    cap = config.CURVE_ABORT_DEBUG_EPOCHS
    patched = code_runner.apply_debug_patches(
        "epochs = DRY_RUN_EPOCHS if DRY_RUN else 20\n"
    )
    assert f"else {cap}" in patched
    assert "else 20" not in patched
    # the DRY_RUN_EPOCHS reference itself is untouched
    assert "DRY_RUN_EPOCHS if DRY_RUN" in patched


def test_scheduler_and_counter_epoch_vars_are_not_clobbered():
    """warmup/patience/best/counter epoch vars are NOT training lengths — forcing
    them to the cap would corrupt scheduler semantics and fake a learning curve."""
    src = (
        "warmup_epochs = 5\n"
        "patience_epochs = 3\n"
        "best_epoch = 0\n"
        "epochs_done = 0\n"
    )
    patched = code_runner.apply_debug_patches(src)
    assert "warmup_epochs = 5" in patched
    assert "patience_epochs = 3" in patched
    assert "best_epoch = 0" in patched
    assert "epochs_done = 0" in patched


def test_epoch_literal_with_exponent_is_replaced_whole():
    """A scientific-notation literal must be replaced entirely, not leave `e3`."""
    cap = config.CURVE_ABORT_DEBUG_EPOCHS
    patched = code_runner.apply_debug_patches("num_epochs = 1e3\n")
    assert f"num_epochs = {cap}" in patched
    assert "e3" not in patched


def test_debug_predict_thresholds_are_conservative():
    """The predictive-abort gates must be far looser than the acceptance targets,
    so a noisy 1-epoch/5%-data run only prunes EGREGIOUS variants."""
    assert config.DEBUG_PREDICT_OVERKILL_MAX == 0.60
    assert config.DEBUG_PREDICT_NG_RECALL_MIN == 0.50
    # looser than acceptance (we must never false-prune a borderline near-target run)
    assert config.DEBUG_PREDICT_OVERKILL_MAX > config.OVERKILL_RELAXED_MAX
    assert config.DEBUG_PREDICT_NG_RECALL_MIN < config.NG_RECALL_RELAXED_MIN


# --- predictive early-abort on the debug micro-run -------------------------------

_MODULE = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.evaluator_agent"
)


class _FakeActions:
    escalate = False


class _FakeContext:
    """Minimal tool_context mirroring the state evaluate_and_update_fn reads."""

    def __init__(self):
        self.state = dict({
            "current_script": "print('stub')",
            "outer_iteration": 0,
            "inner_iteration": 0,
            "current_best_score": 0.90,
            "best_miss_rate": 0.10,
            "best_overkill_rate": 0.70,
            "best_accuracy": 0.60,
            "best_f1": 0.65,
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


def _run_evaluator(debug_stdout, debug_duration_ms=5000.0):
    """Run evaluate_and_update_fn with run_script's FIRST (debug) call returning
    ``debug_stdout``. A second (full-run) call returns a benign result; we track
    call_count so a prune (1 call) is distinguishable from a full run (2 calls)."""
    debug_result = mock.Mock(
        returncode=0, timed_out=False, duration_ms=debug_duration_ms,
        stdout=debug_stdout, stderr="",
    )
    # benign near-target full-run result so the non-pruned path completes cleanly.
    # The full path may call run_script several times (e.g. multiseed confirmation),
    # so the FIRST call returns the debug result and every later call the full result.
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


def test_egregious_overkill_micro_run_is_pruned():
    """overkill=0.9 in the micro-run -> pruned, no full run, checkpoint records it."""
    message, call_count, saved, context = _run_evaluator(
        _metrics_line(tp=100, tn=10, fp=90, fn=0)  # overkill 0.90, ng_recall 1.0
    )
    assert call_count == 1, "full run must NOT be executed after a predictive prune"
    assert "DEBUG PREDICT PRUNED" in message
    assert saved is not None
    assert saved["failure_reason"] == "debug_predicted_low_utility"
    assert saved["smoke_metrics"]["overkill_rate"] == 0.9
    assert saved["smoke_score"] is not None
    assert saved["full_run_executed"] is False
    assert saved["full_run_reason"] == "smoke_pruned_egregious"
    assert "METRICS" in saved["stdout_tail"]  # debug metrics carried into stdout_tail
    assert context.state["latest_smoke_run"]["metrics"]["overkill_rate"] == 0.9
    assert context.state["inner_iteration"] == 1
    assert context.state["no_improve_count"] == 1


def test_near_target_micro_run_is_not_pruned():
    """A borderline near-target micro-run (overkill 0.07, ng_recall 0.98) survives
    and gets the full run — we must never false-prune a promising candidate."""
    message, call_count, saved, _ = _run_evaluator(
        _metrics_line(tp=98, tn=93, fp=7, fn=2)  # overkill 0.07, ng_recall 0.98
    )
    assert call_count >= 2, "near-target candidate must proceed to the full run"
    assert "DEBUG PREDICT PRUNED" not in message
    if saved is not None:
        assert saved.get("failure_reason") != "debug_predicted_low_utility"
        assert saved["smoke_metrics"]["ng_recall"] == 0.98
        assert saved["smoke_score"] is not None
        assert saved["full_run_executed"] is True
        assert saved["full_run_reason"] == "full_run_after_smoke"


def test_missing_micro_run_metrics_is_not_pruned():
    """Absence of parseable METRICS on 5% data is NOT evidence of a bad script,
    so the run falls through to the full run rather than being pruned."""
    message, call_count, saved, _ = _run_evaluator(
        "epoch 1 done; no metrics block emitted here\n"
    )
    assert call_count >= 2, "metric-less micro-run must proceed to the full run"
    assert "DEBUG PREDICT PRUNED" not in message
    if saved is not None:
        assert saved.get("failure_reason") != "debug_predicted_low_utility"
        assert saved["smoke_metrics"] is None
        assert saved["smoke_score"] is None
        assert saved["full_run_executed"] is True
        assert saved["full_run_reason"] == "full_run_after_smoke"


if __name__ == "__main__":
    test_broken_script_fails_fast_in_debug_mode()
    test_debug_timeout_capped_at_config_value()
    test_debug_patches_do_not_mutate_original_script()
    test_epoch_regex_preserves_variable_name()
    test_debug_predict_thresholds_are_conservative()
    test_egregious_overkill_micro_run_is_pruned()
    test_near_target_micro_run_is_not_pruned()
    test_missing_micro_run_metrics_is_not_pruned()
    print("all tests passed")
