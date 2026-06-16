import importlib
import json
from pathlib import Path
from unittest import mock

import pytest

from mle_star_agent import config
from mle_star_agent.shared import metric_guard


class _FakeActions:
    escalate = False


class _FakeContext:
    def __init__(self, state):
        self.state = dict(state)
        self.actions = _FakeActions()


def _metrics_dict(**overrides):
    data = {
        "accuracy": 0.75,
        "ng_recall": 0.75,
        "miss_rate": 0.25,
        "overkill_rate": 0.25,
        "f1": 0.75,
        "avg_latency_ms": 1.0,
        "threshold": 0.5,
        "ng_count": 20,
        "g_count": 20,
        "tp": 15,
        "tn": 15,
        "fp": 5,
        "fn": 5,
        "roc_auc": 0.7,
        "prob_gap": 0.2,
    }
    data.update(overrides)
    return data


def _metrics_stdout(**overrides):
    return "METRICS: " + json.dumps(_metrics_dict(**overrides))


def _result(stdout, returncode=0, duration_ms=60000.0):
    return mock.Mock(
        returncode=returncode,
        timed_out=False,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr="",
    )


def test_metric_guard_rejects_degenerate_all_g_and_all_ng_solutions():
    all_g = _metrics_dict(
        accuracy=0.5,
        ng_recall=0.0,
        miss_rate=1.0,
        overkill_rate=0.0,
        f1=0.0,
        threshold=1.0,
        tp=0,
        tn=20,
        fp=0,
        fn=20,
    )
    all_ng = _metrics_dict(
        accuracy=0.5,
        ng_recall=1.0,
        miss_rate=0.0,
        overkill_rate=1.0,
        f1=0.6667,
        threshold=0.0,
        tp=20,
        tn=0,
        fp=20,
        fn=0,
    )

    assert metric_guard.guard_metrics(all_g, 60000.0, context="all-g") is None
    assert metric_guard.guard_metrics(all_ng, 60000.0, context="all-ng") is None


def test_metric_guard_rejects_corner_thresholds_even_when_metrics_are_not_all_one_class():
    low_corner = _metrics_dict(threshold=0.0, overkill_rate=0.2, miss_rate=0.1)
    high_corner = _metrics_dict(threshold=1.0, overkill_rate=0.2, miss_rate=0.1)

    assert metric_guard.guard_metrics(low_corner, 60000.0, context="threshold-0") is None
    assert metric_guard.guard_metrics(high_corner, 60000.0, context="threshold-1") is None


def test_phase1_full_candidate_evaluation_hard_fails_degenerate_all_g():
    candidate_evaluator = importlib.import_module(
        "mle_star_agent.phases.phase1_init.candidate_evaluator_agent"
    )
    result_dict = {
        "slot": 0,
        "name": "all_g_candidate",
        "architecture": "stub",
        "status": "smoke_pending_full",
    }
    stdout = _metrics_stdout(
        accuracy=0.5,
        ng_recall=0.0,
        miss_rate=1.0,
        overkill_rate=0.0,
        f1=0.0,
        threshold=1.0,
        tp=0,
        tn=20,
        fp=0,
        fn=20,
    )

    with mock.patch.object(
        candidate_evaluator.code_runner,
        "run_script",
        return_value=_result(stdout),
    ):
        updated = candidate_evaluator._run_full_candidate_evaluation(
            result_dict,
            "print('candidate')",
            "test_degenerate_guard",
        )

    assert updated["status"] == "failed"
    assert updated["metrics"] is None
    assert updated["full_run_executed"] is True


def test_submission_evaluation_hard_fails_degenerate_all_ng(tmp_path):
    submission = importlib.import_module(
        "mle_star_agent.phases.phase4_submission.submission_agent"
    )
    context = _FakeContext({"submission_source": "test"})
    stdout = _metrics_stdout(
        accuracy=0.5,
        ng_recall=1.0,
        miss_rate=0.0,
        overkill_rate=1.0,
        f1=0.6667,
        threshold=0.0,
        tp=20,
        tn=0,
        fp=20,
        fn=0,
    )

    with (
        mock.patch.object(config, "CHECKPOINT_DIR", tmp_path),
        mock.patch.object(config, "CKPT_SUBMISSION", tmp_path / "submission.json"),
        mock.patch.object(submission.code_runner, "run_script", return_value=_result(stdout)),
    ):
        message = submission.evaluate_submission_fn(context, "print('submission')")

    saved = json.loads((tmp_path / "submission.json").read_text())
    assert "Submission FAILED" in message
    assert context.state["final_metrics"] == {}
    assert saved["metrics"] is None
    assert saved["pass_fail"]["reasons"] == ["metrics_missing"]


def test_phase2_no_signal_auc_aborts_outer_loop_and_skips_further_refinement(tmp_path):
    evaluator = importlib.import_module(
        "mle_star_agent.phases.phase2_refinement.evaluator_agent"
    )
    context = _FakeContext({
        "current_script": "print('candidate')",
        "outer_iteration": 0,
        "inner_iteration": 0,
        "current_best_score": 0.40,
        "best_miss_rate": 0.60,
        "best_overkill_rate": 0.30,
        "best_accuracy": 0.50,
        "best_f1": 0.45,
        "no_improve_count": 0,
        "refinement_plan": {
            "target_component": "threshold_sweep",
            "changes_summary": "try threshold",
        },
    })
    debug = _result(_metrics_stdout(roc_auc=0.80, prob_gap=0.3), duration_ms=5000.0)
    full = _result(_metrics_stdout(roc_auc=0.55, prob_gap=0.2), duration_ms=60000.0)

    with (
        mock.patch.object(config, "CHECKPOINT_DIR", tmp_path),
        mock.patch.object(config, "CKPT_BEST_PIPELINE", tmp_path / "best_pipeline.json"),
        mock.patch.object(config, "CKPT_TRIED_APPROACHES", tmp_path / "tried_approaches.json"),
        mock.patch.object(evaluator.code_runner, "run_script", side_effect=[debug, full]),
    ):
        message = evaluator.evaluate_and_update_fn(context)

    saved = json.loads((tmp_path / "refinement_0_0.json").read_text())
    best = json.loads((tmp_path / "best_pipeline.json").read_text())
    assert "NO_CROSS_LOT_SIGNAL" in message
    assert "no cross-lot signal" in message.lower()
    assert context.state["stop_outer_loop"] is True
    assert context.actions.escalate is True
    assert saved["failure_reason"] == "no_cross_lot_signal"
    assert saved["full_run_executed"] is True
    assert best["stop_outer_loop"] is True


def test_phase1_refuses_to_initialize_phase2_when_no_valid_candidate(tmp_path):
    merger = importlib.import_module("mle_star_agent.phases.phase1_init.merger_agent")
    context = _FakeContext({
        "candidate_scores": [
            {"name": "failed_a", "status": "failed", "metrics": None},
            {
                "name": "dead_l0",
                "status": "success",
                "metrics": _metrics_dict(
                    ng_recall=0.0,
                    miss_rate=1.0,
                    overkill_rate=0.0,
                    f1=0.0,
                    threshold=1.0,
                    tp=0,
                    tn=20,
                    fp=0,
                    fn=20,
                ),
            },
        ],
        "candidate_scripts": [
            {"name": "failed_a", "script": "print('failed')"},
            {"name": "dead_l0", "script": "print('dead')"},
        ],
        "token_count": 123,
    })

    with (
        mock.patch.object(config, "CHECKPOINT_DIR", tmp_path),
        mock.patch.object(config, "CKPT_L0", tmp_path / "L0.json"),
        mock.patch.object(config, "CKPT_PHASE2_INIT", tmp_path / "phase2_init.json"),
    ):
        message = merger.initialize_phase2_fn(context)

    assert "NO_VALID_PHASE1_CANDIDATE" in message
    assert not (tmp_path / "L0.json").exists()
    assert not (tmp_path / "phase2_init.json").exists()
    assert context.state.get("stop_outer_loop") is True
