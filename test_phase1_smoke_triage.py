import importlib
import json
import tempfile
from pathlib import Path
from unittest import mock

from mle_star_agent import config


_MODULE = importlib.import_module(
    "mle_star_agent.phases.phase1_init.candidate_evaluator_agent"
)


class _FakeContext:
    def __init__(self, scripts=None):
        self.state = {
            "candidate_scripts": scripts or [],
        }


def _metrics_line(tp, tn, fp, fn, prob_gap=0.2, threshold=0.5):
    return (
        'METRICS: {"tp": %d, "tn": %d, "fp": %d, "fn": %d, '
        '"threshold": %s, "avg_latency_ms": 1, "prob_gap": %s}'
        % (tp, tn, fp, fn, threshold, prob_gap)
    )


def _result(stdout, returncode=0, duration_ms=5000.0):
    return mock.Mock(
        returncode=returncode,
        timed_out=False,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr="",
    )


def _guard_passing_script(marker):
    """Minimal script body that passes the static stub guard in run_slot_fn
    (>= 50 lines and defines build_model)."""
    filler = "\n".join(f"# filler line {i}" for i in range(60))
    return f"# marker: {marker}\ndef build_model():\n    pass\n{filler}\n"


def _candidate(slot=0, name=None, script=None):
    name = name or f"candidate_{slot}"
    return {
        "name": name,
        "architecture": "stub",
        "script": _guard_passing_script(script or name),
    }


def test_phase1_stub_script_is_rejected_without_running():
    context = _FakeContext([{
        "name": "stubby",
        "architecture": "stub",
        "script": "print('METRICS: {}')",
    }])
    run_slot = _MODULE._make_run_slot_fn(0)

    with mock.patch.object(_MODULE.code_runner, "run_script") as run_script:
        message = run_slot(context)

    result = context.state["candidate_result_0"]
    assert run_script.call_count == 0
    assert "REJECTED" in message
    assert result["status"] == "failed"
    assert result["full_run_reason"] == "stub_or_truncated_script"
    assert result["metrics"] is None


def test_phase1_egregious_smoke_candidate_is_skipped():
    context = _FakeContext([_candidate()])
    run_slot = _MODULE._make_run_slot_fn(0)

    with mock.patch.object(
        _MODULE.code_runner,
        "run_script",
        return_value=_result(_metrics_line(tp=100, tn=10, fp=90, fn=0)),
    ) as run_script:
        message = run_slot(context)

    result = context.state["candidate_result_0"]
    assert run_script.call_count == 1
    assert "smoke-pruned" in message
    assert result["status"] == "smoke_pruned"
    assert result["full_run_executed"] is False
    assert result["full_run_reason"] == "smoke_pruned_egregious"
    assert result["metrics"] is None
    assert result["smoke_metrics"]["overkill_rate"] == 0.9
    assert result["smoke_score"] is not None


def test_phase1_near_target_smoke_candidate_gets_full_run_and_checkpoint():
    context = _FakeContext([_candidate()])
    run_slot = _MODULE._make_run_slot_fn(0)

    smoke = _result(_metrics_line(tp=98, tn=93, fp=7, fn=2), duration_ms=5000.0)
    full = _result(_metrics_line(tp=98, tn=93, fp=7, fn=2), duration_ms=60000.0)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        with (
            mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
            mock.patch.object(config, "CKPT_CANDIDATE_SCORES", checkpoint_dir / "candidate_scores.json"),
            mock.patch.object(_MODULE, "NUM_SLOTS", 1),
            mock.patch.object(_MODULE.code_runner, "run_script", side_effect=[smoke, full, full, full]) as run_script,
        ):
            run_slot(context)
            _MODULE.consolidate_candidate_scores_fn(context)
            saved = json.loads((checkpoint_dir / "candidate_scores.json").read_text())

    result = saved["scores"][0]
    assert run_script.call_count == 4
    assert result["status"] == "success"
    assert result["full_run_executed"] is True
    assert result["full_run_reason"] == "top_smoke_rank"
    assert result["metrics"]["ng_recall"] == 0.98
    assert result["smoke_metrics"]["ng_recall"] == 0.98
    assert result["smoke_score"] is not None


def test_phase1_missing_smoke_metrics_candidate_gets_full_run():
    context = _FakeContext([_candidate()])
    run_slot = _MODULE._make_run_slot_fn(0)

    smoke = _result("epoch 1 done; no metrics block emitted here", duration_ms=5000.0)
    full = _result(_metrics_line(tp=98, tn=93, fp=7, fn=2), duration_ms=60000.0)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        with (
            mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
            mock.patch.object(config, "CKPT_CANDIDATE_SCORES", checkpoint_dir / "candidate_scores.json"),
            mock.patch.object(_MODULE, "NUM_SLOTS", 1),
            mock.patch.object(_MODULE.code_runner, "run_script", side_effect=[smoke, full, full, full]) as run_script,
        ):
            run_slot(context)
            _MODULE.consolidate_candidate_scores_fn(context)

    result = context.state["candidate_scores"][0]
    assert run_script.call_count == 4
    assert result["status"] == "success"
    assert result["full_run_executed"] is True
    assert result["full_run_reason"] == "missing_smoke_metrics"
    assert result["smoke_metrics"] is None
    assert result["smoke_score"] is None


def test_phase1_smoke_scores_rank_top_and_uncertain_candidates():
    scripts = [
        _candidate(0, "high", "script_high"),
        _candidate(1, "uncertain", "script_uncertain"),
        _candidate(2, "low", "script_low"),
    ]
    context = _FakeContext(scripts)
    context.state.update({
        "candidate_result_0": {
            "slot": 0,
            "name": "high",
            "architecture": "stub",
            "script": "script_high",
            "status": "smoke_pending_full",
            "smoke_metrics": {"ng_recall": 0.98, "miss_rate": 0.02, "overkill_rate": 0.05, "accuracy": 0.95, "f1": 0.95, "roc_auc": 0.96, "prob_gap": 0.4, "threshold": 0.5},
            "smoke_score": 0.95,
            "metrics": None,
        },
        "candidate_result_1": {
            "slot": 1,
            "name": "uncertain",
            "architecture": "stub",
            "script": "script_uncertain",
            "status": "smoke_pending_full",
            "smoke_metrics": {"ng_recall": 0.96, "miss_rate": 0.04, "overkill_rate": 0.08, "accuracy": 0.93, "f1": 0.93, "roc_auc": 0.94, "prob_gap": 0.3, "threshold": 0.5},
            "smoke_score": 0.91,
            "metrics": None,
        },
        "candidate_result_2": {
            "slot": 2,
            "name": "low",
            "architecture": "stub",
            "script": "script_low",
            "status": "smoke_pending_full",
            "smoke_metrics": {"ng_recall": 0.70, "miss_rate": 0.30, "overkill_rate": 0.20, "accuracy": 0.70, "f1": 0.70, "roc_auc": 0.70, "prob_gap": 0.1, "threshold": 0.5},
            "smoke_score": 0.50,
            "metrics": None,
        },
    })
    full = _result(_metrics_line(tp=98, tn=93, fp=7, fn=2), duration_ms=60000.0)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        with (
            mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
            mock.patch.object(config, "CKPT_CANDIDATE_SCORES", checkpoint_dir / "candidate_scores.json"),
            mock.patch.object(config, "PHASE1_SMOKE_TOP_K", 1),
            mock.patch.object(config, "PHASE1_SMOKE_UNCERTAINTY_BAND", 0.05),
            mock.patch.object(_MODULE.code_runner, "run_script", return_value=full) as run_script,
        ):
            _MODULE.consolidate_candidate_scores_fn(context)
            saved = json.loads((checkpoint_dir / "candidate_scores.json").read_text())

    scores = {s["name"]: s for s in context.state["candidate_scores"]}
    saved_scores = {s["name"]: s for s in saved["scores"]}
    assert run_script.call_count == 6  # two selected candidates, three seeds each
    assert scores["high"]["full_run_executed"] is True
    assert scores["high"]["full_run_reason"] == "top_smoke_rank"
    assert scores["uncertain"]["full_run_executed"] is True
    assert scores["uncertain"]["full_run_reason"] == "smoke_uncertainty_band"
    assert scores["low"]["full_run_executed"] is False
    assert scores["low"]["full_run_reason"] == "below_smoke_rank_cutoff"
    assert scores["low"]["status"] == "smoke_deferred"
    assert saved_scores["low"]["smoke_metrics"]["ng_recall"] == 0.70
    assert saved_scores["low"]["smoke_score"] == 0.50
    assert saved_scores["low"]["full_run_executed"] is False
    assert saved_scores["low"]["full_run_reason"] == "below_smoke_rank_cutoff"
