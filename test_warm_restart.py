"""Tests for plateau-triggered warm-restart refinement.

Warm restart is a bounded first response to inner-loop stagnation: try one
optimizer/lr-schedule jolt of the current best script before falling through to
the existing ideation/restart path.
"""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mle_star_agent import config

evaluator_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.evaluator_agent"
)
planner_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.planner_agent"
)


class FakeState(dict):
    pass


class FakeActions:
    escalate = False


class FakeContext:
    def __init__(self, state):
        self.state = FakeState(state)
        self.actions = FakeActions()


def _non_improving_result():
    stdout = (
        'METRICS: {"tp": 27, "tn": 21, "fp": 9, "fn": 4, "threshold": 0.2, '
        '"avg_latency_ms": 1, "prob_gap": 0.4, "roc_auc": 0.8}'
    )
    return mock.Mock(returncode=0, timed_out=False, duration_ms=60000.0, stdout=stdout, stderr="")


def _stagnating_context(extra=None):
    state = {
        "current_script": "# weaker candidate\nx = 1\n",
        "best_pipeline_script": "# current best\nmodel = best()\n",
        "outer_iteration": 1,
        "inner_iteration": 2,
        "current_best_score": 0.95,
        "best_overkill_rate": 0.30,
        "best_miss_rate": 0.05,
        "best_accuracy": 0.85,
        "best_f1": 0.90,
        "no_improve_count": config.INNER_STAGNATION_MAX_UNCONSTRAINED - 1,
        "token_count": 0,
        "selected_refinement_strategy": "focal_loss: x",
        "selected_strategy_fingerprint": {
            "target_component": "weighted_loss",
            "mechanism_class": "focal_loss",
        },
        "refinement_plan": {
            "target_component": "weighted_loss",
            "changes_summary": "focal loss",
        },
    }
    if extra:
        state.update(extra)
    return FakeContext(state)


class WarmRestartStagnationTests(unittest.TestCase):
    def test_stagnation_arms_warm_restart_before_ideation(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best.json"),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", checkpoint_dir / "kb.json"),
                mock.patch.object(evaluator_agent.code_runner, "run_script", return_value=_non_improving_result()),
                mock.patch.object(evaluator_agent, "trigger_ideation", return_value="IDEATION_SHOULD_NOT_RUN") as ideation,
            ):
                ctx = _stagnating_context()
                out = evaluator_agent.evaluate_and_update_fn(ctx)

        self.assertIn("WARM_RESTART_ARMED", out)
        self.assertTrue(ctx.state.get("pending_warm_restart"))
        self.assertFalse(ctx.actions.escalate)
        self.assertEqual(ctx.state["outer_iteration"], 1)
        self.assertEqual(ctx.state["inner_iteration"], 3)
        self.assertEqual(ctx.state["no_improve_count"], 0)
        ideation.assert_not_called()

    def test_pending_warm_restart_is_consumed_once_by_strategy_gate(self):
        ctx = FakeContext({
            "pending_warm_restart": True,
            "warm_restart_best_sha": "abcdef12",
            "outer_iteration": 1,
            "inner_iteration": 3,
            "tried_approaches": [],
        })

        first = planner_agent.ensure_selected_strategy_fn(ctx)
        second = planner_agent.ensure_selected_strategy_fn(ctx)

        self.assertIn("WARM_RESTART_STRATEGY_INSTALLED", first)
        self.assertIn("CosineAnnealingWarmRestarts", ctx.state["selected_refinement_strategy"])
        self.assertEqual(
            ctx.state["selected_strategy_fingerprint"],
            {
                "target_component": "optimizer/lr-schedule",
                "mechanism_class": "warm_restart_abcdef12",
            },
        )
        self.assertFalse(ctx.state.get("pending_warm_restart", False))
        self.assertIn("STRATEGY_READY", second)

    def test_failed_warm_restart_falls_through_to_ideation_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best.json"),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", checkpoint_dir / "kb.json"),
                mock.patch.object(evaluator_agent.code_runner, "run_script", return_value=_non_improving_result()),
                mock.patch.object(evaluator_agent, "trigger_ideation", return_value="IDEATION_TRIGGERED") as ideation,
            ):
                ctx = _stagnating_context({
                    "no_improve_count": 0,
                    "selected_refinement_strategy": (
                        "plateau_warm_restart: re-raise base LR with "
                        "CosineAnnealingWarmRestarts"
                    ),
                    "selected_strategy_fingerprint": {
                        "target_component": "optimizer/lr-schedule",
                        "mechanism_class": "warm_restart_abcdef12",
                    },
                    "refinement_plan": {
                        "target_component": "optimizer/lr-schedule",
                        "changes_summary": "warm restart",
                    },
                    "warm_restart_attempted_best_abcdef12": True,
                })
                out = evaluator_agent.evaluate_and_update_fn(ctx)

        self.assertIn("WARM_RESTART_FAILED", out)
        self.assertIn("IDEATION_TRIGGERED", out)
        self.assertTrue(ctx.actions.escalate)
        self.assertEqual(ctx.state["outer_iteration"], 2)
        self.assertEqual(ctx.state["inner_iteration"], 0)
        self.assertFalse(ctx.state.get("pending_warm_restart", False))
        ideation.assert_called_once()

        warm_attempts = [
            a for a in ctx.state["tried_approaches"]
            if (a.get("strategy_fingerprint") or {}).get("mechanism_class") == "warm_restart_abcdef12"
        ]
        self.assertEqual(len(warm_attempts), 1)
        self.assertEqual(warm_attempts[0]["target_component"], "optimizer/lr-schedule")


if __name__ == "__main__":
    unittest.main()
