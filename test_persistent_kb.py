"""Tests for the MLEvolve persistent AOI knowledge base.

Verifies the two halves of the feature wired through Phase 2 refinement:
  1. evaluate_and_update_fn appends exactly one 4-field record to
     persistent_aoi_kb.json (CKPT_PERSISTENT_KB) after a (mocked) run.
  2. load_tried_approaches_fn folds that KB into the planner context string,
     emitting a PERSISTENT_KB_SUMMARY block.

The evaluator runs the candidate script via code_runner.run_script (once for the
debug pre-check, once for the full run); both are mocked so no real training
happens. All checkpoint paths are redirected into a TemporaryDirectory.
"""

import importlib
import json
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
    """Minimal stand-in for an ADK ToolContext: a dict-backed .state and .actions."""

    def __init__(self, state):
        self.state = FakeState(state)
        self.actions = FakeActions()


class PersistentKBTests(unittest.TestCase):
    def test_evaluator_writes_kb_and_planner_summarizes_it(self):
        # A clean, parseable METRICS line so the full run yields real metrics.
        # overkill = fp / (fp + tn) = 9 / 30 = 0.30 (> 0.12), so the multiseed
        # confirmation path is not triggered and run_script is called twice.
        stdout = 'METRICS: {"tp": 27, "tn": 21, "fp": 9, "fn": 4, "threshold": 0.2, "avg_latency_ms": 1}'
        run_result = mock.Mock(
            returncode=0, timed_out=False, duration_ms=12.0, stdout=stdout, stderr=""
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            kb_path = checkpoint_dir / "persistent_aoi_kb.json"
            patches = (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(
                    config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"
                ),
                mock.patch.object(
                    config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"
                ),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path),
                mock.patch.object(
                    evaluator_agent.code_runner, "run_script", return_value=run_result
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                eval_ctx = FakeContext({
                    "current_script": "print('candidate pipeline script')\n# body...",
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "current_best_score": 0.0,
                    "best_overkill_rate": 1.0,
                    "best_miss_rate": 1.0,
                    "best_accuracy": 0.0,
                    "best_f1": 0.0,
                    "no_improve_count": 0,
                    "token_count": 0,
                    "selected_refinement_strategy": "focal_loss: down-weight easy negatives",
                    "refinement_plan": {
                        "target_component": "weighted_loss",
                        "changes_summary": "swap BCE for focal loss",
                    },
                })
                evaluator_agent.evaluate_and_update_fn(eval_ctx)

                # --- KB file created with exactly one 4-field record ---
                self.assertTrue(kb_path.is_file(), "persistent_aoi_kb.json was not created")
                records = json.loads(kb_path.read_text(encoding="utf-8"))
                self.assertIsInstance(records, list)
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(
                    set(record.keys()), {"plan", "code_snippet", "metrics", "label"}
                )
                self.assertEqual(record["plan"], "focal_loss: down-weight easy negatives")
                self.assertTrue(record["code_snippet"].startswith("print('candidate"))
                self.assertLessEqual(len(record["code_snippet"]), 300)
                self.assertIn("ng_recall", record["metrics"])
                self.assertIn(record["label"], {"success", "failure"})

                # --- Planner context string includes the KB summary block ---
                planner_ctx = FakeContext({"input_modality": "mono"})
                context_str = planner_agent.load_tried_approaches_fn(planner_ctx)
                self.assertIn("PERSISTENT_KB_SUMMARY", context_str)


if __name__ == "__main__":
    unittest.main()
