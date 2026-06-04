"""Tests for the MLEvolve persistent AOI knowledge base (quadruple schema).

Each KB record is now {plan, code_diff, metrics, tags}:
  - plan:      selected_refinement_strategy string
  - code_diff: TRUNCATED unified diff (prev best -> current); never a full script
  - metrics:   {ng_recall, miss_rate, overkill_rate, accuracy, improved}
  - tags:      categorical list [failure_mode (if any), "improved"|"regressed"]

Coverage:
  (a) evaluate_and_update_fn appends a new quadruple record with plan, a real
      truncated code_diff, metrics, and tags including the failure_mode.
  (b) make_code_diff truncates long diffs and handles the empty-old case.
  (c) _persistent_kb_summary renders a MIXED list of legacy + new records
      without error.

The evaluator runs the candidate script via code_runner.run_script (debug
pre-check + full run); both are mocked so no real training happens. All
checkpoint paths are redirected into a TemporaryDirectory.
"""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mle_star_agent import config
from mle_star_agent.shared.code_diff import DIFF_CHAR_CAP, make_code_diff

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


class MakeCodeDiffTests(unittest.TestCase):
    def test_empty_old_emits_new_script_note(self):
        new = "line one\nline two\nline three"
        diff = make_code_diff("", new)
        self.assertIn("new script (no previous best)", diff)
        # New content rendered as additions, not diffed against nothing.
        self.assertIn("+line one", diff)
        self.assertIn("+line three", diff)

    def test_whitespace_old_treated_as_empty(self):
        diff = make_code_diff("   \n\t\n", "real = 1")
        self.assertIn("new script (no previous best)", diff)
        self.assertIn("+real = 1", diff)

    def test_real_diff_between_two_scripts(self):
        old = "a = 1\nb = 2\nc = 3\n"
        new = "a = 1\nb = 20\nc = 3\n"
        diff = make_code_diff(old, new)
        self.assertIn("-b = 2", diff)
        self.assertIn("+b = 20", diff)
        # Unchanged context lines are still bounded; this is a genuine unified diff.
        self.assertIn("@@", diff)

    def test_truncation_caps_length(self):
        # Build an old/new pair whose diff blows well past the cap.
        old = "\n".join(f"old_{i} = {i}" for i in range(2000))
        new = "\n".join(f"new_{i} = {i}" for i in range(2000))
        diff = make_code_diff(old, new)
        self.assertLessEqual(len(diff), DIFF_CHAR_CAP + 80)
        self.assertIn("truncated", diff)

    def test_custom_cap_truncates_new_script_note(self):
        new = "x = 1\n" * 5000
        diff = make_code_diff("", new, cap=200)
        self.assertLessEqual(len(diff), 200 + 80)
        self.assertIn("truncated", diff)


class EvaluatorWritesQuadrupleTests(unittest.TestCase):
    def test_new_record_has_plan_diff_metrics_and_failure_mode_tag(self):
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
                    # A non-empty previous best so the diff is a real unified diff.
                    "best_pipeline_script": "print('OLD pipeline')\n# old body\nx = 1\n",
                    "current_script": "print('NEW candidate pipeline')\n# new body\nx = 2\n",
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
                    # failure_mode surfaced via diagnosis_brief.failure_classification.
                    "diagnosis_brief": {
                        "failure_classification": {"failure_mode": "g_ng_overlap"},
                    },
                })
                evaluator_agent.evaluate_and_update_fn(eval_ctx)

                # --- KB file created with exactly one quadruple record ---
                self.assertTrue(kb_path.is_file(), "persistent_aoi_kb.json was not created")
                records = json.loads(kb_path.read_text(encoding="utf-8"))
                self.assertIsInstance(records, list)
                self.assertEqual(len(records), 1)
                record = records[0]
                self.assertEqual(
                    set(record.keys()), {"plan", "code_diff", "metrics", "tags"}
                )
                # plan
                self.assertEqual(record["plan"], "focal_loss: down-weight easy negatives")
                # code_diff is a REAL unified diff (prev best -> current), truncated.
                diff = record["code_diff"]
                self.assertIn("-print('OLD pipeline')", diff)
                self.assertIn("+print('NEW candidate pipeline')", diff)
                self.assertLessEqual(len(diff), DIFF_CHAR_CAP + 80)
                # No full script stored.
                self.assertNotIn("# new body\nx = 2\nprint", diff)
                # metrics carry the expected keys (overkill_rate, not overkill).
                for key in ("ng_recall", "miss_rate", "overkill_rate", "accuracy", "improved"):
                    self.assertIn(key, record["metrics"])
                # tags: failure_mode + improved/regressed outcome.
                self.assertIn("g_ng_overlap", record["tags"])
                self.assertTrue(
                    ("improved" in record["tags"]) or ("regressed" in record["tags"])
                )


class PlannerSummaryTests(unittest.TestCase):
    def test_retrieve_kb_records_filters_by_failure_mode_and_caps_recent(self):
        records = [
            {"plan": "old overlap", "tags": ["g_ng_overlap", "regressed"]},
            {"plan": "threshold", "tags": ["threshold_collapse", "regressed"]},
            {"plan": "mid overlap", "tags": ["g_ng_overlap", "improved"]},
            {"plan": "new overlap", "tags": ["g_ng_overlap", "regressed"]},
        ]

        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=2)

        self.assertEqual([r["plan"] for r in selected], ["mid overlap", "new overlap"])
        for record in selected:
            self.assertIn("g_ng_overlap", planner_agent._kb_record_tags(record))

    def test_retrieve_kb_records_backfills_with_recent_others(self):
        legacy = {"plan": "legacy failure", "metrics": {}, "label": "failure"}
        records = [
            {"plan": "old threshold", "tags": ["threshold_collapse", "regressed"]},
            {"plan": "only overlap", "tags": ["g_ng_overlap", "regressed"]},
            legacy,
            {"plan": "near acceptance", "tags": ["near_acceptance", "improved"]},
        ]

        selected = planner_agent.retrieve_kb_records(records, "g_ng_overlap", k=3)

        self.assertEqual(
            [r["plan"] for r in selected],
            ["only overlap", "legacy failure", "near acceptance"],
        )
        self.assertIn("regressed", planner_agent._kb_record_tags(legacy))

    def test_failure_mode_summary_lists_only_top_k_matching_records(self):
        records = [
            {
                "plan": "threshold plan",
                "code_diff": "+threshold",
                "metrics": {"ng_recall": 0.4},
                "tags": ["threshold_collapse", "regressed"],
            },
            {
                "plan": "old overlap plan",
                "code_diff": "+old",
                "metrics": {"ng_recall": 0.5},
                "tags": ["g_ng_overlap", "regressed"],
            },
            {
                "plan": "new overlap plan",
                "code_diff": "+new\n" + ("x" * 300),
                "metrics": {"ng_recall": 0.7},
                "tags": ["g_ng_overlap", "improved"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            kb_path = Path(tmp) / "persistent_aoi_kb.json"
            kb_path.write_text(json.dumps(records), encoding="utf-8")
            with mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path):
                summary = planner_agent._persistent_kb_summary(
                    failure_mode="g_ng_overlap",
                    k=2,
                )

        self.assertIn("PERSISTENT_KB (top-K matching failure_mode=g_ng_overlap)", summary)
        self.assertIn("TAG_ROLLUP", summary)
        self.assertIn("old overlap plan", summary)
        self.assertIn("new overlap plan", summary)
        self.assertNotIn("threshold plan", summary.split("MATCHED_RECORDS", 1)[-1])
        self.assertNotIn("x" * 200, summary)

    def test_summary_renders_mixed_legacy_and_new_records(self):
        # Legacy record: old {plan, code_snippet, metrics, label} shape.
        legacy = {
            "plan": "legacy_strategy: old approach",
            "code_snippet": "print('full legacy script')",
            "metrics": {"ng_recall": 0.5, "miss_rate": 0.5, "overkill": 0.2,
                        "accuracy": 0.7, "improved": False},
            "label": "failure",
        }
        # New quadruple records.
        new_regressed = {
            "plan": "focal_loss: down-weight easy negatives",
            "code_diff": make_code_diff("a = 1\n", "a = 2\n"),
            "metrics": {"ng_recall": 0.6, "miss_rate": 0.4, "overkill_rate": 0.3,
                        "accuracy": 0.75, "improved": False},
            "tags": ["g_ng_overlap", "regressed"],
        }
        new_improved = {
            "plan": "spatial_attention head",
            "code_diff": make_code_diff("", "model = build()\n"),
            "metrics": {"ng_recall": 0.82, "miss_rate": 0.18, "overkill_rate": 0.1,
                        "accuracy": 0.9, "improved": True},
            "tags": ["g_ng_overlap", "improved"],
        }
        mixed = [legacy, new_regressed, new_improved]

        with tempfile.TemporaryDirectory() as tmp:
            kb_path = Path(tmp) / "persistent_aoi_kb.json"
            kb_path.write_text(json.dumps(mixed), encoding="utf-8")
            with mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path):
                summary = planner_agent._persistent_kb_summary()

        self.assertIn("PERSISTENT_KB_SUMMARY", summary)
        self.assertIn("3 total record(s)", summary)
        self.assertIn("TAG_ROLLUP", summary)
        # The legacy "failure" label is back-compat-mapped to a regressed outcome.
        self.assertIn("failure_mode=", summary)
        self.assertIn("g_ng_overlap", summary)
        # Plans are surfaced; no full script body is dumped.
        self.assertIn("focal_loss", summary)
        self.assertNotIn("full legacy script", summary)

    def test_summary_empty_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb_path = Path(tmp) / "does_not_exist.json"
            with mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path):
                self.assertEqual(planner_agent._persistent_kb_summary(), "")

    def test_load_tried_approaches_threads_current_failure_mode_to_kb_summary(self):
        records = [
            {
                "plan": "threshold plan",
                "code_diff": "+threshold",
                "metrics": {"ng_recall": 0.4},
                "tags": ["threshold_collapse", "regressed"],
            },
        ]
        for idx in range(6):
            records.append({
                "plan": f"overlap plan {idx}",
                "code_diff": f"+overlap {idx}",
                "metrics": {"ng_recall": 0.8 + (idx / 100)},
                "tags": ["g_ng_overlap", "improved"],
            })

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            kb_path = checkpoint_dir / "persistent_aoi_kb.json"
            kb_path.write_text(json.dumps(records), encoding="utf-8")
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(
                    config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried.json"
                ),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path),
            ):
                context = FakeContext({
                    "input_modality": "mono",
                    "diagnosis_brief": {
                        "failure_classification": {"failure_mode": "g_ng_overlap"},
                    },
                })
                summary = planner_agent.load_tried_approaches_fn(context)

        self.assertIn("failure_mode=g_ng_overlap", summary)
        self.assertIn("overlap plan 5", summary)
        self.assertNotIn("threshold plan", summary.split("MATCHED_RECORDS", 1)[-1])


if __name__ == "__main__":
    unittest.main()
