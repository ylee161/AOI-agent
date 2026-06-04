import os
import unittest
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from mle_star_agent.phases.phase2_refinement import ideator_agent


class FakeContext:
    def __init__(self, state):
        self.state = state


class SeekHelpTests(unittest.TestCase):
    def test_returns_hints_and_preserves_retrieved_technique_hints(self):
        original_hints = ["existing planner hint"]
        state = {
            "diagnosis_brief": {
                "failure_classification": {"failure_mode": "low_capacity_miss"}
            },
            "current_best_score": 0.42,
            "best_overkill_rate": 0.18,
            "best_miss_rate": 0.58,
            "retrieved_technique_hints": original_hints,
        }
        expected = [
            "focal loss with defect-class alpha",
            "class-balanced sampler for NG recall",
            "hard example replay for misses",
        ]

        with mock.patch.object(
            ideator_agent,
            "generate_technique_hints",
            return_value=expected,
        ) as generate:
            result = ideator_agent.seek_help_fn(
                FakeContext(state),
                problem="planner would otherwise repeat focal_loss fingerprint",
            )

        self.assertIn("SEEK_HELP_HINTS", result)
        for hint in expected:
            self.assertIn(hint, result)
        self.assertEqual(state["retrieved_technique_hints"], original_hints)
        self.assertEqual(state["seek_help_hints"], expected)
        self.assertEqual(len(state["seek_help_hints"]), 3)
        generate.assert_called_once()
        self.assertIn("low_capacity_miss", generate.call_args.args[0])
        self.assertIn("repeat focal_loss fingerprint", generate.call_args.args[0])
        self.assertEqual(generate.call_args.args[1:], (0.18, 0.58, 0.42))

    def test_arxiv_fetch_failure_uses_fallback_and_never_raises(self):
        state = {
            "diagnosis_report": {
                "failure_classification": {"failure_mode": "threshold_collapse"}
            },
            "retrieved_technique_hints": ["macro ideation hint"],
        }

        with mock.patch.object(ideator_agent.httpx, "get", side_effect=RuntimeError("offline")):
            result = ideator_agent.seek_help_fn(
                FakeContext(state),
                problem="threshold-only candidates are banned",
            )

        self.assertIn("SEEK_HELP_HINTS", result)
        self.assertEqual(state["retrieved_technique_hints"], ["macro ideation hint"])
        self.assertIn("seek_help_hints", state)
        self.assertGreaterEqual(len(state["seek_help_hints"]), 3)
        self.assertLessEqual(len(state["seek_help_hints"]), 5)

    def test_empty_or_none_state_never_raises(self):
        empty_result = ideator_agent.seek_help_fn(FakeContext({}), problem="")
        none_result = ideator_agent.seek_help_fn(FakeContext(None), problem=None)

        self.assertIn("SEEK_HELP_HINTS", empty_result)
        self.assertIn("SEEK_HELP_HINTS", none_result)

    def test_problem_text_is_folded_into_arxiv_query_emphasis(self):
        captured_queries = []

        def fake_fetch(query):
            captured_queries.append(query)
            return []

        with mock.patch.object(ideator_agent, "_fetch_arxiv_hints", side_effect=fake_fetch):
            ideator_agent.generate_technique_hints(
                "low_capacity_miss | planner roadblock: avoid repeating focal loss",
                0.1,
                0.7,
                0.3,
            )

        self.assertEqual(len(captured_queries), 1)
        self.assertIn("avoid+repeating+focal+loss", captured_queries[0])
