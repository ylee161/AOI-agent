"""Tests for the cross-branch code FUSION operator (MCGS Evolution/Fusion).

Fusion merges the WINNING code blocks of the top-2 refinement_population members
into one unified candidate that is evaluated like any other refinement attempt.

Coverage:
  (a) The trigger fires only with >=2 population members AND an inner-loop
      stagnation, and arms at most once per outer iteration.
  (b) An armed fusion is consumed by the strategy gate into a fusion directive,
      and a fusion directive routes through the normal evaluate_and_update_fn path
      (mock code_runner) — a successful fused result updates refinement_population
      and writes a persistent-KB record whose plan is the fusion directive.
  (c) For mono input the fusion directive and the on-demand script bundle never
      introduce stereo code (prompt-contract guard).

code_runner.run_script (debug pre-check + full run) is mocked so no real training
happens; all checkpoint paths are redirected into a TemporaryDirectory.
"""

import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mle_star_agent import config

fusion = importlib.import_module("mle_star_agent.phases.phase2_refinement.fusion")
evaluator_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.evaluator_agent"
)
planner_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.planner_agent"
)
refinement_coder_agent = importlib.import_module(
    "mle_star_agent.phases.phase2_refinement.refinement_coder_agent"
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


def _member(script: str, overkill: float, miss: float, ng_recall: float = 0.9) -> dict:
    """Build a refinement_population entry the way the evaluator would."""
    return {
        "outer": 0,
        "inner": 0,
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "script": script,
        "metrics": {
            "overkill_rate": overkill,
            "miss_rate": miss,
            "ng_recall": ng_recall,
            "accuracy": 0.9,
        },
        "archive_reason": "lower_overkill_candidate",
    }


def _stagnation_state(num_members: int = 2, no_improve: int = None) -> dict:
    """A state below relaxed acceptance with the inner loop stuck (the same signal
    the ideator's restart branch uses)."""
    if no_improve is None:
        no_improve = config.INNER_STAGNATION_MAX_UNCONSTRAINED
    members = [
        _member(f"# base script\nmodel = build_base()\nx = {i}\n", 0.10 + 0.05 * i, 0.04 * i)
        for i in range(num_members)
    ]
    return {
        "refinement_population": members,
        "no_improve_count": no_improve,
        "current_best_score": 0.0,
        "best_overkill_rate": 1.0,
        "best_miss_rate": 1.0,
        "best_accuracy": 0.0,
        "best_f1": 0.0,
        "outer_iteration": 1,
        "inner_iteration": 0,
    }


# ---------------------------------------------------------------------------
# (a) Trigger gating
# ---------------------------------------------------------------------------
class FusionTriggerTests(unittest.TestCase):
    def test_fires_with_two_members_at_stagnation(self):
        self.assertTrue(fusion.should_trigger_fusion(_stagnation_state(num_members=2)))

    def test_not_fires_with_one_member(self):
        self.assertFalse(fusion.should_trigger_fusion(_stagnation_state(num_members=1)))

    def test_not_fires_without_stagnation(self):
        # Two members but the inner loop has not stagnated (no_improve_count = 0).
        state = _stagnation_state(num_members=2, no_improve=0)
        self.assertFalse(fusion.should_trigger_fusion(state))

    def test_arms_at_most_once_per_outer(self):
        state = _stagnation_state(num_members=2)
        first = fusion.arm_fusion_if_eligible(state, target_outer=2)
        self.assertIn("FUSION_ARMED", first)
        self.assertTrue(state.get("pending_fusion"))
        self.assertTrue(state.get("fusion_attempted_outer_2"))

        # Second arm for the same outer is rejected — at most once per outer.
        state.pop("pending_fusion", None)
        second = fusion.arm_fusion_if_eligible(state, target_outer=2)
        self.assertIn("FUSION_NOT_ARMED", second)
        self.assertFalse(state.get("pending_fusion", False))

    def test_arm_rejected_without_eligibility(self):
        state = _stagnation_state(num_members=1)
        self.assertIn(
            "FUSION_NOT_ARMED", fusion.arm_fusion_if_eligible(state, target_outer=2)
        )
        self.assertFalse(state.get("pending_fusion", False))

    def test_evaluator_arms_fusion_at_stagnation_branch(self):
        """End-to-end: the evaluator's stagnation-restart branch arms fusion when the
        archive holds >=2 members, BEFORE it resets no_improve_count."""
        # A valid (non-degenerate) run that does NOT beat the current best, so it
        # reaches the stagnation branch rather than the predictive-abort path.
        stdout = (
            'METRICS: {"tp": 27, "tn": 21, "fp": 9, "fn": 4, "threshold": 0.2, '
            '"avg_latency_ms": 1, "prob_gap": 0.4, "roc_auc": 0.8}'
        )
        run_result = mock.Mock(
            returncode=0, timed_out=False, duration_ms=60000.0, stdout=stdout, stderr=""
        )
        members = [
            _member("# A\nmodel = a()\n", 0.10, 0.02),
            _member("# B\nmodel = b()\n", 0.15, 0.04),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            patches = (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best.json"),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", checkpoint_dir / "kb.json"),
                mock.patch.object(evaluator_agent.code_runner, "run_script", return_value=run_result),
                # Keep the ideation arXiv fetch out of the test.
                mock.patch.object(evaluator_agent, "trigger_ideation", return_value="IDEATION_SKIPPED"),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                # no_improve already at the stagnation threshold - 1 so this run trips it.
                ctx = FakeContext({
                    "current_script": "# weaker candidate\nx = 1\n",
                    "best_pipeline_script": "# prev best\ny = 1\n",
                    "outer_iteration": 1,
                    "inner_iteration": 1,
                    # Current best dominates the run (ng_recall 0.95 > 0.871) so it does
                    # NOT improve; overkill 0.30 is below relaxed acceptance but under the
                    # 0.50 high-overkill restart threshold, so Priority 1b (stagnation)
                    # fires rather than Priority 1a.
                    "current_best_score": 0.95,
                    "best_overkill_rate": 0.30,
                    "best_miss_rate": 0.05,
                    "best_accuracy": 0.85,
                    "best_f1": 0.90,
                    "no_improve_count": config.INNER_STAGNATION_MAX_UNCONSTRAINED - 1,
                    "token_count": 0,
                    "refinement_population": members,
                    "selected_refinement_strategy": "focal_loss: x",
                    "warm_restart_attempted_best_9d503588": True,
                })
                out = evaluator_agent.evaluate_and_update_fn(ctx)

        self.assertIn("INNER_STAGNATION", out)
        self.assertIn("FUSION_ARMED", out)
        self.assertTrue(ctx.state.get("pending_fusion"))
        # Armed for the upcoming outer iteration (n+1 == 2).
        self.assertTrue(ctx.state.get("fusion_attempted_outer_2"))


# ---------------------------------------------------------------------------
# (b) Consumption by the gate + routing through the normal evaluate path
# ---------------------------------------------------------------------------
class FusionStrategyGateTests(unittest.TestCase):
    def test_gate_installs_fusion_directive_from_pending_flag(self):
        members = [
            _member("# base\nmodel = base()\n", 0.10, 0.02),
            _member("# donor\nmodel = donor()\n", 0.20, 0.05),
        ]
        ctx = FakeContext({
            "pending_fusion": True,
            "refinement_population": members,
            "tried_approaches": [],
            "outer_iteration": 2,
            "inner_iteration": 0,
            "input_modality": "stereo",
        })
        status = planner_agent.ensure_selected_strategy_fn(ctx)

        self.assertIn("FUSION_STRATEGY_INSTALLED", status)
        self.assertTrue(
            ctx.state["selected_refinement_strategy"].startswith(fusion.FUSION_DIRECTIVE_MARKER)
        )
        fp = ctx.state["selected_strategy_fingerprint"]
        self.assertEqual(fp["target_component"], fusion.FUSION_TARGET_COMPONENT)
        self.assertTrue(fp["mechanism_class"].startswith("fusion_"))
        # The flag is consumed so it cannot leak into later cycles.
        self.assertFalse(ctx.state.get("pending_fusion", False))

    def test_gate_skips_already_failed_fusion_pair(self):
        """A fusion pair that already failed is not retried — the gate falls through
        to the normal strategy logic (respecting the dedup contract)."""
        members = [
            _member("# base\nmodel = base()\n", 0.10, 0.02),
            _member("# donor\nmodel = donor()\n", 0.20, 0.05),
        ]
        fingerprint = fusion.fusion_fingerprint(members[0], members[1])
        ctx = FakeContext({
            "pending_fusion": True,
            "refinement_population": members,
            # Same pair recorded as a prior P0/P1 failure.
            "tried_approaches": [{
                "result": {"improved": False},
                "failure_reason": "no_improvement",
                "strategy_fingerprint": fingerprint,
            }],
            "outer_iteration": 2,
            "inner_iteration": 0,
            "diagnosis_report": {"target_component": "weighted_loss"},
        })
        status = planner_agent.ensure_selected_strategy_fn(ctx)

        # Fell through to the normal fallback path; no fusion directive installed.
        self.assertNotIn("FUSION_STRATEGY_INSTALLED", status)
        self.assertFalse(
            (ctx.state.get("selected_refinement_strategy") or "").startswith(
                fusion.FUSION_DIRECTIVE_MARKER
            )
        )
        self.assertFalse(ctx.state.get("pending_fusion", False))

    def test_fusion_directive_routes_through_evaluate_and_updates_population_and_kb(self):
        fused_script = "# FUSED candidate\nmodel = base_with_donor_loss()\nprint('ok')\n"
        directive = fusion.build_fusion_directive(
            {"input_modality": "stereo"},
            [
                _member("# base\nmodel = base()\n", 0.10, 0.02),
                _member("# donor\nmodel = donor()\n", 0.20, 0.05),
            ],
        )
        # overkill = fp/(fp+tn) = 9/30 = 0.30 (> 0.12), so multiseed is not triggered.
        # prob_gap + a 60s runtime keep the run out of the degenerate-metrics guard so
        # the fused candidate is archived as a real population member.
        stdout = (
            'METRICS: {"tp": 27, "tn": 21, "fp": 9, "fn": 4, "threshold": 0.2, '
            '"avg_latency_ms": 1, "prob_gap": 0.4, "roc_auc": 0.8}'
        )
        run_result = mock.Mock(
            returncode=0, timed_out=False, duration_ms=60000.0, stdout=stdout, stderr=""
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            kb_path = checkpoint_dir / "persistent_aoi_kb.json"
            patches = (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best.json"),
                mock.patch.object(config, "CKPT_PERSISTENT_KB", kb_path),
                mock.patch.object(evaluator_agent.code_runner, "run_script", return_value=run_result),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                ctx = FakeContext({
                    "best_pipeline_script": "# prev best\ny = 1\n",
                    "current_script": fused_script,
                    "outer_iteration": 2,
                    "inner_iteration": 0,
                    "current_best_score": 0.0,
                    "best_overkill_rate": 1.0,
                    "best_miss_rate": 1.0,
                    "best_accuracy": 0.0,
                    "best_f1": 0.0,
                    "no_improve_count": 0,
                    "token_count": 0,
                    "refinement_population": [],
                    "selected_refinement_strategy": directive,
                    "selected_strategy_fingerprint": {
                        "target_component": "fusion", "mechanism_class": "fusion_abc_def"
                    },
                    "diagnosis_brief": {"failure_classification": {"failure_mode": "g_ng_overlap"}},
                })
                evaluator_agent.evaluate_and_update_fn(ctx)

                # Population updated with the fused script.
                population = ctx.state.get("refinement_population", [])
                self.assertTrue(
                    any(e.get("script") == fused_script for e in population),
                    "fused script was not archived into refinement_population",
                )
                # Persistent-KB record written with the fusion directive as its plan.
                self.assertTrue(kb_path.is_file())
                records = json.loads(kb_path.read_text(encoding="utf-8"))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["plan"], directive)
                self.assertTrue(records[0]["plan"].startswith(fusion.FUSION_DIRECTIVE_MARKER))


# ---------------------------------------------------------------------------
# (c) Mono input never yields stereo code in the fusion contract
# ---------------------------------------------------------------------------
class FusionModalityTests(unittest.TestCase):
    def _members(self):
        # A donor that DOES contain stereo code — fusion must still keep mono output.
        return [
            _member("# base mono\nimg = load(path)\n", 0.10, 0.02),
            _member("# donor\nimg_l = load(l)\nimg_r = load(r)\n", 0.20, 0.05),
        ]

    def test_mono_directive_forbids_stereo(self):
        directive = fusion.build_fusion_directive({"input_modality": "mono"}, self._members())
        self.assertIn("mono", directive.lower())
        self.assertIn("NEVER introduce stereo", directive)
        # The stereo-branch instruction must not appear for mono input.
        self.assertNotIn("keep the existing stereo handling", directive)

    def test_stereo_directive_keeps_stereo(self):
        directive = fusion.build_fusion_directive({"input_modality": "stereo"}, self._members())
        self.assertIn("keep the existing stereo handling", directive)

    def test_mono_script_bundle_carries_mono_contract(self):
        members = self._members()
        bundle = fusion.render_fusion_scripts({"input_modality": "mono"}, members)
        # Mono prohibition travels with the fetched scripts.
        self.assertIn("NEVER introduce stereo", bundle)
        # Both scripts are present for the coder to fuse.
        self.assertIn("FUSION_SCRIPT_0", bundle)
        self.assertIn("FUSION_SCRIPT_1", bundle)

    def test_load_fusion_scripts_tool_mono_contract(self):
        ctx = FakeContext({
            "input_modality": "mono",
            "refinement_population": self._members(),
        })
        out = refinement_coder_agent.load_fusion_scripts_fn(ctx)
        self.assertIn("NEVER introduce stereo", out)
        self.assertIn("FUSION_SCRIPT_0", out)
        self.assertIn("FUSION_SCRIPT_1", out)

    def test_load_fusion_scripts_tool_requires_two_members(self):
        ctx = FakeContext({
            "input_modality": "mono",
            "refinement_population": [self._members()[0]],
        })
        out = refinement_coder_agent.load_fusion_scripts_fn(ctx)
        self.assertIn("FUSION_SCRIPTS_UNAVAILABLE", out)


if __name__ == "__main__":
    unittest.main()
