import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


class AcceptanceScoringTests(unittest.TestCase):
    def test_reducing_large_overkill_is_improvement_when_acceptance_distance_drops(self):
        from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement

        current = {
            "accuracy": 0.6229,
            "ng_recall": 0.9355,
            "miss_rate": 0.0645,
            "overkill_rate": 0.70,
            "f1": 0.716,
        }
        new = {
            "accuracy": 0.70,
            "ng_recall": 0.93,
            "miss_rate": 0.07,
            "overkill_rate": 0.50,
            "f1": 0.75,
        }

        self.assertTrue(is_acceptance_improvement(new, current))

    def test_both_fail_prefers_lower_overkill_over_perfect_recall_high_overkill(self):
        from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement

        current = {
            "accuracy": 0.65,
            "ng_recall": 1.0,
            "miss_rate": 0.0,
            "overkill_rate": 0.70,
            "f1": 0.75,
        }
        new = {
            "accuracy": 0.72,
            "ng_recall": 0.97,
            "miss_rate": 0.03,
            "overkill_rate": 0.20,
            "f1": 0.82,
        }

        self.assertTrue(is_acceptance_improvement(new, current))

    def test_passing_relaxed_acceptance_beats_higher_recall_with_bad_overkill(self):
        from mle_star_agent.shared.acceptance_scoring import is_acceptance_improvement

        current = {
            "accuracy": 0.90,
            "ng_recall": 1.0,
            "miss_rate": 0.0,
            "overkill_rate": 0.40,
            "f1": 0.80,
        }
        new = {
            "accuracy": 0.93,
            "ng_recall": 0.97,
            "miss_rate": 0.03,
            "overkill_rate": 0.08,
            "f1": 0.90,
        }

        self.assertTrue(is_acceptance_improvement(new, current))


class AveragedCandidateSelectionTests(unittest.TestCase):
    def test_l0_selection_prefers_better_average_over_better_single_seed(self):
        from mle_star_agent.phases.phase1_init.merger_agent import _select_best_successful_candidate

        noisy_single_seed_winner = {
            "name": "noisy_single_seed_winner",
            "status": "success",
            "metrics": {
                "accuracy": 0.94,
                "ng_recall": 1.0,
                "miss_rate": 0.0,
                "overkill_rate": 0.09,
                "f1": 0.91,
            },
            "selection_evaluation": {
                "status": "success",
                "metrics": {
                    "accuracy": 0.90,
                    "ng_recall": 0.94,
                    "miss_rate": 0.06,
                    "overkill_rate": 0.09,
                    "f1": 0.86,
                },
            },
        }
        better_average = {
            "name": "better_average",
            "status": "success",
            "metrics": {
                "accuracy": 0.91,
                "ng_recall": 0.97,
                "miss_rate": 0.03,
                "overkill_rate": 0.08,
                "f1": 0.88,
            },
            "selection_evaluation": {
                "status": "success",
                "metrics": {
                    "accuracy": 0.93,
                    "ng_recall": 0.98,
                    "miss_rate": 0.02,
                    "overkill_rate": 0.08,
                    "f1": 0.90,
                },
            },
        }

        best = _select_best_successful_candidate([noisy_single_seed_winner, better_average])

        self.assertEqual(best["name"], "better_average")
        self.assertEqual(best["selection_metrics"]["miss_rate"], 0.02)

    def test_incomplete_average_does_not_override_single_seed_metrics(self):
        from mle_star_agent.shared.selection_metrics import selection_metrics_for_record

        candidate = {
            "metrics": {
                "accuracy": 0.91,
                "ng_recall": 0.97,
                "miss_rate": 0.03,
                "overkill_rate": 0.08,
                "f1": 0.88,
            },
            "selection_evaluation": {
                "status": "failed",
                "metrics": {
                    "accuracy": 0.99,
                    "ng_recall": 1.0,
                    "miss_rate": 0.0,
                    "overkill_rate": 0.0,
                    "f1": 1.0,
                },
            },
        }

        self.assertEqual(selection_metrics_for_record(candidate)["miss_rate"], 0.03)


class AblationCompletenessTests(unittest.TestCase):
    def test_ablation_plan_covers_overkill_and_calibration_failure_modes(self):
        from mle_star_agent.phases.phase2_refinement.ablation_agent import ABLATION_VARIANTS

        names = {v["name"] for v in ABLATION_VARIANTS}

        self.assertGreaterEqual(len(ABLATION_VARIANTS), 8)
        self.assertIn("threshold_acceptance_distance", names)
        self.assertIn("fp_penalty_loss", names)
        self.assertIn("temperature_scaling", names)
        self.assertIn("lot_normalization", names)
        # Factorial-ablation refactor (commit 5967065) replaced the standalone
        # "training_schedule" probe with the "optimizer_lr_schedule" probe.
        self.assertIn("optimizer_lr_schedule", names)

    def test_partial_ablation_results_are_not_complete(self):
        from mle_star_agent.phases.phase2_refinement.ablation_agent import (
            _is_complete_ablation_results,
        )

        partial = [
            {"variant_index": 1, "status": "success"},
            {"variant_index": 2, "status": "success"},
        ]

        self.assertFalse(_is_complete_ablation_results(partial))

    def test_all_ablation_variant_indices_are_complete(self):
        from mle_star_agent.phases.phase2_refinement.ablation_agent import (
            NUM_ABLATION_VARIANTS,
            _is_complete_ablation_results,
        )

        complete = [
            {"variant_index": i, "status": "success"}
            for i in range(NUM_ABLATION_VARIANTS)
        ]

        self.assertTrue(_is_complete_ablation_results(complete))

    def test_skipped_ablation_variants_fill_their_slot(self):
        # Under factorial + targeted ablation (commit 5967065), a "skipped" variant
        # is a legitimate, intended outcome (mono drops stereo-off; targeted later
        # iterations skip low-impact variants). It still fills its slot, so the
        # SLOT-completeness check counts it — otherwise targeted iterations could
        # never complete and would loop forever. The guard against running diagnosis
        # on zero real evidence lives in the diagnosis checkpoint gate + empty-ranking
        # scorer, not in _is_complete_ablation_results.
        from mle_star_agent.phases.phase2_refinement.ablation_agent import (
            NUM_ABLATION_VARIANTS,
            _is_complete_ablation_results,
        )

        skipped = [
            {"variant_index": i, "status": "skipped"}
            for i in range(NUM_ABLATION_VARIANTS)
        ]

        self.assertTrue(_is_complete_ablation_results(skipped))

        # But a set still MISSING variant indices is never slot-complete, even if
        # every present result was skipped — this is the real completeness guard.
        missing_one = skipped[:-1]
        self.assertFalse(_is_complete_ablation_results(missing_one))

    def test_ablation_variant_generation_uses_pro_model(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.ablation_agent")

        variant_agent = module.ablation_sequential.sub_agents[0]

        self.assertIs(variant_agent.model, config.MODEL_PRO)

    def test_ablation_scripts_embed_variant_name_for_validator_policy(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.ablation_agent")
        variant_index = next(
            i
            for i, variant in enumerate(module.ABLATION_VARIANTS)
            if variant["name"] == "threshold_acceptance_distance"
        )
        variant_agent = module._variant_step_agents[variant_index]

        self.assertIn("ABLATION_VARIANT_NAME", variant_agent.instruction)


class CodeValidatorPolicyTests(unittest.TestCase):
    def test_threshold_acceptance_distance_has_narrow_validator_exception(self):
        import importlib

        module = importlib.import_module("mle_star_agent.guards.code_validator_agent")
        instruction = module._INSTRUCTION

        self.assertIn("threshold_acceptance_distance", instruction)
        self.assertIn("acceptance-distance minimization is allowed", instruction)
        self.assertIn("For all other scripts", instruction)
        self.assertIn("strict two-stage priority", instruction)
        self.assertIn("Do NOT use acceptance-distance averaging", instruction)


class ThresholdCurveEvidenceTests(unittest.TestCase):
    def test_evaluator_stores_threshold_curve_for_diagnosis(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        class FakeState(dict):
            pass

        class FakeActions:
            escalate = False

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
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
                self.actions = FakeActions()

        stdout = "\n".join([
            'METRICS: {"tp": 29, "tn": 10, "fp": 20, "fn": 2, "threshold": 0.2, "avg_latency_ms": 1}',
            'THRESHOLD_CURVE: [{"t": 0.1, "recall": 1.0, "miss_rate": 0.0, "overkill": 0.8, "accuracy": 0.6}, {"t": 0.4, "recall": 0.97, "miss_rate": 0.03, "overkill": 0.2, "accuracy": 0.85}]',
        ])
        result = mock.Mock(returncode=0, timed_out=False, duration_ms=12.0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
                mock.patch.object(module.code_runner, "run_script", return_value=result),
            ):
                context = FakeContext()
                module.evaluate_and_update_fn(context)

                self.assertEqual(context.state["latest_threshold_curve"][1]["t"], 0.4)
                saved = json.loads((checkpoint_dir / "error_analysis_0_0.json").read_text())
                self.assertEqual(saved["threshold_curve"][0]["overkill"], 0.8)

    def test_promising_improvement_requires_multiseed_confirmation(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        promising = {
            "accuracy": 0.84,
            "ng_recall": 0.97,
            "miss_rate": 0.03,
            "overkill_rate": 0.12,
            "f1": 0.86,
        }
        loose_overkill = {
            "accuracy": 0.82,
            "ng_recall": 0.97,
            "miss_rate": 0.03,
            "overkill_rate": 0.18,
            "f1": 0.84,
        }
        still_bad = {
            "accuracy": 0.70,
            "ng_recall": 0.97,
            "miss_rate": 0.03,
            "overkill_rate": 0.50,
            "f1": 0.75,
        }

        self.assertTrue(module._requires_multiseed_confirmation(promising, improved=True))
        self.assertFalse(module._requires_multiseed_confirmation(loose_overkill, improved=True))
        self.assertFalse(module._requires_multiseed_confirmation(still_bad, improved=True))
        self.assertFalse(module._requires_multiseed_confirmation(promising, improved=False))

    def test_phase2_uses_average_before_accepting_single_seed_improvement(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        current = {
            "accuracy": 0.93,
            "ng_recall": 0.98,
            "miss_rate": 0.02,
            "overkill_rate": 0.08,
            "f1": 0.90,
        }
        noisy_single_seed = {
            "accuracy": 0.94,
            "ng_recall": 1.0,
            "miss_rate": 0.0,
            "overkill_rate": 0.08,
            "f1": 0.92,
        }
        worse_average = {
            "accuracy": 0.90,
            "ng_recall": 0.94,
            "miss_rate": 0.06,
            "overkill_rate": 0.08,
            "f1": 0.86,
        }

        improved, selected_metrics, status = module._confirm_improvement_with_selection_average(
            script="print('stub')",
            metrics=noisy_single_seed,
            current_metrics=current,
            initially_improved=True,
            run_average=lambda _script: (worse_average, [{"seed": 42}, {"seed": 101}, {"seed": 202}]),
        )

        self.assertFalse(improved)
        self.assertEqual(selected_metrics["miss_rate"], 0.06)
        self.assertEqual(status["status"], "success")

    def test_refinement_population_archives_lower_overkill_nonbest_candidate(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        state = {}
        current = {
            "accuracy": 0.80,
            "ng_recall": 1.0,
            "miss_rate": 0.0,
            "overkill_rate": 0.70,
            "f1": 0.82,
        }
        candidate = {
            "accuracy": 0.82,
            "ng_recall": 0.94,
            "miss_rate": 0.06,
            "overkill_rate": 0.18,
            "f1": 0.84,
            "threshold": 0.55,
        }

        module._update_refinement_population(
            state,
            script="print('candidate')",
            metrics=candidate,
            current_metrics=current,
            improved=False,
            outer_iteration=0,
            inner_iteration=2,
        )

        self.assertEqual(len(state["refinement_population"]), 1)
        archived = state["refinement_population"][0]
        self.assertEqual(archived["metrics"]["overkill_rate"], 0.18)
        self.assertEqual(archived["archive_reason"], "lower_overkill_candidate")

    def test_diagnosis_brief_includes_threshold_curve(self):
        from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief

        curve = [
            {"t": 0.1, "recall": 1.0, "miss_rate": 0.0, "overkill": 0.8, "accuracy": 0.6},
            {"t": 0.4, "recall": 0.97, "miss_rate": 0.03, "overkill": 0.2, "accuracy": 0.85},
        ]

        brief = generate_diagnosis_brief(
            [],
            {
                "accuracy": 0.60,
                "ng_recall": 0.90,
                "miss_rate": 0.10,
                "overkill_rate": 0.70,
                "f1": 0.65,
            },
            threshold_curve=curve,
        )

        self.assertEqual(brief["threshold_curve"][1]["t"], 0.4)
        self.assertEqual(brief["threshold_curve_summary"]["points"], 2)
        self.assertEqual(brief["threshold_curve_summary"]["best_threshold"], 0.4)

    def test_high_overkill_without_threshold_escape_forces_stereo_target(self):
        from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief

        curve = [
            {"t": 0.1, "recall": 1.0, "miss_rate": 0.0, "overkill": 0.8, "accuracy": 0.6},
            {"t": 0.4, "recall": 0.97, "miss_rate": 0.03, "overkill": 0.35, "accuracy": 0.78},
        ]

        brief = generate_diagnosis_brief(
            [],
            {
                "accuracy": 0.60,
                "ng_recall": 0.94,
                "miss_rate": 0.06,
                "overkill_rate": 0.70,
                "f1": 0.65,
            },
            calibration_stats={"G_prob_mean": 0.75, "NG_prob_mean": 0.90},
            threshold_curve=curve,
        )

        failure = brief["failure_classification"]
        self.assertEqual(failure["recommended_target"], "stereo_fusion")
        self.assertIn("Thresholding cannot solve", failure["recommended_action"])


class StrategyFingerprintTests(unittest.TestCase):
    def test_low_confidence_diagnosis_requires_preflight_probe_before_normal_strategy(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "diagnosis_report": {
                        "failure_classification": {
                            "failure_mode": "g_ng_overlap",
                            "confidence": "low",
                        }
                    },
                })

        context = FakeContext()
        result = module.save_strategy_candidates_fn(
            context,
            strategy_a="fp_penalty_loss: add dynamic G false-positive penalty",
            strategy_b="stereo_diff_features: add abs L-R channel",
            strategy_c="temperature_scaling: calibrate logits",
            selected="a",
            selection_reason="Diagnosis points at weighted loss.",
            strategy_a_target_component="weighted_loss",
            strategy_a_mechanism_class="fp_penalty_loss",
            strategy_b_target_component="stereo_fusion",
            strategy_b_mechanism_class="stereo_diff_features",
            strategy_c_target_component="calibration",
            strategy_c_mechanism_class="temperature_scaling",
        )

        self.assertIn("PREFLIGHT_PROBE_REQUIRED", result)
        self.assertNotIn("selected_refinement_strategy", context.state)

    def test_low_confidence_diagnosis_allows_normal_strategy_after_probe_evidence(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 1,
                    "diagnosis_report": {
                        "failure_classification": {
                            "failure_mode": "g_ng_overlap",
                            "confidence": "low",
                        }
                    },
                    "latest_probe_metrics": {
                        "ng_recall": 0.95,
                        "overkill_rate": 0.20,
                        "probability_gap": 0.12,
                    },
                })

        context = FakeContext()
        result = module.save_strategy_candidates_fn(
            context,
            strategy_a="stereo_diff_features: add abs L-R channel",
            strategy_b="fp_penalty_loss: add dynamic G false-positive penalty",
            strategy_c="temperature_scaling: calibrate logits",
            selected="a",
            selection_reason="Probe shows separability gap.",
            strategy_a_target_component="stereo_fusion",
            strategy_a_mechanism_class="stereo_diff_features",
            strategy_b_target_component="weighted_loss",
            strategy_b_mechanism_class="fp_penalty_loss",
            strategy_c_target_component="calibration",
            strategy_c_mechanism_class="temperature_scaling",
        )

        self.assertIn("Strategy candidates saved", result)
        self.assertEqual(context.state["selected_strategy_fingerprint"]["target_component"], "stereo_fusion")

    def test_strategy_gate_fallback_uses_preflight_probe_for_low_confidence_diagnosis(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "diagnosis_report": {
                        "target_component": "weighted_loss",
                        "recommended_changes": "Add FP penalty.",
                        "failure_classification": {"confidence": "low"},
                    },
                })

        context = FakeContext()
        result = module.ensure_selected_strategy_fn(context)

        self.assertIn("preflight_probe", result)
        self.assertEqual(
            context.state["selected_strategy_fingerprint"]["target_component"],
            "preflight_probe",
        )

    def test_planner_recovers_missing_failed_refinement_history_from_checkpoints(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 3,
                })

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            history_path = checkpoint_dir / "tried_approaches.json"
            history_path.write_text(json.dumps({"tried_approaches": []}))
            (checkpoint_dir / "refinement_0_2_pending.json").write_text(json.dumps({
                "strategy_executed": "weighted_loss_asymmetric_bce",
                "changes_applied": ["dynamic g_fp_weight"],
            }))
            (checkpoint_dir / "refinement_0_2.json").write_text(json.dumps({
                "outer_iteration": 0,
                "inner_iteration": 2,
                "returncode": 0,
                "improved": False,
                "failure_reason": "recall_regression",
                "new_score": 0.2903225806451613,
                "new_overkill": 0.03333333333333333,
                "metrics": {
                    "accuracy": 0.6229508196721312,
                    "ng_recall": 0.2903225806451613,
                    "miss_rate": 0.7096774193548387,
                    "overkill_rate": 0.03333333333333333,
                },
                "diagnosis_evidence": {
                    "target_component": "weighted_loss",
                    "strategy": "asymmetric_bce_loss with dynamic g_fp_weight",
                    "divergence": "CRITICAL: Over-aggressive FP penalty destroyed NG recall",
                },
            }))

            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir), \
                 mock.patch.object(config, "CKPT_TRIED_APPROACHES", history_path):
                context = FakeContext()
                result = module.load_tried_approaches_fn(context)
                saved = json.loads(history_path.read_text())

        self.assertIn("RECOVERED_FROM_REFINEMENT_CHECKPOINTS: 1", result)
        recovered = context.state["tried_approaches"][0]
        self.assertEqual(recovered["target_component"], "weighted_loss")
        self.assertEqual(recovered["strategy_fingerprint"]["mechanism_class"], "fp_penalty_loss")
        self.assertFalse(recovered["result"]["improved"])
        self.assertEqual(recovered["failure_reason"], "recall_regression")
        self.assertEqual(saved["tried_approaches"][0]["result"]["ng_recall"], 0.2903)

    def test_planner_recovers_refinement_history_from_retry_archive(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            history_path = checkpoint_dir / "tried_approaches.json"
            archive_dir = checkpoint_dir / "retry_archives" / "attempt_1"
            archive_dir.mkdir(parents=True)
            (archive_dir / "refinement_0_0.json").write_text(json.dumps({
                "outer_iteration": 0,
                "inner_iteration": 1,
                "returncode": 0,
                "improved": False,
                "new_score": 0.7419354838709677,
                "new_overkill": 0.5333333333333333,
                "metrics": {
                    "accuracy": 0.6065573770491803,
                    "ng_recall": 0.7419354838709677,
                    "miss_rate": 0.25806451612903225,
                    "overkill_rate": 0.5333333333333333,
                },
            }))

            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir), \
                 mock.patch.object(config, "CKPT_TRIED_APPROACHES", history_path):
                context = FakeContext()
                result = module.load_tried_approaches_fn(context)

        self.assertIn("RECOVERED_FROM_REFINEMENT_CHECKPOINTS: 1", result)
        self.assertEqual(context.state["tried_approaches"][0]["source"], "refinement_checkpoint")

    def test_planner_rejects_selected_duplicate_failed_strategy_fingerprint(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 1,
                    "tried_approaches": [{
                        "target_component": "calibration",
                        "strategy_fingerprint": {
                            "target_component": "calibration",
                            "mechanism_class": "temperature_scaling",
                        },
                        "result": {"improved": False},
                    }],
                })

        context = FakeContext()
        result = module.save_strategy_candidates_fn(
            context,
            strategy_a="temperature_scaling: calibrate logits",
            strategy_b="isotonic_calibration: monotonic calibration",
            strategy_c="label_smoothing: soften labels",
            selected="a",
            selection_reason="Try calibration.",
            strategy_a_target_component="calibration",
            strategy_a_mechanism_class="temperature_scaling",
            strategy_b_target_component="calibration",
            strategy_b_mechanism_class="isotonic_calibration",
            strategy_c_target_component="calibration",
            strategy_c_mechanism_class="label_smoothing",
        )

        self.assertIn("DUPLICATE_STRATEGY_REJECTED", result)
        self.assertNotIn("selected_refinement_strategy", context.state)


class TargetedAblationPolicyTests(unittest.TestCase):
    def test_targeted_ablation_skips_irrelevant_variant_after_iteration_zero(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.ablation_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 1,
                    "diagnosis_brief": {
                        "failure_classification": {
                            "failure_mode": "threshold_collapse",
                            "confidence": "high",
                        }
                    },
                })

        context = FakeContext()
        run_no_augmentation = module._make_run_variant_fn(3)
        result = run_no_augmentation(context)

        self.assertIn("targeted ablation", result)
        self.assertEqual(
            context.state["ablation_result_3"]["reason"],
            "targeted_ablation_skipped_not_relevant",
        )


class TargetRotationPolicyTests(unittest.TestCase):
    def test_planner_rejects_stale_target_after_repeated_failed_attempts(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 1,
                    "inner_iteration": 6,
                    "tried_approaches": [
                        {
                            "target_component": "calibration",
                            "result": {"improved": False},
                        }
                        for _ in range(6)
                    ],
                })

        context = FakeContext()
        result = module.save_strategy_candidates_fn(
            context,
            strategy_a="isotonic_calibration: monotonic probability calibration",
            strategy_b="stereo_diff_features: add abs difference channel",
            strategy_c="fp_penalty_loss: penalise G false positives",
            selected="a",
            selection_reason="Try a new calibration mechanism.",
            strategy_a_target_component="calibration",
            strategy_a_mechanism_class="isotonic_calibration",
            strategy_b_target_component="stereo_fusion",
            strategy_b_mechanism_class="stereo_diff_features",
            strategy_c_target_component="weighted_loss",
            strategy_c_mechanism_class="fp_penalty_loss",
        )

        self.assertIn("ROTATION_LOCK_REJECTED", result)
        self.assertNotIn("selected_refinement_strategy", context.state)


class InnerStagnationPolicyTests(unittest.TestCase):
    def test_unconstrained_stagnation_restarts_outer_loop_without_stop_flag(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        class FakeState(dict):
            pass

        class FakeActions:
            escalate = False

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "current_script": "print('stub')",
                    "outer_iteration": 0,
                    "inner_iteration": 2,
                    "current_best_score": 0.90,
                    "best_miss_rate": 0.10,
                    "best_overkill_rate": 0.20,
                    "best_accuracy": 0.60,
                    "best_f1": 0.65,
                    "no_improve_count": 4,
                    "token_count": 0,
                    "warm_restart_attempted_best_e3b0c442": True,
                    "refinement_plan": {
                        "target_component": "calibration",
                        "changes_summary": "no metric movement",
                    },
                })
                self.actions = FakeActions()

        # prob_gap + a >=30s runtime keep this realistic full run out of the
        # degenerate-metrics guard (metric_guard rejects prob_gap==0 and runtime
        # < 30s as dummy-split poisoning) so it reaches the stagnation logic.
        stdout = 'METRICS: {"tp": 27, "tn": 21, "fp": 9, "fn": 4, "threshold": 0.2, "avg_latency_ms": 1, "prob_gap": 0.4}'
        result = mock.Mock(returncode=0, timed_out=False, duration_ms=60000.0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
                mock.patch.object(module.code_runner, "run_script", return_value=result),
            ):
                context = FakeContext()
                message = module.evaluate_and_update_fn(context)

        self.assertIn("INNER_STAGNATION", message)
        self.assertTrue(context.actions.escalate)
        self.assertEqual(context.state["outer_iteration"], 1)
        self.assertEqual(context.state["inner_iteration"], 0)
        self.assertFalse(context.state.get("stop_outer_loop", False))
        self.assertEqual(context.state["no_improve_count"], 0)

    def test_unconstrained_stagnation_threshold_is_short_enough_to_rediagnose(self):
        from mle_star_agent import config
        from mle_star_agent.shared.loop_guard import should_restart_inner_for_stagnation

        state = {
            "current_best_score": 0.90,
            "best_miss_rate": 0.10,
            "best_overkill_rate": 0.70,
            "best_accuracy": 0.60,
            "best_f1": 0.65,
            "no_improve_count": 5,
        }

        self.assertEqual(config.INNER_STAGNATION_MAX_UNCONSTRAINED, 5)
        self.assertTrue(should_restart_inner_for_stagnation(state))


class Phase2EarlyStopPolicyTests(unittest.TestCase):
    def test_no_improve_does_not_stop_phase2_when_best_is_still_below_acceptance(self):
        from mle_star_agent.shared.loop_guard import should_exit_outer

        state = {
            "outer_iteration": 0,
            "no_improve_count": 2,
            "token_count": 0,
            "current_best_score": 0.9354838709677419,
            "best_overkill_rate": 0.7,
            "best_accuracy": 0.6229508196721312,
            "best_f1": 0.7160493827160493,
        }

        self.assertFalse(should_exit_outer(state))

    def test_no_improve_uses_extended_patience_after_relaxed_acceptance_is_reached(self):
        from mle_star_agent.shared.loop_guard import should_exit_outer

        state = {
            "outer_iteration": 0,
            "no_improve_count": 2,
            "token_count": 0,
            "current_best_score": 0.9701,
            "best_overkill_rate": 0.08,
            "best_accuracy": 0.93,
            "best_f1": 0.9,
        }

        self.assertFalse(should_exit_outer(state))
        state["no_improve_count"] = 5
        self.assertTrue(should_exit_outer(state))

    def test_no_improve_can_stop_phase2_after_final_acceptance_is_reached(self):
        from mle_star_agent.shared.loop_guard import should_exit_outer

        state = {
            "outer_iteration": 0,
            "no_improve_count": 2,
            "token_count": 0,
            "current_best_score": 1.0,
            "best_miss_rate": 0.0,
            "best_overkill_rate": 0.05,
            "best_accuracy": 0.97,
            "best_f1": 0.98,
        }

        self.assertTrue(should_exit_outer(state))

    def test_best_miss_rate_round_trips_through_resume_state(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase1_init.merger_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            l0_path = checkpoint_dir / "L0.json"
            phase2_path = checkpoint_dir / "phase2_init.json"
            best_path = checkpoint_dir / "best_pipeline.json"
            l0_path.write_text(json.dumps({
                "L0_script": "print('base')",
                "L0_score": 0.97,
                "best_candidate_name": "candidate",
            }))
            phase2_path.write_text(json.dumps({"current_best_score": 0.97}))
            best_path.write_text(json.dumps({
                "current_best_score": 0.97,
                "best_miss_rate": 0.025,
                "best_overkill_rate": 0.07,
                "best_accuracy": 0.94,
                "best_f1": 0.9,
                "best_pipeline_script": "print('best')",
            }))

            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir), \
                 mock.patch.object(config, "CKPT_L0", l0_path), \
                 mock.patch.object(config, "CKPT_PHASE2_INIT", phase2_path), \
                 mock.patch.object(config, "CKPT_BEST_PIPELINE", best_path), \
                 mock.patch.object(config, "CKPT_CANDIDATE_SCORES", checkpoint_dir / "candidate_scores.json"):
                context = FakeContext()
                module.check_and_load_phase2_init_fn(context)

        self.assertEqual(context.state["best_miss_rate"], 0.025)

    def test_phase2_initialization_seeds_ensemble_baseline_from_l0(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase1_init.merger_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "candidate_scores": [{
                        "name": "candidate_a",
                        "status": "success",
                        "metrics": {
                            "accuracy": 0.6229508196721312,
                            "ng_recall": 0.9354838709677419,
                            "miss_rate": 0.06451612903225812,
                            "overkill_rate": 0.7,
                            "f1": 0.7160493827160493,
                        },
                    }],
                    "candidate_scripts": [{
                        "name": "candidate_a",
                        "script": "print('baseline')",
                    }],
                    "token_count": 123,
                })

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir), \
                 mock.patch.object(config, "CKPT_L0", checkpoint_dir / "L0.json"), \
                 mock.patch.object(config, "CKPT_PHASE2_INIT", checkpoint_dir / "phase2_init.json"):
                context = FakeContext()
                module.initialize_phase2_fn(context)
                saved = json.loads((checkpoint_dir / "phase2_init.json").read_text())

        self.assertEqual(context.state["ensemble_best_score"], context.state["current_best_score"])
        self.assertEqual(context.state["ensemble_best_overkill"], context.state["best_overkill_rate"])
        self.assertEqual(saved["ensemble_best_score"], saved["current_best_score"])
        self.assertEqual(saved["ensemble_best_overkill"], saved["best_overkill_rate"])


class EnsembleDegenerateModelPolicyTests(unittest.TestCase):
    def test_all_ng_ensemble_is_not_an_improvement_even_from_uninitialized_baseline(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase3_ensemble.ensemble_evaluator_agent")

        degenerate_all_ng = {
            "accuracy": 0.4,
            "ng_recall": 1.0,
            "miss_rate": 0.0,
            "overkill_rate": 1.0,
            "f1": 0.5714,
        }
        uninitialized = {
            "accuracy": 0.0,
            "ng_recall": 0.0,
            "miss_rate": 1.0,
            "overkill_rate": 1.0,
            "f1": 0.0,
        }

        self.assertFalse(
            module._is_ensemble_improvement(
                1.0,
                1.0,
                0.0,
                1.0,
                new_metrics=degenerate_all_ng,
                current_metrics=uninitialized,
            )
        )

    def test_ensemble_overkill_regression_is_not_an_improvement(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase3_ensemble.ensemble_evaluator_agent")

        current = {
            "accuracy": 0.70,
            "ng_recall": 0.94,
            "miss_rate": 0.06,
            "overkill_rate": 0.30,
            "f1": 0.76,
        }
        higher_recall_worse_overkill = {
            "accuracy": 0.71,
            "ng_recall": 0.98,
            "miss_rate": 0.02,
            "overkill_rate": 0.45,
            "f1": 0.78,
        }

        self.assertTrue(module._has_overkill_regression(higher_recall_worse_overkill, current))
        self.assertFalse(
            module._is_ensemble_improvement(
                0.98,
                0.45,
                0.94,
                0.30,
                new_metrics=higher_recall_worse_overkill,
                current_metrics=current,
            )
        )


class EnsembleTriedApproachesPolicyTests(unittest.TestCase):
    def test_repeated_failed_ensemble_strategy_is_rejected_from_checkpoint_history(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase3_ensemble.ensemble_coder_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({"ensemble_iteration": 1})

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            history_path = checkpoint_dir / "tried_ensemble_approaches.json"
            history_path.write_text(json.dumps({
                "tried_ensemble_approaches": [{
                    "strategy_name": "threshold-specialists",
                    "combination_method": "max probability",
                    "result": {"improved": False},
                }]
            }))

            with mock.patch.object(config, "CKPT_TRIED_ENSEMBLE_APPROACHES", history_path):
                context = FakeContext()
                result = module.save_ensemble_strategy_fn(
                    context,
                    strategy_name="threshold-specialists",
                    combination_method="max probability",
                    component_descriptions="1. high recall model\n2. low overkill model",
                )

        self.assertIn("ENSEMBLE_STRATEGY_REJECTED", result)
        self.assertNotIn("ensemble_strategy", context.state)

    def test_ensemble_evaluator_persists_tried_ensemble_approaches(self):
        import importlib

        from mle_star_agent import config
        from mle_star_agent.shared.code_runner import RunResult

        module = importlib.import_module("mle_star_agent.phases.phase3_ensemble.ensemble_evaluator_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "ensemble_script": "print('stub')",
                    "ensemble_iteration": 0,
                    "ensemble_best_score": 0.9355,
                    "ensemble_best_overkill": 0.70,
                    "ensemble_best_accuracy": 0.6229,
                    "ensemble_best_f1": 0.716,
                    "ensemble_strategy": {
                        "strategy_name": "weighted-average",
                        "combination_method": "mean probability",
                    },
                })
                self.actions = type("Actions", (), {"escalate": False})()

        stdout = (
            'METRICS: {"accuracy": 0.4, "ng_recall": 1.0, "miss_rate": 0.0, '
            '"overkill_rate": 1.0, "f1": 0.5714, "threshold": 0.01, '
            '"ng_count": 4, "g_count": 6, "tp": 4, "tn": 0, "fp": 6, "fn": 0}'
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_ENSEMBLE", checkpoint_dir / "ensemble.json"),
                mock.patch.object(config, "CKPT_TRIED_ENSEMBLE_APPROACHES", checkpoint_dir / "tried_ensemble_approaches.json"),
                mock.patch.object(module.code_runner, "run_script", return_value=RunResult(0, stdout, "", 12.3, False)),
            ):
                context = FakeContext()
                module.evaluate_ensemble_fn(context)
                saved = json.loads((checkpoint_dir / "tried_ensemble_approaches.json").read_text())

        self.assertEqual(saved["tried_ensemble_approaches"][0]["strategy_name"], "weighted-average")
        self.assertFalse(saved["tried_ensemble_approaches"][0]["result"]["improved"])


class ValidatorContractPolicyTests(unittest.TestCase):
    def test_static_contract_check_requires_calibration_stats_marker(self):
        from mle_star_agent.guards.code_validator_agent import missing_required_full_run_markers

        script = 'print("METRICS: {}")\nprint("PREDICTIONS: []")\n'

        self.assertIn("CALIBRATION_STATS:", missing_required_full_run_markers(script))

    def test_refinement_coder_has_no_lite_mode_downgrade_callback(self):
        from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import refinement_coder_agent

        self.assertIsNone(refinement_coder_agent.before_model_callback)

    def test_coder_instructions_keep_fp_budget_but_template_uses_recall_calibration(self):
        # Phase 2 refinement still free-writes scripts, so its INSTRUCTION must
        # carry the FP-budget contract. Phase 1 candidates are now rendered from
        # the canonical template, whose threshold policy is recall-targeted
        # validation calibration.
        from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import _INSTRUCTION as refinement_instruction
        from mle_star_agent.shared.script_template import get_script_template

        self.assertIn("FP <= 2", refinement_instruction)
        self.assertIn("filter out thresholds", refinement_instruction)

        template = get_script_template(data_split_path="checkpoints/data_split_grouped.json")
        self.assertIn("VAL_NG_RECALL_TARGET", template)
        self.assertIn("recall_candidates = [c for c in all_candidates if c['recall'] >= VAL_NG_RECALL_TARGET]", template)
        self.assertIn("best_threshold = min(recall_candidates, key=lambda c: c['threshold'])['threshold']", template)

    def test_refinement_instruction_requires_dynamic_fp_penalty_when_overkill_is_high(self):
        from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import _INSTRUCTION

        self.assertIn('state["best_overkill_rate"] > 0.08', _INSTRUCTION)
        self.assertIn("fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08)", _INSTRUCTION)
        self.assertIn("must be executable code, not only a comment", _INSTRUCTION)

    def test_static_fp_penalty_check_detects_missing_dynamic_penalty(self):
        from mle_star_agent.guards.code_validator_agent import static_fp_penalty_check_fn

        missing = "criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)"
        present = """
best_overkill_rate = 0.70
fp_weight = 1.0 + 5.0 * max(0, best_overkill_rate - 0.08)
loss = fp_weight * false_positive_loss + base_loss
"""

        self.assertFalse(static_fp_penalty_check_fn(missing)["ok"])
        self.assertTrue(static_fp_penalty_check_fn(present)["ok"])

    def test_static_degenerate_threshold_guard_detects_missing_warning(self):
        from mle_star_agent.guards.code_validator_agent import static_degenerate_threshold_guard_check_fn

        missing = "if threshold <= 0.15: print('low threshold')"
        present = """
if threshold <= 0.15 and overkill_rate > 0.50:
    print("DEGENERATE_THRESHOLD_WARNING: low threshold with high overkill")
"""

        self.assertFalse(static_degenerate_threshold_guard_check_fn(missing)["ok"])
        self.assertTrue(static_degenerate_threshold_guard_check_fn(present)["ok"])

    def test_ensemble_instruction_prioritizes_second_stage_verifier_cascade(self):
        from mle_star_agent.phases.phase3_ensemble.ensemble_coder_agent import _INSTRUCTION

        self.assertIn("Iteration 1 — Second-stage verifier cascade", _INSTRUCTION)
        self.assertIn("only review samples predicted NG", _INSTRUCTION)
        self.assertIn("rescue obvious G false positives", _INSTRUCTION)


class SmallDataStrategyPolicyTests(unittest.TestCase):
    def test_coder_and_planner_prompts_prefer_small_data_safe_strategy_order(self):
        from mle_star_agent.phases.phase1_init.baseline_coder_agent import _INSTRUCTION as baseline_instruction
        from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import _INSTRUCTION as refinement_instruction
        from mle_star_agent.phases.phase2_refinement.planner_agent import _INSTRUCTION as planner_instruction

        for instruction in (baseline_instruction, refinement_instruction, planner_instruction):
            self.assertIn("small-data-safe", instruction)
            self.assertIn("freeze or partially-freeze", instruction)
            self.assertIn("weight decay + dropout", instruction)
            self.assertIn("AOI-safe augmentation", instruction)
            self.assertIn("global feature-difference-only", instruction)
            self.assertIn("larger backbone", instruction)
            self.assertIn("two-independent-backbone stereo", instruction)
            self.assertIn("Calibration/threshold curves are REPORTING", instruction)

    def test_planner_rejects_known_failed_mg7_mg8_fingerprint(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.planner_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 1,
                    "latest_probe_metrics": {"probability_gap": 0.1},
                })

        context = FakeContext()
        result = module.save_strategy_candidates_fn(
            context,
            strategy_a="global_feature_difference: add concat([f_L, f_R, |f_L-f_R|]) only",
            strategy_b="freeze_regularized_head: freeze backbone with dropout",
            strategy_c="aoi_safe_augmentation: paired light affine and crop",
            selected="a",
            selection_reason="Try feature difference.",
            strategy_a_target_component="stereo_fusion",
            strategy_a_mechanism_class="global_feature_difference_only",
            strategy_b_target_component="model_architecture",
            strategy_b_mechanism_class="freeze_regularized_head",
            strategy_c_target_component="augmentation",
            strategy_c_mechanism_class="aoi_safe_paired_augmentation",
        )

        self.assertIn("KNOWN_FAILED_STRATEGY_REJECTED", result)
        self.assertNotIn("selected_refinement_strategy", context.state)

    def test_small_data_static_validator_flags_unsafe_patterns(self):
        from mle_star_agent.shared.small_data_strategy_validator import (
            validate_small_data_strategy_source,
        )

        unsafe = """
import json
data_split = json.load(open("checkpoints/data_split.json"))
model = models.resnet50(weights="DEFAULT")
self.left = models.resnet18(weights=None)
self.right = models.resnet18(weights=None)
self.fc = nn.Linear(1536, 512)
transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5), transforms.RandomErasing()])
print("METRICS: {}")
"""

        report = validate_small_data_strategy_source(unsafe)

        self.assertFalse(report["ok"])
        self.assertTrue(report["uses_legacy_split"])
        self.assertTrue(report["large_capacity_without_regularization"])
        self.assertTrue(report["unsafe_augmentation"])
        self.assertTrue(report["missing_metric_reporting"])
        self.assertTrue(report["known_failed_fingerprint"])

    def test_small_data_static_validator_accepts_freeze_regularized_safe_aug(self):
        from mle_star_agent.shared.small_data_strategy_validator import (
            validate_small_data_strategy_source,
        )

        safe = """
import json
data_split = json.load(open("checkpoints/data_split_grouped.json"))
for p in self.backbone.parameters():
    p.requires_grad = False
self.head = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))
optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-4, weight_decay=1e-4)
params = transforms.RandomAffine.get_params((-3, 3), (0.02, 0.02), (0.98, 1.02), (0, 0), img_size)
img_l = TF.affine(img_l, *params)
img_r = TF.affine(img_r, *params)
roc_auc = roc_auc_score(y_true, ng_probs)
prob_gap = ng_mean - g_mean
print(f"THRESHOLD_CURVE: {threshold_curve}")
print(f"DEGENERATE_PREDICTION_WARNING: score_range={score_range}")
print(f"METRICS: {{'roc_auc': {roc_auc}, 'prob_gap': {prob_gap}}}")
"""

        report = validate_small_data_strategy_source(safe)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["has_freeze"])
        self.assertTrue(report["has_weight_decay"])
        self.assertTrue(report["has_dropout"])
        self.assertTrue(report["has_aoi_safe_paired_augmentation"])

    def test_small_data_static_validator_identifies_partial_unfreeze_last_block(self):
        from mle_star_agent.shared.small_data_strategy_validator import (
            validate_small_data_strategy_source,
        )

        partial = """
import json
data_split = json.load(open("checkpoints/data_split_grouped.json"))
for p in self.resnet.parameters():
    p.requires_grad = False
for name, p in self.resnet.named_parameters():
    if name.startswith("layer4"):
        p.requires_grad = True
self.head = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
roc_auc = roc_auc_score(y_true, ng_probs)
prob_gap = ng_mean - g_mean
print(f"THRESHOLD_CURVE: {threshold_curve}")
print(f"DEGENERATE_PREDICTION_WARNING: score_range={score_range}")
print(f"METRICS: {{'roc_auc': {roc_auc}, 'prob_gap': {prob_gap}}}")
"""

        report = validate_small_data_strategy_source(partial)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["has_freeze"])
        self.assertTrue(report["has_partial_unfreeze"])
        self.assertFalse(report["full_freeze_without_partial_unfreeze"])

    def test_small_data_static_validator_flags_full_freeze_only_for_flat_gap_followup(self):
        from mle_star_agent.shared.small_data_strategy_validator import (
            validate_small_data_strategy_source,
        )

        full_freeze = """
import json
data_split = json.load(open("checkpoints/data_split_grouped.json"))
for p in self.resnet.parameters():
    p.requires_grad = False
self.head = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))
optimizer = torch.optim.AdamW(self.head.parameters(), lr=1e-4, weight_decay=1e-4)
roc_auc = roc_auc_score(y_true, ng_probs)
prob_gap = ng_mean - g_mean
print(f"THRESHOLD_CURVE: {threshold_curve}")
print(f"DEGENERATE_PREDICTION_WARNING: score_range={score_range}")
print(f"METRICS: {{'roc_auc': {roc_auc}, 'prob_gap': {prob_gap}}}")
"""

        report = validate_small_data_strategy_source(full_freeze)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["full_freeze_without_partial_unfreeze"])
        self.assertIn("full_freeze_without_partial_unfreeze", report["warnings"])

    def test_diagnosis_brief_carries_small_data_policy(self):
        from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief

        brief = generate_diagnosis_brief(
            [],
            {
                "accuracy": 0.61,
                "ng_recall": 0.86,
                "miss_rate": 0.14,
                "overkill_rate": 0.62,
                "f1": 0.70,
                "roc_auc": 0.80,
                "prob_gap": 0.37,
            },
            calibration_stats={"G_prob_mean": 0.23, "NG_prob_mean": 0.60},
        )

        policy = brief["small_data_strategy_policy"]
        self.assertIn("freeze_or_partial_freeze_small_head", policy["prefer_order"])
        self.assertIn(["stereo_fusion", "global_feature_difference_only"], policy["known_failed_fingerprints"])
        self.assertEqual(policy["primary_signals"], ["val_auc", "prob_gap"])

    def test_diagnosis_flags_full_freeze_underfit_when_prob_gap_collapses(self):
        from mle_star_agent.shared.diagnosis_scorer import generate_diagnosis_brief

        brief = generate_diagnosis_brief(
            [],
            {
                "accuracy": 0.52,
                "ng_recall": 0.95,
                "miss_rate": 0.05,
                "overkill_rate": 0.88,
                "f1": 0.65,
                "roc_auc": 0.59,
                # prob_gap must be at/below the separability floor
                # (config.PROBE_PROBABILITY_GAP_MIN = 0.01) to count as a true
                # collapse; values above it are intentionally left to train (see
                # the "let borderline cases train" config note), so 0.02 correctly
                # classifies as low_capacity_miss, not full_freeze_underfit.
                "prob_gap": 0.005,
            },
            calibration_stats={"G_prob_mean": 0.48, "NG_prob_mean": 0.50},
        )

        failure = brief["failure_classification"]
        self.assertEqual(failure["failure_mode"], "full_freeze_underfit")
        self.assertEqual(failure["recommended_target"], "model_architecture")
        self.assertIn("partial-unfreeze", failure["recommended_action"])
        self.assertIn("layer4", failure["recommended_action"])


class SequentialExecutionPolicyTests(unittest.TestCase):
    def test_candidate_evaluator_runs_slots_sequentially(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase1_init.candidate_evaluator_agent")

        agent = module.candidate_evaluator_agent.sub_agents[1]

        self.assertEqual(agent.__class__.__name__, "SequentialAgent")
        self.assertEqual(agent.name, "candidate_sequential_evaluator")

    def test_ablation_runs_variants_sequentially(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.ablation_agent")

        agent = module.ablation_agent.sub_agents[1]

        self.assertEqual(agent.__class__.__name__, "SequentialAgent")
        self.assertEqual(agent.name, "ablation_sequential")


class CheckpointLineagePolicyTests(unittest.TestCase):
    def test_ablation_checkpoint_rejected_when_best_script_changes(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.ablation_agent")

        current_state = {"best_pipeline_script": "print('new pipeline')"}
        stale_data = {
            "lineage": module._ablation_lineage({"best_pipeline_script": "print('old pipeline')"}),
            "ablation_results": [
                {"variant_index": i, "status": "success"}
                for i in range(module.NUM_ABLATION_VARIANTS)
            ],
        }

        self.assertFalse(module._is_ablation_checkpoint_current(stale_data, current_state))

    def test_diagnosis_checkpoint_rejected_when_ablation_evidence_changes(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.diagnosis_agent")

        current_state = {
            "ablation_results": [{"variant_index": 0, "name": "current", "metrics": {"overkill_rate": 0.5}}],
        }
        stale_data = {
            "lineage": module._diagnosis_lineage({
                "ablation_results": [{"variant_index": 0, "name": "old", "metrics": {"overkill_rate": 0.9}}],
            }),
            "diagnosis_report": {"ablation_ranking": [{"name": "old"}]},
        }

        self.assertFalse(module._is_diagnosis_checkpoint_current(stale_data, current_state))

    def test_submission_checkpoint_rejected_when_selected_script_changes(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase4_submission.submission_agent")

        stale_data = {
            "lineage": module._submission_lineage("print('old submission')"),
            "metrics": {"ng_recall": 1.0},
            "pass_fail": {"relaxed_minimum_pass": True},
        }

        self.assertFalse(
            module._is_submission_checkpoint_current(stale_data, "print('new submission')")
        )


class ErrorAnalysisParsingTests(unittest.TestCase):
    def test_predictions_line_is_parsed_into_fp_fn_samples(self):
        from mle_star_agent.shared.metrics_parser import parse_error_analysis, parse_metrics

        stdout = """
METRICS: {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "threshold": 0.4}
PREDICTIONS: [
  {"sample_id": "ng-hit", "true_label": "NG", "predicted_label": "NG", "ng_probability": 0.8, "threshold": 0.4},
  {"sample_id": "good-hit", "true_label": "G", "predicted_label": "G", "ng_probability": 0.2, "threshold": 0.4},
  {"sample_id": "good-overkill", "true_label": "G", "predicted_label": "NG", "ng_probability": 0.7, "threshold": 0.4},
  {"sample_id": "missed-ng", "true_label": "NG", "predicted_label": "G", "ng_probability": 0.3, "threshold": 0.4}
]
"""
        analysis = parse_error_analysis(stdout, metrics=parse_metrics(stdout))

        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["source"], "PREDICTIONS")
        self.assertEqual(analysis["fp_count"], 1)
        self.assertEqual(analysis["fn_count"], 1)
        self.assertEqual(analysis["fp_samples"][0]["sample_id"], "good-overkill")
        self.assertEqual(analysis["fn_samples"][0]["sample_id"], "missed-ng")
        self.assertTrue(analysis["metrics_consistency"]["matches_metrics"])

    def test_predictions_are_summarized_by_lot_for_separability_report(self):
        from mle_star_agent.shared.metrics_parser import parse_error_analysis, parse_metrics

        stdout = """
METRICS: {"tp": 1, "tn": 1, "fp": 2, "fn": 1, "threshold": 0.4}
PREDICTIONS: [
  {"sample_id": "lot-a/ng-hit", "lot": "lot-a", "true_label": "NG", "predicted_label": "NG", "ng_probability": 0.8, "threshold": 0.4},
  {"sample_id": "lot-a/good-overkill", "lot": "lot-a", "true_label": "G", "predicted_label": "NG", "ng_probability": 0.7, "threshold": 0.4},
  {"sample_id": "lot-b/good-overkill", "lot": "lot-b", "true_label": "G", "predicted_label": "NG", "ng_probability": 0.6, "threshold": 0.4},
  {"sample_id": "lot-b/missed-ng", "lot": "lot-b", "true_label": "NG", "predicted_label": "G", "ng_probability": 0.3, "threshold": 0.4},
  {"sample_id": "lot-b/good-hit", "lot": "lot-b", "true_label": "G", "predicted_label": "G", "ng_probability": 0.2, "threshold": 0.4}
]
"""
        analysis = parse_error_analysis(stdout, metrics=parse_metrics(stdout))

        self.assertEqual(analysis["per_lot"]["lot-a"]["fp"], 1)
        self.assertEqual(analysis["per_lot"]["lot-a"]["overkill_rate"], 1.0)
        self.assertEqual(analysis["per_lot"]["lot-b"]["fp"], 1)
        self.assertEqual(analysis["per_lot"]["lot-b"]["fn"], 1)
        self.assertEqual(analysis["per_lot"]["lot-b"]["overkill_rate"], 0.5)

    def test_missing_error_analysis_returns_checkpointable_placeholder(self):
        from mle_star_agent.shared.metrics_parser import parse_error_analysis, parse_metrics

        stdout = 'METRICS: {"tp": 29, "tn": 9, "fp": 21, "fn": 2, "threshold": 0.25}'
        analysis = parse_error_analysis(stdout, metrics=parse_metrics(stdout))

        self.assertFalse(analysis["available"])
        self.assertEqual(analysis["missing_reason"], "No ERROR_ANALYSIS or PREDICTIONS block found in stdout.")
        self.assertEqual(analysis["metrics_consistency"]["expected_fp"], 21)
        self.assertEqual(analysis["metrics_consistency"]["expected_fn"], 2)

    def test_probe_metrics_parser_normalizes_separability_fields(self):
        from mle_star_agent.shared.metrics_parser import parse_probe_metrics

        stdout = (
            'PROBE_METRICS: {"G_prob_mean": 0.49, "NG_prob_mean": 0.51, '
            '"ng_recall": 0.95, "overkill_rate": 0.62}'
        )
        probe = parse_probe_metrics(stdout)

        self.assertEqual(probe["source"], "PROBE_METRICS")
        self.assertEqual(probe["g_prob_mean"], 0.49)
        self.assertEqual(probe["ng_prob_mean"], 0.51)
        self.assertAlmostEqual(probe["probability_gap"], 0.02)
        self.assertEqual(probe["ng_recall"], 0.95)
        self.assertEqual(probe["overkill_rate"], 0.62)


class ErrorAnalysisEvaluatorCheckpointTests(unittest.TestCase):
    def test_evaluator_saves_error_analysis_checkpoint_from_predictions(self):
        import importlib

        from mle_star_agent import config
        from mle_star_agent.shared.code_runner import RunResult

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        class FakeState(dict):
            @property
            def _value(self):
                return dict(self)

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "current_script": "print('already mocked')",
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "current_best_score": 0.0,
                    "best_overkill_rate": 1.0,
                    "best_accuracy": 0.0,
                    "best_f1": 0.0,
                    "no_improve_count": 0,
                    "token_count": 0,
                })
                self.actions = type("Actions", (), {"escalate": False})()

        stdout = """
METRICS: {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "threshold": 0.4}
PREDICTIONS: [
  {"sample_id": "good-overkill", "true_label": "G", "predicted_label": "NG", "ng_probability": 0.7, "threshold": 0.4},
  {"sample_id": "missed-ng", "true_label": "NG", "predicted_label": "G", "ng_probability": 0.3, "threshold": 0.4}
]
"""

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
                mock.patch.object(
                    module.code_runner,
                    "run_script",
                    return_value=RunResult(0, stdout, "", 12.3, False),
                ),
            ):
                context = FakeContext()
                module.evaluate_and_update_fn(context)

                error_path = checkpoint_dir / "error_analysis_0_0.json"
                self.assertTrue(error_path.exists())
                saved = json.loads(error_path.read_text())
                self.assertTrue(saved["available"])
                self.assertEqual(saved["fp_samples"][0]["sample_id"], "good-overkill")
                self.assertEqual(saved["fn_samples"][0]["sample_id"], "missed-ng")
                self.assertEqual(context.state["latest_error_analysis"]["source"], "PREDICTIONS")
                self.assertEqual(context.state["latest_error_analysis_path"], str(error_path))

    def test_evaluator_rejects_bad_probe_before_full_metrics(self):
        import importlib

        from mle_star_agent import config
        from mle_star_agent.shared.code_runner import RunResult

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        class FakeState(dict):
            @property
            def _value(self):
                return dict(self)

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "current_script": "print('already mocked')",
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "current_best_score": 0.0,
                    "best_overkill_rate": 1.0,
                    "best_accuracy": 0.0,
                    "best_f1": 0.0,
                    "no_improve_count": 0,
                    "token_count": 0,
                })
                self.actions = type("Actions", (), {"escalate": False})()

        # Catastrophic probe: overkill 0.95 exceeds PROBE_OVERKILL_REJECT_MAX (0.90,
        # loosened to "reject only ConvNeXt-style >=90% overkill"). A 0.70 probe is
        # now intentionally allowed to train, so it must be >0.90 to be rejected.
        stdout = (
            'PROBE_METRICS: {"ng_recall": 0.96, "overkill_rate": 0.95, '
            '"G_prob_mean": 0.48, "NG_prob_mean": 0.50}\n'
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
                mock.patch.object(
                    module.code_runner,
                    "run_script",
                    return_value=RunResult(0, stdout, "", 12.3, False),
                ),
            ):
                context = FakeContext()
                message = module.evaluate_and_update_fn(context)

                self.assertIn("PROBE_REJECTED", message)
                # A rejected probe is a generation failure (script never trained), so it
                # bumps generation_fail_count; no_improve only ticks after GENERATION_FAIL_MAX.
                self.assertEqual(context.state["generation_fail_count"], 1)
                self.assertEqual(context.state["no_improve_count"], 0)
                self.assertEqual(context.state["latest_probe_metrics"]["overkill_rate"], 0.95)
                saved = json.loads((checkpoint_dir / "refinement_0_0.json").read_text())
                self.assertEqual(saved["failure_reason"], "probe_rejected")
                self.assertEqual(saved["probe_metrics"]["overkill_rate"], 0.95)


class ErrorAnalysisAgentWorkflowTests(unittest.TestCase):
    def test_error_analysis_report_writer_persists_report_from_latest_evidence(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "inner_iteration": 1,
                    "latest_error_analysis_path": "/tmp/error_analysis_0_0.json",
                    "latest_error_analysis": {
                        "available": True,
                        "fp_count": 21,
                        "fn_count": 2,
                        "metrics_consistency": {"matches_metrics": True},
                    },
                })

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir):
                context = FakeContext()
                result = module.write_error_analysis_report_fn(
                    context,
                    dominant_failure="overkill",
                    threshold_fix_possible=False,
                    evidence_summary="FP count is far above the relaxed target.",
                    recommended_target_component="g_false_positive_control",
                    recommended_changes='["Add calibration", "Analyze G probability overlap"]',
                )

                report_path = checkpoint_dir / "error_analysis_report_0_0.json"
                self.assertIn("error_analysis_report_0_0.json", result)
                self.assertTrue(report_path.exists())
                saved = json.loads(report_path.read_text())
                self.assertEqual(saved["error_analysis_report"]["dominant_failure"], "overkill")
                self.assertEqual(
                    context.state["error_analysis_report"]["recommended_target_component"],
                    "g_false_positive_control",
                )

    def test_inner_loop_runs_error_analysis_after_evaluator(self):
        from mle_star_agent.phases.phase2_refinement import inner_loop_agent
        from mle_star_agent.phases.phase2_refinement.error_analysis_agent import (
            error_analysis_agent,
            error_analysis_gate_agent,
        )
        from mle_star_agent.phases.phase2_refinement.evaluator_agent import evaluator_agent
        from mle_star_agent.phases.phase2_refinement.planner_agent import (
            refinement_planner_agent,
            strategy_gate_agent,
        )
        from mle_star_agent.phases.phase2_refinement.reflexion_agent import reflexion_agent
        from mle_star_agent.phases.phase2_refinement.refinement_coder_agent import refinement_coder_agent

        names = [agent.name for agent in inner_loop_agent.sub_agents]

        # reflexion_agent (commit 5967065) runs after the gate and before planning,
        # so it sits at index 1 in the inner-loop chain.
        self.assertEqual(names[:5], [
            error_analysis_gate_agent.name,
            reflexion_agent.name,
            refinement_planner_agent.name,
            strategy_gate_agent.name,
            refinement_coder_agent.name,
        ])
        self.assertEqual(names[-2:], [evaluator_agent.name, error_analysis_agent.name])

    def test_pre_refinement_gate_allows_missing_report_with_instrumentation_flag(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({"inner_iteration": 1})
                self.actions = type("Actions", (), {"escalate": False})()

        from mle_star_agent import config

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "CHECKPOINT_DIR", Path(tmp)):
                context = FakeContext()
                result = module.check_error_analysis_gate_fn(context)

                self.assertIn("ALLOW_NO_EVIDENCE", result)
                self.assertFalse(context.actions.escalate)
                self.assertTrue(context.state["error_analysis_instrumentation_required"])
                self.assertTrue(context.state["error_analysis_repair_attempted"])

    def test_pre_refinement_gate_blocks_second_missing_report(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "inner_iteration": 2,
                    "error_analysis_repair_attempted": True,
                    "error_analysis_instrumentation_required": True,
                })
                self.actions = type("Actions", (), {"escalate": False})()

        from mle_star_agent import config

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "CHECKPOINT_DIR", Path(tmp)):
                context = FakeContext()
                result = module.check_error_analysis_gate_fn(context)

                self.assertIn("BLOCK_NO_EVIDENCE", result)
                self.assertTrue(context.actions.escalate)
                self.assertTrue(context.state["error_analysis_blocked"])
                self.assertFalse(context.state.get("stop_outer_loop", False))

    def test_pre_refinement_gate_allows_unavailable_evidence_with_instrumentation_flag(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "inner_iteration": 1,
                    "error_analysis_report": {
                        "evidence_available": False,
                        "metrics_consistency": {"matches_metrics": True},
                    },
                })
                self.actions = type("Actions", (), {"escalate": False})()

        context = FakeContext()
        result = module.check_error_analysis_gate_fn(context)

        self.assertIn("ALLOW_NO_EVIDENCE", result)
        self.assertFalse(context.actions.escalate)
        self.assertFalse(context.state.get("stop_outer_loop", False))
        self.assertTrue(context.state["error_analysis_instrumentation_required"])

    def test_pre_refinement_gate_allows_metrics_consistency_warning(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "inner_iteration": 1,
                    "error_analysis_report": {
                        "evidence_available": True,
                        "metrics_consistency": {"matches_metrics": False},
                        "fp_count": 3,
                        "fn_count": 1,
                    },
                })
                self.actions = type("Actions", (), {"escalate": False})()

        context = FakeContext()
        result = module.check_error_analysis_gate_fn(context)

        self.assertIn("ALLOW_CONSISTENCY_WARNING", result)
        self.assertFalse(context.actions.escalate)
        self.assertFalse(context.state.get("stop_outer_loop", False))
        self.assertNotIn("error_analysis_instrumentation_required", context.state)

    def test_pre_refinement_gate_allows_valid_report_after_first_iteration(self):
        import importlib

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.error_analysis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "inner_iteration": 1,
                    "error_analysis_report": {
                        "evidence_available": True,
                        "metrics_consistency": {"matches_metrics": True},
                    },
                })
                self.actions = type("Actions", (), {"escalate": False})()

        context = FakeContext()
        result = module.check_error_analysis_gate_fn(context)

        self.assertIn("ALLOW", result)
        self.assertFalse(context.actions.escalate)

    def test_diagnosis_report_can_include_error_analysis_evidence(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.diagnosis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "ablation_results": [],
                    "error_analysis_report": {
                        "dominant_failure": "overkill",
                        "recommended_target_component": "g_false_positive_control",
                    },
                })

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir):
                context = FakeContext()
                module.write_diagnosis_report_fn(
                    context,
                    target_component="g_false_positive_control",
                    impact_summary="Error analysis shows overkill dominates.",
                    recommended_changes="Reduce FP while preserving FN=0.",
                    ablation_ranking="[]",
                )

                saved = json.loads((checkpoint_dir / "diagnosis_0.json").read_text())
                report = saved["diagnosis_report"]
                self.assertEqual(report["error_analysis_evidence"]["dominant_failure"], "overkill")

    def test_diagnosis_report_persists_structured_prediction_contract(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.diagnosis_agent")

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 0,
                    "ablation_results": [],
                })

        prediction = {
            "expected_overkill_rate_max": 0.25,
            "expected_ng_recall_min": 0.93,
            "expected_miss_rate_max": 0.07,
            "failure_if": "recall drops below 0.93 or overkill stays above 0.25",
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir):
                context = FakeContext()
                module.write_diagnosis_report_fn(
                    context,
                    target_component="weighted_loss",
                    impact_summary="Loss should reduce G false positives.",
                    recommended_changes="Add dynamic g_fp_weight.",
                    ablation_ranking="[]",
                    prediction_contract=json.dumps(prediction),
                )

                saved = json.loads((checkpoint_dir / "diagnosis_0.json").read_text())

        self.assertEqual(
            saved["diagnosis_report"]["prediction"]["expected_ng_recall_min"],
            0.93,
        )

    def test_evaluator_marks_failed_prediction_and_blocks_false_improvement(self):
        import importlib

        from mle_star_agent import config

        module = importlib.import_module("mle_star_agent.phases.phase2_refinement.evaluator_agent")

        class FakeState(dict):
            pass

        class FakeActions:
            escalate = False

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "current_script": "print('stub')",
                    "outer_iteration": 0,
                    "inner_iteration": 0,
                    "current_best_score": 0.9354838709677419,
                    "best_miss_rate": 0.06451612903225806,
                    "best_overkill_rate": 0.70,
                    "best_accuracy": 0.6229508196721312,
                    "best_f1": 0.716,
                    "no_improve_count": 0,
                    "diagnosis_report": {
                        "target_component": "weighted_loss",
                        "prediction": {
                            "expected_overkill_rate_max": 0.25,
                            "expected_ng_recall_min": 0.93,
                            "expected_miss_rate_max": 0.07,
                        },
                    },
                    "selected_refinement_strategy": "asymmetric_bce_loss with dynamic g_fp_weight",
                    "selected_strategy_fingerprint": {
                        "target_component": "weighted_loss",
                        "mechanism_class": "fp_penalty_loss",
                    },
                    "refinement_plan": {
                        "target_component": "weighted_loss",
                        "changes_summary": "dynamic g_fp_weight",
                    },
                })
                self.actions = FakeActions()

        stdout = (
            'METRICS: {"accuracy": 0.6229508196721312, "ng_recall": 0.2903225806451613, '
            '"miss_rate": 0.7096774193548387, "overkill_rate": 0.03333333333333333, '
            '"f1": 0.4390243902439024, "threshold": 0.72, "avg_latency_ms": 1, '
            '"ng_count": 31, "g_count": 30, "tp": 9, "tn": 29, "fp": 1, "fn": 22}'
        )
        result = mock.Mock(returncode=0, timed_out=False, duration_ms=12.0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            with (
                mock.patch.object(config, "CHECKPOINT_DIR", checkpoint_dir),
                mock.patch.object(config, "CKPT_TRIED_APPROACHES", checkpoint_dir / "tried_approaches.json"),
                mock.patch.object(config, "CKPT_BEST_PIPELINE", checkpoint_dir / "best_pipeline.json"),
                mock.patch.object(module.code_runner, "run_script", return_value=result),
            ):
                context = FakeContext()
                module.evaluate_and_update_fn(context)
                attempt = json.loads((checkpoint_dir / "refinement_0_0.json").read_text())
                history = json.loads((checkpoint_dir / "tried_approaches.json").read_text())

        self.assertFalse(attempt["improved"])
        self.assertEqual(attempt["failure_reason"], "diagnosis_prediction_failed")
        self.assertEqual(attempt["prediction_verification"]["status"], "failed")
        self.assertIn("ng_recall", attempt["prediction_verification"]["failed_constraints"])
        self.assertEqual(history["tried_approaches"][0]["failure_reason"], "diagnosis_prediction_failed")

    def test_code_validator_requires_per_sample_prediction_output(self):
        from mle_star_agent.guards.code_validator_agent import _INSTRUCTION

        self.assertIn("CHECK 2C", _INSTRUCTION)
        self.assertIn("PREDICTIONS:", _INSTRUCTION)
        self.assertIn("ERROR_ANALYSIS", _INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
