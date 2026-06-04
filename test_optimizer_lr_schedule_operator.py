import importlib
import unittest


class OptimizerLrScheduleOperatorTests(unittest.TestCase):
    def test_planner_accepts_optimizer_lr_schedule_as_selectable_target(self):
        planner_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.planner_agent"
        )

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 2,
                    "inner_iteration": 0,
                    "latest_probe_metrics": {
                        "recommended_target_component": "optimizer/lr-schedule",
                    },
                })

        context = FakeContext()
        result = planner_agent.save_strategy_candidates_fn(
            context,
            strategy_a=(
                "adamw_cosine_restart_tune: keep AdamW but tune lr, weight_decay, "
                "and CosineAnnealingWarmRestarts T_0"
            ),
            strategy_b=(
                "sgd_momentum_plateau: switch AdamW to SGD momentum and use "
                "ReduceLROnPlateau on validation loss"
            ),
            strategy_c=(
                "adamw_plateau_decay: keep AdamW but swap cosine restarts for "
                "ReduceLROnPlateau and stronger weight decay"
            ),
            selected="b",
            selection_reason="Probe recommends optimizer/lr-schedule after a plateau.",
            strategy_a_target_component="optimizer/lr-schedule",
            strategy_a_mechanism_class="adamw_cosine_restart_tune",
            strategy_b_target_component="optimizer/lr-schedule",
            strategy_b_mechanism_class="sgd_momentum_plateau",
            strategy_c_target_component="optimizer/lr-schedule",
            strategy_c_mechanism_class="adamw_plateau_decay",
        )

        self.assertIn("Strategy candidates saved", result)
        self.assertEqual(
            context.state["selected_strategy_fingerprint"],
            {
                "target_component": "optimizer/lr-schedule",
                "mechanism_class": "sgd_momentum_plateau",
            },
        )
        self.assertIn("SGD", context.state["selected_refinement_strategy"])

    def test_planner_rejects_duplicate_optimizer_lr_schedule_combo(self):
        planner_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.planner_agent"
        )

        class FakeState(dict):
            pass

        class FakeContext:
            def __init__(self):
                self.state = FakeState({
                    "outer_iteration": 2,
                    "inner_iteration": 1,
                    "tried_approaches": [{
                        "target_component": "optimizer/lr-schedule",
                        "strategy_fingerprint": {
                            "target_component": "optimizer/lr-schedule",
                            "mechanism_class": "sgd_momentum_plateau",
                        },
                        "result": {"improved": False},
                        "failure_reason": "no_improvement",
                    }],
                })

        context = FakeContext()
        result = planner_agent.save_strategy_candidates_fn(
            context,
            strategy_a="sgd_momentum_plateau: retry the same optimizer schedule",
            strategy_b="adamw_cosine_restart_tune: tune AdamW cosine schedule",
            strategy_c="adamw_plateau_decay: AdamW plus ReduceLROnPlateau",
            selected="a",
            selection_reason="This should be rejected as a duplicate combo.",
            strategy_a_target_component="optimizer/lr-schedule",
            strategy_a_mechanism_class="sgd_momentum_plateau",
            strategy_b_target_component="optimizer/lr-schedule",
            strategy_b_mechanism_class="adamw_cosine_restart_tune",
            strategy_c_target_component="optimizer/lr-schedule",
            strategy_c_mechanism_class="adamw_plateau_decay",
        )

        self.assertIn("DUPLICATE_STRATEGY_REJECTED", result)
        self.assertNotIn("selected_refinement_strategy", context.state)

    def test_tried_approach_label_identifies_optimizer_schedule_combo(self):
        evaluator_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.evaluator_agent"
        )

        label = evaluator_agent._attempt_label(
            "training_schedule",
            {
                "target_component": "optimizer/lr-schedule",
                "mechanism_class": "sgd_momentum_plateau",
            },
        )

        self.assertEqual(label, "optimizer/lr-schedule:sgd_momentum_plateau")

    def test_ablation_catalog_exposes_optimizer_lr_schedule_probe(self):
        ablation_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.ablation_agent"
        )

        variants = {
            variant["name"]: variant
            for variant in ablation_agent.ABLATION_VARIANTS
        }

        self.assertIn("optimizer_lr_schedule", variants)
        self.assertIn(
            "optimizer/lr-schedule",
            variants["optimizer_lr_schedule"]["target_component"],
        )
        self.assertIn("SGD", variants["optimizer_lr_schedule"]["description"])
        self.assertIn(
            "ReduceLROnPlateau",
            variants["optimizer_lr_schedule"]["description"],
        )

    def test_refinement_coder_requires_concrete_optimizer_schedule_variants(self):
        refinement_coder_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.refinement_coder_agent"
        )

        instruction = refinement_coder_agent._INSTRUCTION

        self.assertIn("optimizer/lr-schedule", instruction)
        self.assertIn("SGD", instruction)
        self.assertIn("AdamW", instruction)
        self.assertIn("CosineAnnealingWarmRestarts", instruction)
        self.assertIn("ReduceLROnPlateau", instruction)
        self.assertIn("epochs = DRY_RUN_EPOCHS if DRY_RUN else 20", instruction)
        self.assertIn("EPOCH_LOG", instruction)


if __name__ == "__main__":
    unittest.main()
