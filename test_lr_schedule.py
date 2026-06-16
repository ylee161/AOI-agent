import os
import importlib
import unittest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


class LrSchedulePromptTests(unittest.TestCase):
    def test_baseline_recipe_requires_lr_schedule_without_changing_epoch_contract(self):
        # Phase 1 candidates are rendered from the canonical template, which owns
        # the epoch contract and the scheduler stepping; the coder instruction
        # only has to demand a real schedule from build_scheduler().
        baseline_coder_agent = importlib.import_module(
            "mle_star_agent.phases.phase1_init.baseline_coder_agent"
        )
        from mle_star_agent.shared.script_template import get_script_template

        instruction = baseline_coder_agent._INSTRUCTION
        self.assertIn("REAL LR schedule", instruction)
        self.assertIn("CosineAnnealingWarmRestarts", instruction)

        template = get_script_template(data_split_path="checkpoints/data_split_grouped.json")
        self.assertIn("epochs          = DRY_RUN_EPOCHS if DRY_RUN else 20", template)
        self.assertIn("scheduler.step()", template)

    def test_refinement_recipe_requires_lr_schedule_when_refining_scripts(self):
        refinement_coder_agent = importlib.import_module(
            "mle_star_agent.phases.phase2_refinement.refinement_coder_agent"
        )

        instruction = refinement_coder_agent._INSTRUCTION

        self.assertIn("learning-rate schedule", instruction)
        self.assertIn("CosineAnnealingWarmRestarts", instruction)
        self.assertIn("ReduceLROnPlateau", instruction)
        self.assertIn("scheduler.step()", instruction)


class LrScheduleValidatorTests(unittest.TestCase):
    def test_validator_passes_scheduler_with_step_call(self):
        from mle_star_agent.shared.lr_schedule_validator import (
            validate_lr_schedule_source,
        )

        script = """
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
for epoch in range(epochs):
    train_one_epoch()
    validate()
    scheduler.step(epoch + 1)
"""

        report = validate_lr_schedule_source(script)

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["constructs_lr_scheduler"])
        self.assertTrue(report["calls_scheduler_step"])
        self.assertEqual(report["reasons"], [])

    def test_validator_passes_reduce_on_plateau_with_val_loss_step(self):
        from mle_star_agent.shared.lr_schedule_validator import (
            validate_lr_schedule_source,
        )

        script = """
from torch.optim.lr_scheduler import ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
for epoch in range(epochs):
    val_loss = validate()
    scheduler.step(val_loss)
"""

        report = validate_lr_schedule_source(script)

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["scheduler_classes"], ["ReduceLROnPlateau"])

    def test_validator_fails_fixed_lr_without_scheduler(self):
        from mle_star_agent.shared.lr_schedule_validator import (
            validate_lr_schedule_source,
        )

        script = """
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
for epoch in range(epochs):
    train_one_epoch()
    validate()
"""

        report = validate_lr_schedule_source(script)

        self.assertFalse(report["ok"])
        self.assertIn("missing_lr_scheduler", report["reasons"])
        self.assertIn("missing_scheduler_step", report["reasons"])
        self.assertIn("no learning-rate scheduler constructed", report["messages"])
        self.assertIn("scheduler.step() is not called", report["messages"])

    def test_code_validator_exposes_lr_schedule_hard_gate(self):
        code_validator_agent = importlib.import_module(
            "mle_star_agent.guards.code_validator_agent"
        )

        instruction = code_validator_agent._INSTRUCTION

        self.assertIn("CHECK 2B1 — Mandatory Learning-Rate Schedule", instruction)
        self.assertIn("static_lr_schedule_check_fn", instruction)
        self.assertIn("HARD gate", instruction)
        self.assertFalse(
            code_validator_agent.static_lr_schedule_check_fn(
                "optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)"
            )["ok"]
        )


if __name__ == "__main__":
    unittest.main()
