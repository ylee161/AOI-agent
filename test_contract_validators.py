import os
import importlib
import unittest
from unittest import mock

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


def _base_contract_script() -> str:
    return """
import json

DRY_RUN = True
DRY_RUN_EPOCHS = 1
epochs = DRY_RUN_EPOCHS if DRY_RUN else 20

scheduler = object()
scheduler.step()

print("PROBE_METRICS: " + json.dumps({}))
print("EPOCH_LOG: " + json.dumps({}))
print("METRICS: " + json.dumps({}))
print("CALIBRATION_STATS: " + json.dumps({}))
print("THRESHOLD_CURVE: " + json.dumps([]))
print("PREDICTIONS: " + json.dumps([]))
"""


def _contract_report(script: str) -> dict:
    from mle_star_agent.guards.code_validator_agent import static_contract_check_fn

    return static_contract_check_fn(script)


class GeneratedScriptContractValidatorTests(unittest.TestCase):
    def assert_contract_passes(self, script: str) -> None:
        report = _contract_report(script)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["rejection_reasons"], [])

    def assert_contract_rejects(self, script: str, expected_reason: str) -> None:
        report = _contract_report(script)
        self.assertFalse(report["ok"], report)
        self.assertIn(expected_reason, report["rejection_reasons"])

    def test_accepts_script_with_epoch_log_marker(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_epoch_log_marker(self):
        script = _base_contract_script().replace('print("EPOCH_LOG: " + json.dumps({}))\n', "")

        self.assert_contract_rejects(script, "missing required marker EPOCH_LOG:")

    def test_accepts_script_with_metrics_marker(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_metrics_marker(self):
        script = _base_contract_script().replace('print("METRICS: " + json.dumps({}))\n', "")

        self.assert_contract_rejects(script, "missing required marker METRICS:")

    def test_accepts_script_with_threshold_curve_marker(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_threshold_curve_marker(self):
        script = _base_contract_script().replace('print("THRESHOLD_CURVE: " + json.dumps([]))\n', "")

        self.assert_contract_rejects(script, "missing required marker THRESHOLD_CURVE:")

    def test_accepts_script_with_calibration_stats_marker(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_calibration_stats_marker(self):
        script = _base_contract_script().replace('print("CALIBRATION_STATS: " + json.dumps({}))\n', "")

        self.assert_contract_rejects(script, "missing required marker CALIBRATION_STATS:")

    def test_accepts_script_with_dry_run_epoch_ternary(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_dry_run_epoch_ternary(self):
        script = _base_contract_script().replace(
            "epochs = DRY_RUN_EPOCHS if DRY_RUN else 20",
            "epochs = 20",
        )

        self.assert_contract_rejects(
            script,
            "missing epoch ternary: epochs = DRY_RUN_EPOCHS if DRY_RUN else",
        )

    def test_accepts_script_with_scheduler_step_call(self):
        self.assert_contract_passes(_base_contract_script())

    def test_rejects_script_missing_scheduler_step_call(self):
        script = _base_contract_script().replace("scheduler.step()\n", "")

        self.assert_contract_rejects(script, "missing scheduler.step() call")

    def test_run_script_fn_rejects_invalid_contract_before_execution(self):
        code_validator_agent = importlib.import_module(
            "mle_star_agent.guards.code_validator_agent"
        )

        script = _base_contract_script().replace('print("METRICS: " + json.dumps({}))\n', "")

        with mock.patch.object(code_validator_agent.code_runner, "run_script") as run_script:
            result = code_validator_agent.run_script_fn(script)

        run_script.assert_not_called()
        self.assertNotEqual(result["returncode"], 0)
        self.assertIn("CONTRACT_VALIDATION_FAILED", result["stderr"])
        self.assertIn("missing required marker METRICS:", result["stderr"])


if __name__ == "__main__":
    unittest.main()
