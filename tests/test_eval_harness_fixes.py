import json
import importlib
from pathlib import Path
from unittest import mock

from mle_star_agent import config
from mle_star_agent.guards.code_validator_agent import (
    generated_script_contract_rejection_reasons,
)
from mle_star_agent.shared.checkpoint_io import load_checkpoint
from mle_star_agent.shared.checkpoint_lineage import file_sha256
from mle_star_agent.shared.script_template import get_script_template


def _minimal_generated_script(extra: str) -> str:
    markers = "\n".join(
        [
            "print('PROBE_METRICS:', {})",
            "print('EPOCH_LOG:', {})",
            "print('CALIBRATION_STATS:', {})",
            "print('THRESHOLD_CURVE:', [])",
            "print('METRICS:', {})",
            "print('PREDICTIONS:', [])",
            "print('ERROR_ANALYSIS:', {})",
        ]
    )
    return f"""
epochs = DRY_RUN_EPOCHS if DRY_RUN else 20
{extra}
scheduler.step()
{markers}
"""


def test_template_forces_dataloader_workers_to_zero_off_cuda():
    script = get_script_template("/tmp/split.json")

    assert "NUM_WORKERS = 2 if device.type == 'cuda' else 0" in script
    assert "num_workers=NUM_WORKERS" in script


def test_template_aborts_before_metrics_when_eval_slice_is_empty():
    script = get_script_template("/tmp/split.json")

    assert "def _require_non_empty_eval" in script
    assert "_require_non_empty_eval('validation', val_metric_labels, val_sample_ids)" in script
    assert "_require_non_empty_eval('test', test_metric_labels, test_sample_ids)" in script


def test_validator_rejects_reducelronplateau_verbose_kwarg():
    script = _minimal_generated_script(
        "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)"
    )

    reasons = generated_script_contract_rejection_reasons(script)

    assert any("ReduceLROnPlateau verbose" in reason for reason in reasons)


def test_eval_slice_comparison_detects_archived_mismatch():
    from mle_star_agent.shared.eval_harness import compare_eval_slices

    baseline_metrics = {
        "split": "data_split_mixed.json",
        "ng_count": 40,
        "g_count": 39,
    }
    baseline_predictions = [{"sample_id": f"b{i}"} for i in range(79)]
    ablation_results = [
        {
            "lineage": {"data_split_sha256": "grouped-sha"},
            "metrics": {"ng_count": 62, "g_count": 40},
        }
    ]

    finding = compare_eval_slices(
        baseline_metrics=baseline_metrics,
        baseline_predictions=baseline_predictions,
        ablation_results=ablation_results,
        ablation_split_name="data_split_grouped.json",
    )

    assert finding["verdict"] == "different"
    assert finding["baseline_slice"] == "data_split_mixed.json test: 79 samples (NG=40, G=39)"
    assert finding["ablation_slice"] == "data_split_grouped.json test: 102 samples (NG=62, G=40)"


def test_current_candidate_and_ablation_paths_share_active_split(tmp_path):
    baseline_coder_agent = importlib.import_module(
        "mle_star_agent.phases.phase1_init.baseline_coder_agent"
    )
    ablation_agent = importlib.import_module(
        "mle_star_agent.phases.phase2_refinement.ablation_agent"
    )

    split_path = tmp_path / "data_split_mixed.json"
    split_path.write_text(json.dumps({"train": [], "val": [], "test": []}), encoding="utf-8")
    candidate_scripts_path = tmp_path / "candidate_scripts.json"

    class Context:
        state = {"input_modality": "stereo", "label_granularity": "board"}

    block = """
def build_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(9 * IMAGE_SIZE * IMAGE_SIZE, 1)).to(device)

def build_optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-3)

def build_scheduler(optimizer):
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2)
"""

    with (
        mock.patch.object(config, "CKPT_DATA_SPLIT", split_path),
        mock.patch.object(config, "CKPT_CANDIDATE_SCRIPTS", candidate_scripts_path),
    ):
        result = baseline_coder_agent.append_candidate_block_fn(
            Context(), "linear", block, "linear"
        )
        saved = load_checkpoint(candidate_scripts_path)
        rendered = saved["scripts"][0]["script"]
        lineage = ablation_agent._ablation_lineage({"best_pipeline_script": rendered})

    assert "Saved script 'linear'" in result
    assert f"DATA_SPLIT_PATH = {str(split_path)!r}" in rendered
    assert lineage["data_split_sha256"] == file_sha256(split_path)


def test_loop_guard_stops_when_no_acceptable_path_remains_after_stagnation():
    from mle_star_agent.shared import loop_guard
    from mle_star_agent import config

    state = {
        "no_improve_count": config.INNER_STAGNATION_MAX_UNCONSTRAINED,
        "current_best_score": 0.50,
        "best_miss_rate": 0.50,
        "best_overkill_rate": 0.60,
        "best_accuracy": 0.55,
        "ablation_results": [
            {
                "status": "success",
                "metrics": {
                    "ng_recall": 0.90,
                    "miss_rate": 0.10,
                    "overkill_rate": 0.40,
                    "accuracy": 0.60,
                },
            },
            {"status": "failed", "metrics": None},
        ],
    }

    assert loop_guard.should_stop_when_no_acceptable_path_remains(state) is True
