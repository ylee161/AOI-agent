"""Mini-goal 7 regression checks for grouped split and metric guards."""

import json
from collections import defaultdict
from pathlib import Path

from mle_star_agent import config
from mle_star_agent.shared import metric_guard


ROOT = Path(__file__).parent


def _board_group(sample_id: str) -> str:
    lot, pair_key = sample_id.split("/", 1)
    row = pair_key.split("-", 1)[0]
    return f"{lot}::row{row}"


def test_grouped_split_has_no_board_group_overlap():
    split = json.loads((ROOT / "checkpoints" / "data_split_grouped.json").read_text())
    group_to_splits = defaultdict(set)
    for split_name in ("train", "val", "test"):
        for sample in split[split_name]:
            group_to_splits[_board_group(sample["sample_id"])].add(split_name)

    leaked = {group: sorted(parts) for group, parts in group_to_splits.items() if len(parts) > 1}
    assert leaked == {}
    assert split["stats"]["zero_leakage"] is True


def test_root_dummy_data_split_json_is_absent():
    assert not (ROOT / "data_split.json").exists()


def test_pipeline_defaults_to_grouped_split_path():
    expected = ROOT / "checkpoints" / "data_split_grouped.json"
    assert config.CKPT_DATA_SPLIT == expected

    best = json.loads((ROOT / "checkpoints" / "best_pipeline.json").read_text())
    script = best.get("best_pipeline_script", "")
    assert str(expected) in script
    assert str(ROOT / "checkpoints" / "data_split.json") not in script


def test_degenerate_metric_guard_rejects_dummy_and_flat_results():
    dummy = {
        "ng_count": 1,
        "g_count": 1,
        "prob_gap": 0.0,
        "roc_auc": 0.0,
        "miss_rate": 0.0,
        "overkill_rate": 1.0,
        "accuracy": 0.5,
    }
    assert not metric_guard.is_persistable(dummy, duration_ms=2600)

    flat = {"ng_count": 31, "g_count": 29, "prob_gap": 0.0, "roc_auc": 0.5}
    assert not metric_guard.is_persistable(flat, duration_ms=300_000)
    assert metric_guard.scores_effectively_identical([0.391, 0.391, 0.391])
