"""Tests for per_side (Arm B) label granularity in the data pipeline + template.

per_side turns each side of a stereo board into its own 3-channel mono sample,
labelled by its OWN defect count, and pools the two sides back to a board score
(max) at eval so the headline metric stays board-level. board mode (the 9-channel
stereo default) must remain byte-for-byte unchanged. HONEST NOTE under test: this
attacks the under-learning / clean-side-dilution axis only — it is NOT expected to
move cross-lot generalisation.
"""
import tempfile
from pathlib import Path

from mle_star_agent.shared import data_split as ds
from mle_star_agent.shared.script_template import get_script_template


# A stereo dataset stand-in: one lot folder of stereo PNGs + a Map sheet carrying
# TestResult and per-side NG_SUM_L / NG_SUM_R counts.
def _make_lot(root: Path, lot: str, rows):
    folder = root / lot
    folder.mkdir(parents=True)
    import pandas as pd
    recs = []
    for (r, c, res, ngl, ngr) in rows:
        for side in ("L", "R"):
            (folder / f"{r}-{c}_{side}_{lot}_X.png").write_bytes(b"")
        recs.append({"Row": r, "Column": c, "TestResult": res,
                     "NG_SUM_L": ngl, "NG_SUM_R": ngr})
    pd.DataFrame(recs).to_excel(folder / f"{lot}_Map.xlsx", index=False)
    return str(folder)


def _toy_folders(root: Path):
    # Four lots (four board_codes) with a balanced mix of pass / one-sided fail /
    # two-sided fail boards — enough boards per class for the stratified split.
    # board (r,c): TestResult, NG_SUM_L, NG_SUM_R
    rows = [
        (0, 0, "Pass", 0, 0), (0, 1, "Fail", 3, 0),   # clean / one-sided-L NG
        (1, 0, "Fail", 0, 2), (1, 1, "Pass", 0, 0),   # one-sided-R NG / clean
        (2, 0, "Fail", 1, 4), (2, 1, "Pass", 0, 0),   # two-sided NG / clean
    ]
    return [_make_lot(root, code, rows)
            for code in ("VHB001A", "VHB002B", "VHB003C", "VHB004D")]


def test_per_side_emits_paired_mono_samples_with_board_keys():
    with tempfile.TemporaryDirectory() as tmp:
        folders = _toy_folders(Path(tmp))
        assert ds.dataset_supports_per_side(folders) is True
        split = ds.build_data_split(folders, label_granularity="per_side")

        assert split["metadata"]["label_granularity"] == "per_side"
        # per_side images are single-view -> downstream must treat as mono.
        assert split["metadata"]["input_modality"] == "mono"

        alls = split["train"] + split["val"] + split["test"]
        assert len(alls) == 48  # 24 boards * 2 sides
        for s in alls:
            assert "img" in s and "img_l" not in s          # mono sample
            assert s["label"] in ("NG", "G")
            assert s["board_label"] in ("NG", "G")
            assert s["sample_id"].endswith(("::L", "::R"))

        # The defining property: a side is NG iff its OWN count > 0, which removes
        # the clean-side dilution (a clean side of a failing board is labelled G).
        by_id = {s["sample_id"]: s for s in alls}
        one_sided = by_id["VHB001A/0-1_VHB001A_X::L"]
        assert one_sided["label"] == "NG" and one_sided["board_label"] == "NG"
        clean_side = by_id["VHB001A/0-1_VHB001A_X::R"]
        assert clean_side["label"] == "G" and clean_side["board_label"] == "NG"


def test_per_side_keeps_both_sides_in_same_partition():
    with tempfile.TemporaryDirectory() as tmp:
        folders = _toy_folders(Path(tmp))
        split = ds.build_data_split(folders, label_granularity="per_side")
        where = {}
        for name in ("train", "val", "test"):
            for s in split[name]:
                where.setdefault(s["board_id"], set()).add(name)
        # Pooling is only valid if a board's two sides are never split across
        # partitions.
        assert all(len(v) == 1 for v in where.values())


def test_board_mode_unchanged_when_default():
    with tempfile.TemporaryDirectory() as tmp:
        folders = _toy_folders(Path(tmp))
        split = ds.build_data_split(folders)  # default == board
        assert split["metadata"]["label_granularity"] == "board"
        assert split["metadata"]["input_modality"] == "stereo"
        alls = split["train"] + split["val"] + split["test"]
        assert len(alls) == 24  # one stereo sample per board
        for s in alls:
            assert "img_l" in s and "img_r" in s


def _make_lot_no_side_cols(root: Path, lot: str, rows):
    """Stereo lot whose label sheet has only a board TestResult (no NG_SUM_L/R)."""
    folder = root / lot
    folder.mkdir(parents=True)
    import pandas as pd
    recs = []
    for (r, c, res) in rows:
        for side in ("L", "R"):
            (folder / f"{r}-{c}_{side}_{lot}_X.png").write_bytes(b"")
        recs.append({"Row": r, "Column": c, "TestResult": res})
    pd.DataFrame(recs).to_excel(folder / f"{lot}_Map.xlsx", index=False)
    return str(folder)


def test_per_side_falls_back_when_no_side_columns(tmp_path):
    rows = [(0, 0, "Pass"), (0, 1, "Fail"), (1, 0, "Fail"),
            (1, 1, "Pass"), (2, 0, "Fail"), (2, 1, "Pass")]
    folders = [_make_lot_no_side_cols(tmp_path, code, rows)
               for code in ("VHB001A", "VHB002B", "VHB003C", "VHB004D")]
    # No NG_SUM_L/R columns -> per_side unsupported -> graceful board fallback.
    assert ds.dataset_supports_per_side(folders) is False
    split = ds.build_data_split(folders, label_granularity="per_side")
    assert split["metadata"]["label_granularity"] == "board"
    assert split["metadata"]["input_modality"] == "stereo"


def test_template_per_side_renders_pooling_and_3ch_model():
    block = (
        "import torchvision\n"
        "IN_CHANNELS = 3\nFEATURE_DIFF_CANDIDATE = False\n"
        "def build_model():\n"
        "    m = torchvision.models.resnet18(weights=None)\n"
        "    import torch.nn as _nn; m.fc = _nn.Linear(m.fc.in_features, 1)\n"
        "    return m.to(device)\n"
        "PROBE_EPOCHS = 1\n"
    )
    script = get_script_template(
        data_split_path="/tmp/x.json",
        architecture_block=block,
        label_granularity="per_side",
    )
    compile(script, "<per_side>", "exec")
    assert "class PerSideDataset" in script
    assert "_board_pooled_auc" in script
    assert "board_pooled_val_roc_auc" in script
    assert "PER_SIDE = LABEL_GRANULARITY == 'per_side'" in script


def test_template_selects_best_checkpoint_by_validation_auc_not_loss():
    block = (
        "import torchvision\n"
        "IN_CHANNELS = 3\nFEATURE_DIFF_CANDIDATE = False\n"
        "def build_model():\n"
        "    m = torchvision.models.resnet18(weights=None)\n"
        "    import torch.nn as _nn; m.fc = _nn.Linear(m.fc.in_features, 1)\n"
        "    return m.to(device)\n"
        "PROBE_EPOCHS = 1\n"
    )
    script = get_script_template(
        data_split_path="/tmp/x.json",
        architecture_block=block,
        label_granularity="per_side",
    )
    compile(script, "<per_side>", "exec")

    assert "best_val_auc = -math.inf" in script
    assert "val_auc_for_selection = _selection_auc(val_probs, val_labels, val_sample_ids)" in script
    assert "val_board_pooled_auc" in script
    assert "if _is_better_checkpoint(val_auc_for_selection, val_loss):" in script
    assert "if val_loss < best_val_loss:" not in script


def test_template_calibrates_threshold_for_validation_ng_recall_target():
    block = (
        "import torchvision\n"
        "IN_CHANNELS = 3\nFEATURE_DIFF_CANDIDATE = False\n"
        "def build_model():\n"
        "    m = torchvision.models.resnet18(weights=None)\n"
        "    import torch.nn as _nn; m.fc = _nn.Linear(m.fc.in_features, 1)\n"
        "    return m.to(device)\n"
        "PROBE_EPOCHS = 1\n"
    )
    script = get_script_template(
        data_split_path="/tmp/x.json",
        architecture_block=block,
        label_granularity="per_side",
    )
    compile(script, "<per_side>", "exec")

    assert "VAL_NG_RECALL_TARGET = float(os.getenv('VAL_NG_RECALL_TARGET', '0.90'))" in script
    assert "recall_candidates = [c for c in all_candidates if c['recall'] >= VAL_NG_RECALL_TARGET]" in script
    assert "best_threshold = min(recall_candidates, key=lambda c: c['threshold'])['threshold']" in script
    assert "'threshold_selection_target': VAL_NG_RECALL_TARGET" in script
    assert "'val_ng_recall_at_threshold': val_threshold_metrics['ng_recall']" in script
