import os

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")


def test_fold_aggregation_uses_mean_and_worst_fold_values():
    from mle_star_agent.phases.phase2_refinement.evaluator_agent import (
        _aggregate_cv_fold_metrics,
    )

    folds = [
        {
            "ng_recall": 0.98,
            "overkill_rate": 0.10,
            "accuracy": 0.94,
            "miss_rate": 0.02,
        },
        {
            "ng_recall": 0.94,
            "overkill_rate": 0.06,
            "accuracy": 0.92,
            "miss_rate": 0.06,
        },
        {
            "ng_recall": 1.00,
            "overkill_rate": 0.08,
            "accuracy": 0.96,
            "miss_rate": 0.00,
        },
    ]

    aggregated = _aggregate_cv_fold_metrics(folds)

    assert aggregated["cv_fold_count"] == 3
    assert aggregated["mean_val_ng_recall"] == pytest.approx(0.9733333333)
    assert aggregated["worst_fold_val_ng_recall"] == pytest.approx(0.94)
    assert aggregated["mean_val_overkill"] == pytest.approx(0.08)
    assert aggregated["worst_fold_val_overkill"] == pytest.approx(0.10)
    assert aggregated["mean_val_accuracy"] == pytest.approx(0.94)
    assert aggregated["worst_fold_val_accuracy"] == pytest.approx(0.92)
    assert aggregated["mean_val_miss_rate"] == pytest.approx(0.0266666667)
    assert aggregated["worst_fold_val_miss_rate"] == pytest.approx(0.06)


def test_board_grouped_kfold_validation_boards_are_non_overlapping():
    from mle_star_agent.shared.data_split import board_grouped_kfold

    samples = []
    for board_index in range(6):
        board_code = f"VHBTEST{board_index:02d}"
        for sample_index in range(3):
            samples.append({
                "sample_id": f"{board_code}/{sample_index}",
                "board_code": board_code,
                "label": "NG" if sample_index == 0 else "G",
            })

    folds = board_grouped_kfold(samples, k=3)

    assert len(folds) == 3
    seen_val_boards = set()
    for train_df, val_df in folds:
        train_boards = set(train_df["board_code"])
        val_boards = set(val_df["board_code"])
        assert train_boards
        assert val_boards
        assert train_boards.isdisjoint(val_boards)
        assert seen_val_boards.isdisjoint(val_boards)
        seen_val_boards.update(val_boards)

    assert seen_val_boards == {f"VHBTEST{board_index:02d}" for board_index in range(6)}


def test_acceptance_view_uses_conservative_cv_decision_metrics():
    from mle_star_agent.shared.acceptance_scoring import metrics_view

    aggregated = {
        "mean_val_ng_recall": 0.98,
        "worst_fold_val_ng_recall": 0.94,
        "mean_val_overkill": 0.07,
        "worst_fold_val_overkill": 0.12,
        "mean_val_accuracy": 0.93,
        "worst_fold_val_accuracy": 0.88,
        "mean_val_miss_rate": 0.02,
        "worst_fold_val_miss_rate": 0.06,
    }

    view = metrics_view(aggregated)

    assert view["ng_recall"] == pytest.approx(0.94)
    assert view["miss_rate"] == pytest.approx(0.06)
    assert view["overkill_rate"] == pytest.approx(0.07)
    assert view["accuracy"] == pytest.approx(0.93)
