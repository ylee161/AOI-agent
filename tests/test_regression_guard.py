"""Tests for the anti-regression guard on best_pipeline.json writes.

Covers BUG 1: a candidate scoring below the KB hard floor (is_floor entry)
must NOT be allowed to overwrite best_pipeline.json; a candidate at or above
the floor must be allowed through.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from mle_star_agent import config
from mle_star_agent.shared import regression_guard
from mle_star_agent.shared.regression_guard import kb_floor_roc_auc, regression_blocked


FLOOR = 0.65


def _write_kb_with_floor(path: Path, floor_roc_auc: float = FLOOR) -> None:
    """Write a minimal KB list containing one is_floor entry."""
    path.write_text(
        json.dumps(
            [
                {"plan": "ordinary entry", "tags": ["regressed"], "metrics": {}},
                {
                    "is_floor": True,
                    "tags": ["proven_floor", "do_not_regress_below"],
                    "floor_score": {"roc_auc": floor_roc_auc},
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


# ── floor lookup ──────────────────────────────────────────────────────────────

def test_kb_floor_lookup_reads_is_floor_entry(tmp_path):
    kb = tmp_path / "kb.json"
    _write_kb_with_floor(kb)
    assert kb_floor_roc_auc(kb) == pytest.approx(FLOOR)


def test_kb_floor_none_when_no_floor_entry(tmp_path):
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps([{"plan": "x", "metrics": {"roc_auc": 0.9}}]), encoding="utf-8")
    assert kb_floor_roc_auc(kb) is None


def test_kb_floor_none_when_missing_file(tmp_path):
    assert kb_floor_roc_auc(tmp_path / "does_not_exist.json") is None


# ── decision helper ─────────────────────────────────────────────────────────────

def test_candidate_below_floor_is_blocked(tmp_path):
    kb = tmp_path / "kb.json"
    _write_kb_with_floor(kb)
    blocked, floor = regression_blocked(0.55, kb_path=kb)
    assert blocked is True
    assert floor == pytest.approx(FLOOR)


def test_candidate_above_floor_is_allowed(tmp_path):
    kb = tmp_path / "kb.json"
    _write_kb_with_floor(kb)
    blocked, floor = regression_blocked(0.70, kb_path=kb)
    assert blocked is False
    assert floor == pytest.approx(FLOOR)


def test_candidate_at_floor_is_allowed(tmp_path):
    kb = tmp_path / "kb.json"
    _write_kb_with_floor(kb)
    blocked, _ = regression_blocked(FLOOR, kb_path=kb)
    assert blocked is False


def test_no_floor_means_never_blocked(tmp_path):
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps([{"plan": "x"}]), encoding="utf-8")
    blocked, floor = regression_blocked(0.01, kb_path=kb)
    assert blocked is False
    assert floor is None


# ── end-to-end through _save_best_pipeline ──────────────────────────────────────

def _run_save(tmp_path, candidate_roc_auc):
    """Invoke _save_best_pipeline with config paths redirected to tmp_path.

    Returns the Path to best_pipeline.json (which may or may not exist).
    """
    # Import the *module* explicitly — the package namespace exposes an
    # ``evaluator_agent`` LlmAgent instance that would otherwise shadow it.
    import importlib

    evaluator_agent = importlib.import_module(
        "mle_star_agent.phases.phase2_refinement.evaluator_agent"
    )

    kb = tmp_path / "persistent_aoi_kb.json"
    best = tmp_path / "best_pipeline.json"
    _write_kb_with_floor(kb)

    state = {
        "current_best_score": candidate_roc_auc,
        "best_roc_auc": candidate_roc_auc,
        "best_pipeline_script": "print('candidate')",
        "best_overkill_rate": 0.1,
        "best_miss_rate": 0.1,
        "best_accuracy": 0.8,
        "best_f1": 0.8,
        "best_prob_gap": 0.05,
    }

    with (
        mock.patch.object(config, "CKPT_PERSISTENT_KB", kb),
        mock.patch.object(config, "CKPT_BEST_PIPELINE", best),
        # regression_guard imported the symbol into evaluator_agent's namespace,
        # but it resolves config lazily, so patching config is sufficient.
    ):
        evaluator_agent._save_best_pipeline(state)
    return best


def test_save_best_pipeline_blocks_below_floor(tmp_path):
    best = _run_save(tmp_path, 0.55)
    assert not best.exists(), "best_pipeline.json must NOT be written below the KB floor"


def test_save_best_pipeline_allows_above_floor(tmp_path):
    best = _run_save(tmp_path, 0.70)
    assert best.exists(), "best_pipeline.json must be written at/above the KB floor"
    data = json.loads(best.read_text())
    assert data["best_roc_auc"] == pytest.approx(0.70)
    assert data["best_pipeline_script"] == "print('candidate')"
