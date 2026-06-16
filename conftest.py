"""Global test sandbox: no test may touch the real checkpoints/ directory.

A test once called consolidate_candidate_scores_fn without patching
CKPT_CANDIDATE_SCORES and wrote a stub-fixture candidate ("candidate_0",
architecture="stub", fake METRICS) into the production
checkpoints/candidate_scores.json. The live agent then loaded that checkpoint
and selected the stub as its best Phase 1 candidate, poisoning the entire run.

This fixture redirects config.CHECKPOINT_DIR and every config.CKPT_* path that
lives under the real checkpoints/ directory into a per-test tmp directory, so
checkpoint reads start empty and writes can never reach the repo. Tests that
need a pre-existing checkpoint must create it inside the sandbox (or apply
their own mock.patch on top, which still wins over this fixture).
Read-only *inputs* (the data-split files) are copied into the sandbox so tests
that legitimately consume the split keep working; everything else starts empty.
"""
import shutil
from pathlib import Path

import pytest

from mle_star_agent import config

_REAL_CHECKPOINT_DIR = (config.PROJECT_ROOT / "checkpoints").resolve()

# Pure inputs (produced by data preparation, never written by agent decisions).
_SEED_FILES = ("data_split_grouped.json", "data_split_mixed.json", "data_split.json")


@pytest.fixture(autouse=True)
def sandbox_checkpoints(tmp_path, monkeypatch):
    sandbox = tmp_path / "checkpoints"
    sandbox.mkdir(parents=True, exist_ok=True)
    for seed_name in _SEED_FILES:
        seed_src = _REAL_CHECKPOINT_DIR / seed_name
        if seed_src.is_file():
            shutil.copy2(seed_src, sandbox / seed_name)
    monkeypatch.setattr(config, "CHECKPOINT_DIR", sandbox)
    for name in dir(config):
        value = getattr(config, name)
        if not isinstance(value, Path):
            continue
        try:
            rel = value.resolve().relative_to(_REAL_CHECKPOINT_DIR)
        except ValueError:
            continue
        monkeypatch.setattr(config, name, sandbox / rel)
    yield sandbox
