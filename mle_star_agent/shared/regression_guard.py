"""Anti-regression guard for best_pipeline.json writes.

The persistent KB (checkpoints/persistent_aoi_kb.json) may contain one or more
"hard-floor" entries — records flagged with ``is_floor: true`` whose
``floor_score.roc_auc`` is a proven result we must never regress below. A run
that starts cold or resets can otherwise overwrite a good best_pipeline.json
with a collapsed candidate (the 0.697 -> 0.067 regression we are fixing).

This module exposes the floor lookup and a single decision helper that every
best_pipeline.json writer consults before persisting.
"""

import logging

from mle_star_agent import config
from mle_star_agent.shared.checkpoint_io import (
    checkpoint_exists,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


def kb_floor_roc_auc(kb_path=None):
    """Return the highest ``floor_score.roc_auc`` among ``is_floor`` KB entries.

    Returns ``None`` when no floor entry exists or the KB is missing/unreadable —
    in that case there is nothing to protect and writes proceed unguarded.
    """
    path = kb_path if kb_path is not None else config.CKPT_PERSISTENT_KB
    if not checkpoint_exists(path):
        return None
    try:
        kb = load_checkpoint(path)
    except RuntimeError:
        logger.warning("Regression guard: KB at %s unreadable; skipping floor check.", path)
        return None

    # The KB is an append-only list of records; tolerate a dict wrapper too.
    if isinstance(kb, dict):
        records = kb.get("records") or kb.get("entries") or []
    elif isinstance(kb, list):
        records = kb
    else:
        return None

    floors = []
    for entry in records:
        if not isinstance(entry, dict) or not entry.get("is_floor"):
            continue
        floor_score = entry.get("floor_score") or {}
        roc = floor_score.get("roc_auc") if isinstance(floor_score, dict) else None
        if roc is None:
            continue
        try:
            floors.append(float(roc))
        except (TypeError, ValueError):
            continue

    return max(floors) if floors else None


def regression_blocked(candidate_roc_auc, kb_path=None):
    """Decide whether a best_pipeline.json overwrite must be blocked.

    Returns ``(blocked, floor)`` where ``floor`` is the KB hard-floor roc_auc
    (or ``None`` if there is no floor to enforce). ``blocked`` is True only when
    a floor exists and ``candidate_roc_auc`` is strictly below it.
    """
    floor = kb_floor_roc_auc(kb_path)
    if floor is None:
        return False, None
    try:
        candidate = float(candidate_roc_auc or 0.0)
    except (TypeError, ValueError):
        candidate = 0.0
    blocked = candidate < floor
    if blocked:
        logger.warning(
            "REGRESSION BLOCKED: candidate roc_auc %.4f < KB floor %.4f", candidate, floor
        )
    return blocked, floor
