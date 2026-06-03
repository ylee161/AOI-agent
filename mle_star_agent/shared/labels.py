"""Canonical PASS/FAIL label normalisation — the single boundary that turns a
dataset's raw label strings into the pipeline's internal ``G`` / ``NG`` codes.

The rest of the pipeline (diagnosis, metrics, phase agents) speaks only in the
canonical ``G`` (pass / good) and ``NG`` (fail / defect) codes. To support a
different label convention you do NOT touch any of that code — you only declare
which raw values map to pass vs fail via ``PASS_LABELS`` / ``FAIL_LABELS`` in
``config.py``. This module reads those sets (falling back to the defaults below,
which reproduce the historical G/NG behaviour) and does the mapping.

Matching is case-insensitive and whitespace-trimmed.
"""

import logging

logger = logging.getLogger(__name__)

# Internal canonical codes — do not change; the whole pipeline depends on them.
CANON_FAIL = "NG"
CANON_PASS = "G"

# Defaults preserve the historical behaviour. They are the UNION of every raw
# value the codebase previously recognised in either the data splitter or the
# metrics parser, so adopting this shared normaliser cannot reclassify or drop
# any label that used to be accepted. Override in config.py for new datasets.
DEFAULT_FAIL_LABELS = {"fail", "ng", "1", "defect", "defective", "true", "positive"}
DEFAULT_PASS_LABELS = {"pass", "ok", "g", "good", "0", "false", "negative"}


def _configured_sets() -> tuple[set, set]:
    """Return (pass_set, fail_set) as lower-cased strings.

    Reads ``PASS_LABELS`` / ``FAIL_LABELS`` from config when importable; falls
    back to the history-preserving defaults otherwise (keeps this module usable
    standalone, e.g. in tests, without requiring the full config / API key).
    """
    try:
        from mle_star_agent import config

        pass_raw = getattr(config, "PASS_LABELS", DEFAULT_PASS_LABELS)
        fail_raw = getattr(config, "FAIL_LABELS", DEFAULT_FAIL_LABELS)
    except Exception:  # config not importable (e.g. missing API key in a test)
        pass_raw, fail_raw = DEFAULT_PASS_LABELS, DEFAULT_FAIL_LABELS

    pass_set = {str(v).strip().lower() for v in pass_raw}
    fail_set = {str(v).strip().lower() for v in fail_raw}
    return pass_set, fail_set


def normalize_label(value, *, strict: bool = False, context: str = ""):
    """Map a raw label ``value`` to the canonical ``"G"`` / ``"NG"`` code.

    - Returns ``"NG"`` if the value is in the configured FAIL set,
      ``"G"`` if in the configured PASS set (case-insensitive).
    - ``strict=True``: raise ``ValueError`` on an unrecognised value — used at
      data-ingest so a mislabelled / unconfigured value fails loudly instead of
      being silently dropped.
    - ``strict=False``: return ``None`` on an unrecognised value so callers can
      apply their own fallback.
    """
    if value is None:
        return None
    raw = str(value).strip()
    key = raw.lower()
    pass_set, fail_set = _configured_sets()

    if key in fail_set:
        return CANON_FAIL
    if key in pass_set:
        return CANON_PASS

    if strict:
        where = f" in {context}" if context else ""
        raise ValueError(
            f"Unrecognised label {raw!r}{where}. "
            f"Add it to PASS_LABELS or FAIL_LABELS in config.py. "
            f"Configured PASS={sorted(pass_set)}, FAIL={sorted(fail_set)}."
        )
    return None
