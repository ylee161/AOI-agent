import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mle_star_agent import config


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool


# ─── Debug-mode (accelerated) patching ───────────────────────────────────────
# When run_script(debug_mode=True) is requested we rewrite the script *text*
# before execution so a broken or slow script fails fast instead of timing out:
#   1. the full-run epoch count is forced to config.CURVE_ABORT_DEBUG_EPOCHS
#      (a SHORT curve, not a single noisy epoch, so the smoke run can emit an
#      extrapolatable per-epoch learning curve) — covering both the mandated
#      `epochs = DRY_RUN_EPOCHS if DRY_RUN else 20` ternary and bare integer
#      assignments to training-length epoch variables, and
#   2. the first argument of every DataLoader(...) call is wrapped in a helper
#      that subsets the dataset to 5% of its samples.
# The rewrite is applied to a local copy only — the caller's `script` string
# (the one that actually gets scored) is never mutated.

# Cap the full-run epoch count to the debug cap so the smoke run stays short.
# Two forms are handled, because real scripts (per the coder-agent prompts) emit
# the *ternary* form, not a bare literal:
#   1. _EPOCH_TERNARY_RE rewrites the post-`else` full-run literal in the mandated
#        epochs = DRY_RUN_EPOCHS if DRY_RUN else 20
#      ternary. The debug run does NOT set DRY_RUN, so the script takes the `else`
#      branch — without this rewrite it would run the full 20 epochs and the
#      "short curve" the curve-abort relies on would never materialise.
#   2. _EPOCH_ASSIGN_RE rewrites a bare integer literal assigned to a
#      TRAINING-length epoch variable (epochs / num_epochs / max_epochs / ...).
#
# The LHS name is preserved so downstream references (e.g. `range(num_epochs)`)
# keep working. We deliberately do NOT match scheduler/counter variables such as
# warmup_epochs, patience_epochs, best_epoch or epochs_done: forcing those to the
# cap would corrupt training/scheduler semantics and could manufacture a
# misleading learning curve. The literal pattern also swallows any float/exponent
# tail (e.g. `1e3`) so it is replaced whole rather than leaving a stray `e3`.
# Both require `= <number>` (single `=`), so neither ever matches `==`.
_EPOCH_TERNARY_RE = re.compile(r"(\bif\s+DRY_RUN\s+else\s+)\d+\b")
_EPOCH_ASSIGN_RE = re.compile(
    r"\b((?:num_|n_|max_|total_|train_|num_train_)?epochs?\s*=\s*)"
    r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?",
    re.IGNORECASE,
)

# Match `DataLoader(` followed by an optional `dataset=` keyword and the first
# dataset identifier, so we can wrap that identifier in the 5% cap helper.
_DATALOADER_RE = re.compile(
    r"(\bDataLoader\s*\(\s*)(dataset\s*=\s*)?([A-Za-z_][\w.]*)",
)

# Prepended to every debug-mode script. Subsets a dataset to 5% of its samples;
# degrades to a no-op for anything that is not a sized torch dataset.
_DEBUG_CAP_HELPER = (
    "def __aoi_cap5(__ds):\n"
    "    try:\n"
    "        import torch.utils.data as __tud\n"
    "        __n = len(__ds)\n"
    "        __k = max(1, int(__n * 0.05))\n"
    "        if __k >= __n:\n"
    "            return __ds\n"
    "        return __tud.Subset(__ds, list(range(__k)))\n"
    "    except Exception:\n"
    "        return __ds\n"
)


def apply_debug_patches(script: str) -> str:
    """Return a debug-accelerated copy of `script` (caps epochs to
    config.CURVE_ABORT_DEBUG_EPOCHS and data to 5%).

    Pure function over the script text — does not touch the input string.
    """
    epoch_cap = str(int(config.CURVE_ABORT_DEBUG_EPOCHS))
    patched = _EPOCH_TERNARY_RE.sub(lambda m: m.group(1) + epoch_cap, script)
    patched = _EPOCH_ASSIGN_RE.sub(lambda m: m.group(1) + epoch_cap, patched)
    patched = _DATALOADER_RE.sub(
        lambda m: m.group(1) + (m.group(2) or "") + "__aoi_cap5(" + m.group(3) + ")",
        patched,
    )
    return _DEBUG_CAP_HELPER + "\n" + patched


def run_script(
    script: str,
    timeout: int = config.TIMEOUT_SECONDS,
    env: Optional[dict] = None,
    debug_mode: bool = False,
) -> RunResult:
    if debug_mode:
        script = apply_debug_patches(script)
        debug_env = {
            # Generated scripts use DRY_RUN to skip expensive prediction dumps.
            # The DataLoader patch remains the actual 5% sample limiter.
            "DRY_RUN": "1",
            "DRY_RUN_EPOCHS": str(int(config.CURVE_ABORT_DEBUG_EPOCHS)),
            "DRY_RUN_SAMPLES": "999999",
        }
        env = {**debug_env, **(env or {})}
        # Debug runs are a fast smoke-check: cap the timeout regardless of config.
        timeout = min(timeout, config.DEBUG_CHECK_TIMEOUT_SECONDS)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        return run_script_file(Path(script_path), timeout=timeout, env=env)
    finally:
        Path(script_path).unlink(missing_ok=True)


def run_script_file(path: Path, timeout: int = config.TIMEOUT_SECONDS, env: Optional[dict] = None) -> RunResult:
    import os
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = -1
        raw_out = e.stdout
        raw_err = e.stderr
        stdout = (raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else raw_out) or ""
        stderr = (raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else raw_err) or f"Script timed out after {timeout}s"

    duration_ms = (time.monotonic() - start) * 1000
    return RunResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
