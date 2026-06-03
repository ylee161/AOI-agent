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
#   1. any assignment to an "*epoch*" variable has its RHS forced to 1, and
#   2. the first argument of every DataLoader(...) call is wrapped in a helper
#      that subsets the dataset to 5% of its samples.
# The rewrite is applied to a local copy only — the caller's `script` string
# (the one that actually gets scored) is never mutated.

# Match assignments whose LHS variable name contains "epoch" (case-insensitive)
# and replace only the integer literal on the RHS with 1. The variable name is
# preserved so any downstream reference (e.g. `range(num_epochs)`) keeps working.
# Requires `= <int>` (single `=`), so it never matches `==` comparisons.
_EPOCH_ASSIGN_RE = re.compile(
    r"\b(\w*epochs?\w*\s*=\s*)\d+",
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
    """Return a debug-accelerated copy of `script` (caps epochs to 1 and data to 5%).

    Pure function over the script text — does not touch the input string.
    """
    patched = _EPOCH_ASSIGN_RE.sub(lambda m: m.group(1) + "1", script)
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
