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


def run_script(script: str, timeout: int = config.TIMEOUT_SECONDS, env: Optional[dict] = None) -> RunResult:
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
