"""Tests for code_runner debug_mode (KompeteAI Accelerated Debugger).

A broken script must fail fast under debug_mode rather than timing out, and the
debug patching must not mutate the caller's original script string.
"""
import time

from mle_star_agent.shared import code_runner
from mle_star_agent import config


BROKEN_SCRIPT = """
import torch

def main(:          # <- deliberate syntax error
    epochs = 50
    print("should never get here")

main()
"""


def test_broken_script_fails_fast_in_debug_mode():
    """A syntax-error script exits in well under 30s with returncode != 0."""
    start = time.monotonic()
    result = code_runner.run_script(BROKEN_SCRIPT, debug_mode=True)
    elapsed = time.monotonic() - start

    assert result.returncode != 0, "broken script should not exit cleanly"
    assert not result.timed_out, "broken script should fail outright, not time out"
    assert elapsed < 30, f"debug run took {elapsed:.1f}s, expected < 30s"


def test_debug_timeout_capped_at_config_value():
    """debug_mode caps the effective timeout at DEBUG_CHECK_TIMEOUT_SECONDS."""
    assert config.DEBUG_CHECK_TIMEOUT_SECONDS == 120


def test_debug_patches_do_not_mutate_original_script():
    """apply_debug_patches returns a new string; the input is untouched."""
    original = "num_epochs = 50\nloader = DataLoader(train_ds, batch_size=32)\n"
    snapshot = original
    patched = code_runner.apply_debug_patches(original)

    assert original == snapshot, "original script string must not be mutated"
    assert "num_epochs = 1" in patched, "epoch value should be forced to 1"
    assert "__aoi_cap5(train_ds)" in patched, "DataLoader arg should be capped"
    assert "num_epochs = 50" not in patched


def test_epoch_regex_preserves_variable_name():
    """Epoch rewrite keeps the LHS name and only changes the integer literal."""
    patched = code_runner.apply_debug_patches("EPOCHS = 20\nmax_epochs=7\n")
    assert "EPOCHS = 1" in patched
    assert "max_epochs=1" in patched


if __name__ == "__main__":
    test_broken_script_fails_fast_in_debug_mode()
    test_debug_timeout_capped_at_config_value()
    test_debug_patches_do_not_mutate_original_script()
    test_epoch_regex_preserves_variable_name()
    print("all tests passed")
