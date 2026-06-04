"""Pure helper for the MLEvolve persistent AOI knowledge base.

`make_code_diff` turns a (prev_best, current) script pair into a compact,
truncated unified diff suitable for cross-run memory. We store the diff — never
the full script — so the planner prompt footprint stays bounded.
"""

import difflib

# Hard cap on the stored diff length. ~4000 chars keeps a single KB record small
# enough that a handful of them never dominate the planner prompt.
DIFF_CHAR_CAP = 4000


def make_code_diff(old: str, new: str, cap: int = DIFF_CHAR_CAP) -> str:
    """Return a TRUNCATED unified diff from ``old`` -> ``new``.

    - If ``old`` is empty/whitespace (no previous best script), emit a short
      "new script" header note followed by the new content as ``+`` additions,
      rather than diffing against nothing.
    - The result is capped at ``cap`` characters; overflow is replaced with a
      truncation marker so we never store an unbounded blob.
    """
    old = old or ""
    new = new or ""

    if not old.strip():
        body = "\n".join("+" + line for line in new.splitlines())
        diff = "@@ new script (no previous best) @@\n" + body
    else:
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile="prev_best",
                tofile="current",
                n=2,
            )
        )

    if len(diff) > cap:
        diff = diff[:cap] + f"\n... [truncated, {len(diff)} chars total]"
    return diff
