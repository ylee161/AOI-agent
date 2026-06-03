#!/usr/bin/env bash
# Retry wrapper for `adk run` — restarts the pipeline on transient API errors
# (429/503/rate-limit). Checkpoints ensure no completed work is repeated on
# restart, so re-running resumes rather than starting over.
#
# Usage:  ./run.sh
# Tunable via env:  MAX_RETRIES (default 100)   RETRY_DELAY seconds (default 60)
#
# Tip: add a shortcut to your shell rc, e.g.
#   alias aoi='bash /path/to/AOI-agent/run.sh'
set -u
cd "$(dirname "$0")"

# Prefer the project venv's adk if present, else whatever `adk` is on PATH.
ADK="adk"
[ -x ".venv/bin/adk" ] && ADK=".venv/bin/adk"

MAX_RETRIES="${MAX_RETRIES:-100}"
RETRY_DELAY="${RETRY_DELAY:-60}"

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo ""
    echo "=== Attempt $attempt / $MAX_RETRIES ($(date)) ==="
    echo "start" | "$ADK" run mle_star_agent
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Pipeline completed successfully."
        exit 0
    fi

    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        echo "Pipeline exited with code $EXIT_CODE. Waiting ${RETRY_DELAY}s before retry..."
        sleep "$RETRY_DELAY"
    fi
done

echo "Max retries ($MAX_RETRIES) reached. Check logs."
exit 1
