#!/usr/bin/env bash
# Start the ADK web UI for the AOI agent and open the browser at it.
# Usage: ./start_web.sh [port]   (default port 8000)
set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8000}"
URL="http://127.0.0.1:${PORT}"

cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# Prefer the project venv's adk if present, else whatever `adk` is on PATH.
ADK="adk"
[ -x "${PROJECT_DIR}/.venv/bin/adk" ] && ADK="${PROJECT_DIR}/.venv/bin/adk"

# Cross-platform browser opener: macOS `open`, Linux `xdg-open`.
opener() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1"
  fi
}

# If the port is already serving, don't start a duplicate — just open the UI.
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ADK web already running on ${URL} — opening browser."
  opener "$URL"
  exit 0
fi

echo "Starting ADK web UI on ${URL} ..."

# Wait in the background until the server accepts connections, then open browser.
(
  for _ in $(seq 1 60); do
    if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      opener "$URL"
      break
    fi
    sleep 0.5
  done
) &

# Run the server in the foreground (Ctrl-C to stop). `adk web .` discovers the
# mle_star_agent package in this directory.
exec "$ADK" web \
  --port "${PORT}" \
  --session_service_uri "sqlite:///${PROJECT_DIR}/.adk/adk_web_sessions.db" \
  .
