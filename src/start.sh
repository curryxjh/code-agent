#!/usr/bin/env bash
# Launch the coding agent from any directory.
#
# Usage:
#   ./start.sh                              # default AI Coding
#   AGENT_NAME="My Agent" ./start.sh        # custom name
#   AGENT_ICON="🌹" ./start.sh             # custom icon
#
# The agent operates on your current working directory ($PWD),
# not on the directory where this script lives.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate venv if present
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Run cli.py — sys.path is fixed inside the script, so cwd can be anywhere
python "$SCRIPT_DIR/cli.py" "$@"