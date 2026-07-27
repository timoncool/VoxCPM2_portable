#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

command -v git >/dev/null 2>&1 || {
    printf '%s\n' 'ERROR: Git is required to update this checkout.' >&2
    exit 1
}

[[ -d "$SCRIPT_DIR/.git" ]] || {
    printf '%s\n' 'ERROR: This directory is not a Git checkout.' >&2
    exit 1
}

printf '%s\n' 'Pulling updates from GitHub...'
git pull --ff-only

if [[ -x "$VENV_PYTHON" ]]; then
    printf '%s\n' 'Updating Python dependencies...'
    "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
else
    printf '%s\n' 'WARNING: .venv is missing. Run ./install.sh to install dependencies.' >&2
fi

printf '%s\n' 'Update complete. Run ./run.sh to start VoxCPM2.'
