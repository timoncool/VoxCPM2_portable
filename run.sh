#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    printf '%s\n' 'ERROR: .venv was not found. Run ./install.sh first.' >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/app.py" ]]; then
    printf '%s\n' 'ERROR: app.py was not found.' >&2
    exit 1
fi

mkdir -p "$SCRIPT_DIR/temp" "$SCRIPT_DIR/models" "$SCRIPT_DIR/cache" "$SCRIPT_DIR/output"

export TEMP="$SCRIPT_DIR/temp"
export TMP="$SCRIPT_DIR/temp"
export GRADIO_TEMP_DIR="$SCRIPT_DIR/temp"
export HF_HOME="$SCRIPT_DIR/models"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/models"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/models"
export HF_DATASETS_CACHE="$SCRIPT_DIR/models/datasets"
export TORCH_HOME="$SCRIPT_DIR/models/torch"
export MODELSCOPE_CACHE="$SCRIPT_DIR/models/modelscope"
export XDG_CACHE_HOME="$SCRIPT_DIR/cache"
export OUTPUT_DIR="$SCRIPT_DIR/output"
export PYTHONIOENCODING="utf-8"
export PYTHONUNBUFFERED="1"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="false"

if [[ -x "$SCRIPT_DIR/ffmpeg/bin/ffmpeg" ]]; then
    export PATH="$SCRIPT_DIR/ffmpeg/bin:$PATH"
elif [[ -x "$SCRIPT_DIR/ffmpeg/ffmpeg" ]]; then
    export PATH="$SCRIPT_DIR/ffmpeg:$PATH"
fi

printf '%s\n' 'Starting VoxCPM2...'
printf '%s\n' 'The model (~4-5 GB) is downloaded to models/ on first launch.'
printf '%s\n\n' 'Press Ctrl+C to stop.'

exec "$VENV_PYTHON" -u "$SCRIPT_DIR/app.py"
