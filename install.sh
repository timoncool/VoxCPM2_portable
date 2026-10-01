#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
DOWNLOADS_DIR="$SCRIPT_DIR/downloads"
VOICE_PACK_URL="https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip"

die() {
    printf '\nERROR: %s\n' "$1" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

printf '%s\n' '========================================'
printf '%s\n' '  VoxCPM2 Portable - Unix installer'
printf '%s\n\n' '========================================'

case "$(uname -s)" in
    Darwin)
        PLATFORM_NAME="macOS"
        ;;
    Linux)
        PLATFORM_NAME="Linux"
        ;;
    *)
        die "This installer supports macOS and Linux."
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
        if command_exists "$candidate"; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if ! command_exists "$PYTHON_BIN"; then
    die "Python 3.10-3.13 is required. Install Python 3.12 and run this script again."
fi

PYTHON_OK="$($PYTHON_BIN -c 'import sys; print(int((3, 10) <= sys.version_info < (3, 14)))')"
if [[ "$PYTHON_OK" != "1" ]]; then
    die "Python 3.10-3.13 is required for the pinned PyTorch build. Found: $($PYTHON_BIN --version 2>&1)"
fi

printf 'Platform: %s\n' "$PLATFORM_NAME"
printf 'Python: %s\n\n' "$($PYTHON_BIN --version 2>&1)"

mkdir -p "$DOWNLOADS_DIR" "$SCRIPT_DIR/temp" "$SCRIPT_DIR/models" \
    "$SCRIPT_DIR/cache" "$SCRIPT_DIR/output" "$SCRIPT_DIR/voices" \
    "$SCRIPT_DIR/lora" "$SCRIPT_DIR/train_data"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    printf '%s\n' '[1/4] Creating local Python virtual environment...'
    if command_exists uv; then
        uv venv --clear --seed --python "$PYTHON_BIN" "$VENV_DIR" || die "Could not create .venv with uv."
    else
        "$PYTHON_BIN" -m venv "$VENV_DIR" || die "Could not create .venv. Install the Python venv module and try again."
    fi
elif [[ ! -x "$VENV_DIR/bin/pip" ]]; then
    printf '%s\n' '[1/4] Repairing local Python virtual environment...'
    if command_exists uv; then
        uv venv --clear --seed --python "$PYTHON_BIN" "$VENV_DIR" || die "Could not repair .venv with uv."
    else
        "$PYTHON_BIN" -m venv --clear "$VENV_DIR" || die "Could not repair .venv. Install the Python venv module and try again."
    fi
else
    printf '%s\n' '[1/4] Reusing .venv...'
fi

VENV_PYTHON="$VENV_DIR/bin/python"
printf '%s\n' '[2/4] Updating pip tooling...'
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

printf '%s\n' '[3/4] Installing PyTorch and VoxCPM2 dependencies...'
"$VENV_PYTHON" -m pip install "torch==2.7.1" "torchaudio==2.7.1"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        printf '%s\n' 'Apple Silicon detected: PyTorch will use MPS when available.'
    else
        printf '%s\n' 'Intel Mac detected: PyTorch will use CPU.'
    fi
fi

if ! command_exists ffmpeg; then
    if command_exists brew; then
        printf '%s\n' 'FFmpeg is missing. Installing it with Homebrew...'
        if ! brew install ffmpeg; then
            printf '%s\n' 'WARNING: Homebrew could not install FFmpeg. Install it manually before using video/audio preparation.' >&2
        fi
    else
        printf '%s\n' 'WARNING: FFmpeg is missing. Install it with Homebrew (`brew install ffmpeg`) before using video/audio preparation.' >&2
    fi
else
    printf 'FFmpeg: %s\n' "$(command -v ffmpeg)"
fi

if [[ "${VOXCPM_SKIP_VOICES:-0}" != "1" ]] && [[ -z "$(find "$SCRIPT_DIR/voices" -type f \( -iname '*.mp3' -o -iname '*.wav' \) -print -quit)" ]]; then
    printf '%s\n' '[4/4] Downloading the default voice pack...'
    VOICE_EXTRACT_DIR="$DOWNLOADS_DIR/voice-pack-extract"
    rm -rf "$VOICE_EXTRACT_DIR"
    mkdir -p "$VOICE_EXTRACT_DIR"
    if command_exists curl && curl -fL --retry 3 --output "$DOWNLOADS_DIR/voice-pack.zip" "$VOICE_PACK_URL" \
        && command_exists unzip \
        && unzip -q -o "$DOWNLOADS_DIR/voice-pack.zip" -d "$VOICE_EXTRACT_DIR"; then
        find "$VOICE_EXTRACT_DIR" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.ogg' -o -iname '*.txt' -o -iname '*.lab' \) \
            -exec cp {} "$SCRIPT_DIR/voices/" \;
        printf '%s\n' 'Voice pack installed.'
        rm -rf "$VOICE_EXTRACT_DIR"
    else
        printf '%s\n' 'WARNING: Default voice pack could not be downloaded. It can be downloaded later from the UI.' >&2
    fi
else
    printf '%s\n' '[4/4] Voice pack already exists or was skipped.'
fi

printf 'macOS/Linux installation complete. Run: ./run.sh\n'
