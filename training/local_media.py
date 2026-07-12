"""Resolve local media paths pasted into the LoRA training UI.

The helpers in this module deliberately have no Gradio dependency.  They can
therefore be used by both UI callbacks and command-line/tests without loading
the application or any ML models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


SUPPORTED_AUDIO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
)
SUPPORTED_VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
)
SUPPORTED_MEDIA_SUFFIXES: Final[frozenset[str]] = (
    SUPPORTED_AUDIO_SUFFIXES | SUPPORTED_VIDEO_SUFFIXES
)

_SURROUNDING_QUOTES: Final[dict[str, str]] = {
    '"': '"',
    "'": "'",
    "\N{LEFT DOUBLE QUOTATION MARK}": "\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{LEFT SINGLE QUOTATION MARK}": "\N{RIGHT SINGLE QUOTATION MARK}",
}


class LocalMediaError(ValueError):
    """A user-facing validation error for a pasted local media path."""


def _display_suffixes(suffixes: frozenset[str]) -> str:
    return ", ".join(sorted(suffixes))


def _strip_surrounding_quotes(value: str) -> str:
    """Remove one matching pair of shell/Explorer-style surrounding quotes."""

    value = value.strip()
    if len(value) >= 2 and _SURROUNDING_QUOTES.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    return value


def resolve_pasted_path(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return an existing absolute path from text pasted by the user.

    Matching straight or typographic quotes around the value are ignored.
    Environment variables and ``~`` are expanded.  Relative paths are
    resolved against ``base_dir`` when supplied, otherwise against the current
    working directory.
    """

    if raw_path is None:
        raise LocalMediaError("Enter a file or folder path.")

    value = _strip_surrounding_quotes(os.fspath(raw_path))
    if not value:
        raise LocalMediaError("Enter a file or folder path.")

    value = os.path.expandvars(os.path.expanduser(value))
    path = Path(value)
    if not path.is_absolute():
        root = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
        path = root / path

    try:
        path = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise LocalMediaError(f"Could not resolve path: {value}") from exc

    if not path.exists():
        raise LocalMediaError(f"Path does not exist: {path}")
    return path


def _validate_file_suffix(
    path: Path,
    *,
    suffixes: frozenset[str],
    kind: str,
) -> Path:
    if not path.is_file():
        raise LocalMediaError(f"Expected a file, but found: {path}")
    if path.suffix.casefold() not in suffixes:
        suffix = path.suffix or "(no extension)"
        raise LocalMediaError(
            f"Unsupported {kind} file type '{suffix}'. "
            f"Supported types: {_display_suffixes(suffixes)}"
        )
    return path


def resolve_audio_files(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> list[Path]:
    """Resolve one audio file or all supported audio files in one folder.

    Folder contents are not searched recursively.  The returned order is
    deterministic and case-insensitive, which also makes transcript previews
    stable across platforms.
    """

    path = resolve_pasted_path(raw_path, base_dir=base_dir)
    if path.is_file():
        return [
            _validate_file_suffix(
                path,
                suffixes=SUPPORTED_AUDIO_SUFFIXES,
                kind="audio",
            )
        ]
    if not path.is_dir():
        raise LocalMediaError(f"Path is not a file or folder: {path}")

    audio_files = sorted(
        (
            child.resolve(strict=False)
            for child in path.iterdir()
            if child.is_file()
            and child.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
        ),
        key=lambda child: (child.name.casefold(), child.name),
    )
    if not audio_files:
        raise LocalMediaError(f"No supported audio files found in folder: {path}")
    return audio_files


def resolve_single_media_file(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the one media file required by automatic LoRA preparation.

    A direct file path is accepted.  A folder is accepted only when its
    immediate contents contain exactly one supported audio or video file.
    """

    path = resolve_pasted_path(raw_path, base_dir=base_dir)
    if path.is_file():
        return _validate_file_suffix(
            path,
            suffixes=SUPPORTED_MEDIA_SUFFIXES,
            kind="media",
        )
    if not path.is_dir():
        raise LocalMediaError(f"Path is not a file or folder: {path}")

    media_files = sorted(
        (
            child.resolve(strict=False)
            for child in path.iterdir()
            if child.is_file()
            and child.suffix.casefold() in SUPPORTED_MEDIA_SUFFIXES
        ),
        key=lambda child: (child.name.casefold(), child.name),
    )
    if len(media_files) != 1:
        raise LocalMediaError(
            "Automatic preparation requires exactly one supported media file "
            f"in the folder; found {len(media_files)}: {path}"
        )
    return media_files[0]


def find_transcripts_file(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find ``transcripts.txt`` beside a file or directly inside a folder."""

    path = resolve_pasted_path(raw_path, base_dir=base_dir)
    folder = path.parent if path.is_file() else path
    candidate = folder / "transcripts.txt"
    return candidate.resolve(strict=False) if candidate.is_file() else None


def read_transcripts(
    raw_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Read a neighboring ``transcripts.txt`` as UTF-8, or return ``None``.

    ``utf-8-sig`` also accepts files saved with a UTF-8 BOM while leaving
    ordinary UTF-8 unchanged.
    """

    transcript_path = find_transcripts_file(raw_path, base_dir=base_dir)
    if transcript_path is None:
        return None
    try:
        return transcript_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise LocalMediaError(
            f"Could not read transcript file as UTF-8: {transcript_path}"
        ) from exc
