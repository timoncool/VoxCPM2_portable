"""Shared VRAM-profile and LoRA training input policy.

This module intentionally has no Torch, Gradio, or trainer imports.  The WebUI
and the separate training process can therefore use exactly the same profile
contract without initializing CUDA or loading the model.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


# These are execution-memory presets, not quality presets.  LoRA rank, alpha,
# step count and learning rate are deliberately absent.
VRAM_PROFILES: dict[str, dict[str, Any]] = {
    "8gb": {
        "batch_size": 1,
        "max_audio_seconds": 15.0,
        "max_batch_tokens": 256,
        "max_sample_tokens": 256,
        "gradient_checkpointing": True,
        "split_oversized_samples": True,
        "vram_safety_margin_mb": 512,
    },
    "12gb": {
        "batch_size": 1,
        "max_audio_seconds": 30.0,
        "max_batch_tokens": 384,
        "max_sample_tokens": 384,
        "gradient_checkpointing": True,
        "split_oversized_samples": True,
        "vram_safety_margin_mb": 768,
    },
    "16gb": {
        "batch_size": 1,
        "max_audio_seconds": 45.0,
        "max_batch_tokens": 512,
        "max_sample_tokens": 512,
        "gradient_checkpointing": False,
        "split_oversized_samples": True,
        "vram_safety_margin_mb": 1024,
    },
    "over_16gb": {
        "batch_size": 2,
        "max_audio_seconds": 0.0,
        "max_batch_tokens": 0,
        "max_sample_tokens": 0,
        "gradient_checkpointing": False,
        "split_oversized_samples": False,
        "vram_safety_margin_mb": 0,
    },
}


# Real 8 GB boards commonly report just under 8.0 GiB.  Cards below this
# tolerance are unsupported rather than being misleadingly labelled "8 GB".
MIN_SUPPORTED_VRAM_GIB = 7.5
PROFILE_12GB_THRESHOLD_GIB = 10.0
PROFILE_16GB_THRESHOLD_GIB = 15.0
PROFILE_OVER_16GB_THRESHOLD_GIB = 16.5


_PROFILE_ALIASES = {
    "8": "8gb",
    "8g": "8gb",
    "8gb": "8gb",
    "8_gb": "8gb",
    "12": "12gb",
    "12g": "12gb",
    "12gb": "12gb",
    "12_gb": "12gb",
    "16": "16gb",
    "16g": "16gb",
    "16gb": "16gb",
    "16_gb": "16gb",
    "16gb+": "over_16gb",
    "16_gb+": "over_16gb",
    "16gb_plus": "over_16gb",
    "16_gb_plus": "over_16gb",
    ">16gb": "over_16gb",
    ">16_gb": "over_16gb",
    "over16gb": "over_16gb",
    "over_16gb": "over_16gb",
    "over_16_gb": "over_16gb",
}

_VRAM_AMOUNT_RE = re.compile(
    r"^\s*(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>gib|gb|g|mib|mb)?\s*$",
    re.IGNORECASE,
)


def _validate_profile_table() -> None:
    required = {
        "batch_size",
        "max_audio_seconds",
        "max_batch_tokens",
        "max_sample_tokens",
        "gradient_checkpointing",
        "split_oversized_samples",
        "vram_safety_margin_mb",
    }
    for name, profile in VRAM_PROFILES.items():
        missing = required.difference(profile)
        if missing:
            raise RuntimeError(f"VRAM profile {name!r} is missing: {sorted(missing)}")
        if int(profile["batch_size"]) < 1:
            raise RuntimeError(f"VRAM profile {name!r} has an invalid batch size")
        for key in (
            "max_audio_seconds",
            "max_batch_tokens",
            "max_sample_tokens",
            "vram_safety_margin_mb",
        ):
            if float(profile[key]) < 0:
                raise RuntimeError(f"VRAM profile {name!r} has negative {key}")
        sample_cap = int(profile["max_sample_tokens"])
        batch_cap = int(profile["max_batch_tokens"])
        batch_size = int(profile["batch_size"])
        if sample_cap and batch_cap < sample_cap * batch_size:
            raise RuntimeError(
                f"VRAM profile {name!r} has max_batch_tokens below its per-sample cap"
            )


_validate_profile_table()


def parse_vram_gib(value: Any) -> float:
    """Parse a manual VRAM amount expressed in GiB/GB or MiB/MB.

    A bare number is interpreted as GiB.  Boolean values, infinities, NaN,
    negative values and descriptive labels are rejected.
    """
    if isinstance(value, bool):
        raise ValueError("VRAM amount must be a number, not a boolean")

    unit = "gib"
    if isinstance(value, (int, float)):
        amount = float(value)
    elif isinstance(value, str):
        match = _VRAM_AMOUNT_RE.fullmatch(value)
        if not match:
            raise ValueError(
                f"Invalid VRAM amount {value!r}; use for example 8, 12 GB, or 16384 MB"
            )
        amount = float(match.group("amount").replace(",", "."))
        unit = (match.group("unit") or "gib").lower()
    else:
        raise ValueError(f"Invalid VRAM amount type: {type(value).__name__}")

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("VRAM amount must be a finite number greater than zero")
    if unit in {"mib", "mb"}:
        amount /= 1024.0
    return amount


def profile_for_vram_gib(total_vram_gib: Any) -> str:
    """Map physical/manual VRAM to the conservative execution profile."""
    total = parse_vram_gib(total_vram_gib)
    if total < MIN_SUPPORTED_VRAM_GIB:
        raise ValueError(
            f"VoxCPM2 LoRA training requires about 8 GB VRAM; received {total:.2f} GiB"
        )
    if total < PROFILE_12GB_THRESHOLD_GIB:
        return "8gb"
    if total < PROFILE_16GB_THRESHOLD_GIB:
        return "12gb"
    if total <= PROFILE_OVER_16GB_THRESHOLD_GIB:
        return "16gb"
    return "over_16gb"


def normalize_vram_profile(value: Any, *, allow_empty: bool = False) -> str:
    """Return one canonical profile name from a profile alias or VRAM amount."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_empty:
            return ""
        raise ValueError("A VRAM profile or VRAM amount is required")

    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "").replace("-", "_")
        alias = _PROFILE_ALIASES.get(normalized)
        if alias:
            return alias

    # Numeric strings such as "10.5" and explicit amounts such as "12288 MB"
    # select a profile using the same thresholds as automatic CUDA detection.
    try:
        return profile_for_vram_gib(value)
    except ValueError as exc:
        supported = ", ".join(VRAM_PROFILES)
        raise ValueError(
            f"Unknown VRAM profile/amount {value!r}. Supported profiles: {supported}"
        ) from exc


def resolve_vram_profile(value: Any) -> tuple[str, dict[str, Any]]:
    """Resolve a selection and return an isolated copy of its settings."""
    profile_name = normalize_vram_profile(value)
    return profile_name, dict(VRAM_PROFILES[profile_name])


def load_vram_preference(path: str | os.PathLike[str], fallback: Any) -> str:
    """Load a saved canonical profile, safely falling back on corruption."""
    fallback_name = normalize_vram_profile(fallback)
    preference_path = Path(path)
    try:
        payload = json.loads(preference_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return fallback_name
        return normalize_vram_profile(payload.get("vram_profile"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback_name


def save_vram_preference(path: str | os.PathLike[str], selection: Any) -> str:
    """Atomically persist a manual profile/amount and return its canonical name."""
    profile_name = normalize_vram_profile(selection)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "vram_profile": profile_name,
    }
    try:
        # Keep the entered capacity for diagnostics when selection was numeric.
        if not (isinstance(selection, str) and selection.strip().lower() in _PROFILE_ALIASES):
            payload["manual_vram_gib"] = round(parse_vram_gib(selection), 3)
    except ValueError:
        pass

    preference_path = Path(path)
    preference_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=preference_path.parent,
            prefix=f".{preference_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, preference_path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return profile_name


def validate_lora_hyperparameters(
    rank: Any,
    alpha: Any,
    steps: Any,
    learning_rate: Any,
) -> tuple[int, int, int, float]:
    """Validate values before preprocessing, downloads, or checkpoint deletion."""

    def positive_integer(name: str, value: Any, maximum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        result = int(numeric)
        if result < 1 or result > maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
        return result

    validated_rank = positive_integer("LoRA rank", rank, 1024)
    validated_alpha = positive_integer("LoRA alpha", alpha, 8192)
    validated_steps = positive_integer("Training steps", steps, 10_000_000)
    try:
        validated_lr = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Learning rate must be a finite number greater than zero") from exc
    if not math.isfinite(validated_lr) or not 0 < validated_lr <= 1.0:
        raise ValueError("Learning rate must be greater than zero and at most 1.0")
    return validated_rank, validated_alpha, validated_steps, validated_lr


def build_training_command(
    python_executable: str | os.PathLike[str],
    train_script: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
) -> list[str]:
    """Build the exact unbuffered trainer command used by the WebUI."""
    return [
        os.fspath(python_executable),
        "-u",
        os.fspath(train_script),
        "--config_path",
        os.fspath(config_path),
    ]
