"""Deterministic, auditable splitting for oversized training samples.

The helpers in this module deliberately do not know anything about Gradio or
VoxCPM.  A caller supplies word-level timestamps, receives a contiguous split
plan, validates it, and can then materialize the plan as WAV files.

Two invariants are central:

* every instant in the requested source interval belongs to exactly one part;
* every source word (and every character of a word split by a hard time limit)
  belongs to exactly one transcript fragment.

Silence-only parts are retained in the plan and on disk.  They are marked by an
empty ``SplitPart.text`` so a dataset builder can keep them for accounting while
excluding them from a text-conditioned training manifest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_TOLERANCE_SEC = 1e-6


class SplitPlanningError(ValueError):
    """Raised when a safe, lossless split plan cannot be constructed."""


class SplitValidationError(SplitPlanningError):
    """Raised when a split plan violates a duration or transcript invariant."""


class TranscriptPartitionError(SplitPlanningError):
    """Raised when transcript text cannot be assigned without loss."""


@dataclass(frozen=True, slots=True)
class TimedWord:
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TranscriptFragment:
    """A contiguous character range belonging to one source word."""

    word_index: int
    fragment_index: int
    fragment_count: int
    text: str


@dataclass(frozen=True, slots=True)
class SplitPart:
    index: int
    start: float
    end: float
    fragments: tuple[TranscriptFragment, ...] = ()

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        """Render this part's fragments without inserting spaces inside words."""

        rendered: list[str] = []
        previous_word: int | None = None
        for fragment in self.fragments:
            if fragment.word_index == previous_word:
                rendered[-1] += fragment.text
            else:
                rendered.append(fragment.text)
            previous_word = fragment.word_index
        return " ".join(rendered)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class SplitPlan:
    source_start: float
    source_end: float
    max_part_sec: float
    words: tuple[TimedWord, ...]
    parts: tuple[SplitPart, ...]

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def source_transcript(self) -> str:
        return " ".join(word.text for word in self.words)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    part_count: int
    source_duration_sec: float
    planned_duration_sec: float
    duration_delta_sec: float
    max_part_duration_sec: float
    textless_part_count: int
    textless_duration_sec: float
    transcript_preserved: bool


@dataclass(frozen=True, slots=True)
class SlicedAudio:
    part: SplitPart
    path: Path
    start_frame: int
    end_frame: int
    sample_rate: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def _as_timed_word(value: TimedWord | Mapping[str, object]) -> TimedWord:
    if isinstance(value, TimedWord):
        return value
    try:
        return TimedWord(
            text=str(value["text"]),
            start=float(value["start"]),
            end=float(value["end"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitPlanningError(f"Invalid timed word: {value!r}") from exc


def extract_absolute_words_from_timestamped_segment(result: object) -> tuple[TimedWord, ...]:
    """Convert an onnx-asr VAD result to full-file word timestamps.

    ``onnx_asr.vad.BaseVad`` records ``result.start`` / ``result.end`` in the
    full input timeline, but the nested ASR token timestamps are relative to
    the waveform slice passed to the recognizer.  Keeping that distinction here
    prevents the common bug of combining a relative token start with an
    absolute segment end.
    """

    try:
        segment_start = float(getattr(result, "start"))
        segment_end = float(getattr(result, "end"))
    except (TypeError, ValueError) as exc:
        raise SplitPlanningError("Timestamped segment has invalid bounds") from exc
    if not math.isfinite(segment_start) or not math.isfinite(segment_end):
        raise SplitPlanningError("Timestamped segment bounds must be finite")
    if segment_end <= segment_start:
        raise SplitPlanningError("Timestamped segment has non-positive duration")

    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    if tokens is None or timestamps is None or len(tokens) != len(timestamps):
        raise SplitPlanningError("Timestamped segment has no parallel token timestamps")
    if not tokens:
        return ()

    segment_duration = segment_end - segment_start
    relative_times: list[float] = []
    for index, timestamp in enumerate(timestamps):
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise SplitPlanningError(f"Token {index} has an invalid timestamp") from exc
        if not math.isfinite(value):
            raise SplitPlanningError(f"Token {index} has a non-finite timestamp")
        if value < -DEFAULT_TOLERANCE_SEC or value > segment_duration + DEFAULT_TOLERANCE_SEC:
            raise SplitPlanningError(
                f"Relative token timestamp {value} lies outside segment duration {segment_duration}"
            )
        value = min(max(value, 0.0), segment_duration)
        if relative_times and value < relative_times[-1] - DEFAULT_TOLERANCE_SEC:
            raise SplitPlanningError("Token timestamps are not monotonic")
        relative_times.append(value)

    words: list[TimedWord] = []
    current_text = ""
    current_start = 0.0
    current_end = 0.0
    for index, (token, relative_start) in enumerate(zip(tokens, relative_times)):
        token_text = str(token)
        relative_end = (
            relative_times[index + 1]
            if index + 1 < len(relative_times)
            else segment_duration
        )
        starts_word = token_text.startswith(" ") or index == 0
        piece = token_text.lstrip(" ")
        if starts_word:
            if current_text:
                words.append(
                    TimedWord(
                        text=current_text,
                        start=segment_start + current_start,
                        end=segment_start + current_end,
                    )
                )
            current_text = piece
            current_start = relative_start
            current_end = relative_end
        elif current_text:
            current_text += piece
            current_end = relative_end
        else:
            current_text = piece
            current_start = relative_start
            current_end = relative_end

    if current_text:
        words.append(
            TimedWord(
                text=current_text,
                start=segment_start + current_start,
                end=segment_start + current_end,
            )
        )
    return tuple(words)


def normalize_timed_words(
    words: Iterable[TimedWord | Mapping[str, object]],
    *,
    source_start: float,
    source_end: float,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
) -> tuple[TimedWord, ...]:
    """Validate and normalize timestamps while preserving input word order."""

    if not math.isfinite(source_start) or not math.isfinite(source_end):
        raise SplitPlanningError("Source bounds must be finite")
    if source_end <= source_start:
        raise SplitPlanningError("source_end must be greater than source_start")
    if tolerance_sec < 0:
        raise SplitPlanningError("tolerance_sec must be non-negative")

    normalized: list[TimedWord] = []
    previous_end = source_start
    for index, raw_word in enumerate(words):
        word = _as_timed_word(raw_word)
        text = word.text.strip()
        if not text:
            raise SplitPlanningError(f"Word {index} has empty text")
        if not math.isfinite(word.start) or not math.isfinite(word.end):
            raise SplitPlanningError(f"Word {index} has non-finite timestamps")
        if word.end < word.start - tolerance_sec:
            raise SplitPlanningError(f"Word {index} ends before it starts")
        if word.start < source_start - tolerance_sec or word.end > source_end + tolerance_sec:
            raise SplitPlanningError(
                f"Word {index} [{word.start}, {word.end}] lies outside "
                f"source interval [{source_start}, {source_end}]"
            )
        if word.start < previous_end - tolerance_sec:
            raise SplitPlanningError(f"Word {index} overlaps the previous word")

        start = min(max(word.start, source_start), source_end)
        end = min(max(word.end, start), source_end)
        if start < previous_end:
            start = previous_end
            end = max(end, start)
        normalized_word = TimedWord(text=text, start=start, end=end)
        normalized.append(normalized_word)
        previous_end = end

    return tuple(normalized)


def _safe_gap_intervals(
    words: Sequence[TimedWord], source_start: float, source_end: float
) -> tuple[tuple[float, float], ...]:
    if not words:
        return ((source_start, source_end),)

    gaps: list[tuple[float, float]] = []
    if words[0].start > source_start:
        gaps.append((source_start, words[0].start))
    for left, right in zip(words, words[1:]):
        if right.start >= left.end:
            gaps.append((left.end, right.start))
    if words[-1].end < source_end:
        gaps.append((words[-1].end, source_end))
    return tuple(gaps)


def _closest_safe_cut(
    *,
    cursor: float,
    latest: float,
    desired: float,
    gaps: Sequence[tuple[float, float]],
    tolerance_sec: float,
) -> float | None:
    candidates: list[float] = []
    for gap_start, gap_end in gaps:
        lower = max(gap_start, cursor)
        upper = min(gap_end, latest)
        if upper <= cursor + tolerance_sec or lower > upper + tolerance_sec:
            continue
        candidate = min(max(desired, lower), upper)
        if candidate > cursor + tolerance_sec:
            candidates.append(candidate)
    if not candidates:
        return None
    # Prefer the later boundary when two candidates are equally natural.  This
    # minimizes the number of children without changing transcript assignment.
    return min(candidates, key=lambda value: (abs(value - desired), -value))


def _allocate_units(total: int, weights: Sequence[float], *, minimum_one: bool) -> list[int]:
    """Allocate integer units proportionally, deterministically and exactly."""

    count = len(weights)
    if count == 0:
        if total:
            raise TranscriptPartitionError("Cannot allocate units to zero destinations")
        return []
    minimum = 1 if minimum_one else 0
    if total < minimum * count:
        raise TranscriptPartitionError(
            f"Cannot allocate {total} transcript units across {count} non-empty parts"
        )

    allocations = [minimum] * count
    remaining = total - minimum * count
    if remaining == 0:
        return allocations

    clean_weights = [max(0.0, float(weight)) for weight in weights]
    weight_sum = sum(clean_weights)
    if weight_sum <= 0:
        clean_weights = [1.0] * count
        weight_sum = float(count)

    quotas = [remaining * weight / weight_sum for weight in clean_weights]
    floors = [math.floor(quota) for quota in quotas]
    allocations = [base + floor for base, floor in zip(allocations, floors)]
    leftovers = remaining - sum(floors)
    order = sorted(
        range(count),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in order[:leftovers]:
        allocations[index] += 1
    return allocations


def _split_word_text(text: str, overlap_weights: Sequence[float]) -> tuple[str, ...]:
    allocations = _allocate_units(len(text), overlap_weights, minimum_one=True)
    pieces: list[str] = []
    cursor = 0
    for size in allocations:
        pieces.append(text[cursor : cursor + size])
        cursor += size
    if cursor != len(text) or "".join(pieces) != text:
        raise TranscriptPartitionError(f"Failed to partition word {text!r} exactly")
    return tuple(pieces)


def plan_no_drop_splits(
    words: Iterable[TimedWord | Mapping[str, object]],
    *,
    source_start: float,
    source_end: float,
    max_part_sec: float,
    target_part_sec: float | None = None,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
) -> SplitPlan:
    """Create a contiguous time plan and assign every transcript character once.

    Cuts prefer silence between timestamped words.  When the maximum duration
    forces a cut through one unusually long word, that word is divided into
    contiguous character fragments in proportion to the audio overlap.  If the
    timestamp is so pathological that there are more forced audio pieces than
    characters, the function raises ``TranscriptPartitionError`` instead of
    silently dropping or duplicating text.
    """

    if not math.isfinite(max_part_sec) or max_part_sec <= 0:
        raise SplitPlanningError("max_part_sec must be a positive finite number")
    target = max_part_sec if target_part_sec is None else float(target_part_sec)
    if not math.isfinite(target) or target <= 0 or target > max_part_sec:
        raise SplitPlanningError("target_part_sec must be in (0, max_part_sec]")

    normalized_words = normalize_timed_words(
        words,
        source_start=source_start,
        source_end=source_end,
        tolerance_sec=tolerance_sec,
    )
    gaps = _safe_gap_intervals(normalized_words, source_start, source_end)

    boundaries = [float(source_start)]
    cursor = float(source_start)
    while source_end - cursor > max_part_sec + tolerance_sec:
        latest = min(cursor + max_part_sec, source_end)
        desired = min(cursor + target, latest)
        cut = _closest_safe_cut(
            cursor=cursor,
            latest=latest,
            desired=desired,
            gaps=gaps,
            tolerance_sec=tolerance_sec,
        )
        if cut is None:
            cut = latest
        if cut <= cursor + tolerance_sec:
            raise SplitPlanningError("Split planning made no forward progress")
        boundaries.append(cut)
        cursor = cut
    boundaries.append(float(source_end))

    raw_parts = [
        SplitPart(index=index, start=start, end=end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]
    fragments_by_part: list[list[TranscriptFragment]] = [[] for _ in raw_parts]

    for word_index, word in enumerate(normalized_words):
        if word.duration <= tolerance_sec:
            containing = next(
                (
                    index
                    for index, part in enumerate(raw_parts)
                    if part.start - tolerance_sec <= word.start
                    and (
                        word.start < part.end - tolerance_sec
                        or index == len(raw_parts) - 1
                    )
                ),
                None,
            )
            if containing is None:
                raise SplitPlanningError(f"Could not place zero-duration word {word_index}")
            intersecting = [containing]
            weights = [1.0]
        else:
            intersecting = []
            weights = []
            for index, part in enumerate(raw_parts):
                overlap = min(word.end, part.end) - max(word.start, part.start)
                if overlap > tolerance_sec:
                    intersecting.append(index)
                    weights.append(overlap)
            if not intersecting:
                raise SplitPlanningError(f"Timed word {word_index} was not covered by any part")

        pieces = _split_word_text(word.text, weights)
        fragment_count = len(pieces)
        for fragment_index, (part_index, piece) in enumerate(zip(intersecting, pieces)):
            fragments_by_part[part_index].append(
                TranscriptFragment(
                    word_index=word_index,
                    fragment_index=fragment_index,
                    fragment_count=fragment_count,
                    text=piece,
                )
            )

    parts = tuple(
        replace(part, fragments=tuple(fragments_by_part[part.index]))
        for part in raw_parts
    )
    plan = SplitPlan(
        source_start=float(source_start),
        source_end=float(source_end),
        max_part_sec=float(max_part_sec),
        words=normalized_words,
        parts=parts,
    )
    validate_split_plan(plan, tolerance_sec=max(tolerance_sec, DEFAULT_TOLERANCE_SEC))
    return plan


def reconstruct_plan_transcript(plan: SplitPlan) -> str:
    """Reconstruct the source transcript from fragment provenance."""

    fragments_by_word: dict[int, list[TranscriptFragment]] = {}
    for part in plan.parts:
        for fragment in part.fragments:
            fragments_by_word.setdefault(fragment.word_index, []).append(fragment)

    words: list[str] = []
    for word_index in range(len(plan.words)):
        fragments = sorted(
            fragments_by_word.get(word_index, []), key=lambda value: value.fragment_index
        )
        words.append("".join(fragment.text for fragment in fragments))
    return " ".join(words)


def validate_split_plan(
    plan: SplitPlan,
    *,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    allow_textless_parts: bool = True,
) -> SplitAudit:
    """Validate time coverage, maximum duration, and transcript provenance."""

    if not plan.parts:
        raise SplitValidationError("Split plan has no parts")
    if abs(plan.parts[0].start - plan.source_start) > tolerance_sec:
        raise SplitValidationError("First part does not begin at source_start")
    if abs(plan.parts[-1].end - plan.source_end) > tolerance_sec:
        raise SplitValidationError("Last part does not end at source_end")

    for expected_index, part in enumerate(plan.parts):
        if part.index != expected_index:
            raise SplitValidationError("Part indices are not contiguous")
        if part.duration <= 0:
            raise SplitValidationError(f"Part {part.index} has non-positive duration")
        if part.duration > plan.max_part_sec + tolerance_sec:
            raise SplitValidationError(
                f"Part {part.index} exceeds max duration: {part.duration:.6f}s"
            )
        if expected_index:
            previous = plan.parts[expected_index - 1]
            if abs(previous.end - part.start) > tolerance_sec:
                raise SplitValidationError(
                    f"Gap or overlap between parts {previous.index} and {part.index}"
                )

    planned_duration = sum(part.duration for part in plan.parts)
    duration_delta = planned_duration - plan.source_duration
    if abs(duration_delta) > tolerance_sec * max(1, len(plan.parts)):
        raise SplitValidationError(
            f"Duration accounting mismatch: {duration_delta:+.9f}s"
        )

    fragments_by_word: dict[int, list[TranscriptFragment]] = {}
    flattened_order: list[tuple[int, int]] = []
    for part in plan.parts:
        for fragment in part.fragments:
            if fragment.word_index < 0 or fragment.word_index >= len(plan.words):
                raise SplitValidationError("Fragment refers to an unknown source word")
            fragments_by_word.setdefault(fragment.word_index, []).append(fragment)
            flattened_order.append((fragment.word_index, fragment.fragment_index))
    if flattened_order != sorted(flattened_order):
        raise SplitValidationError("Transcript fragments are out of source order")

    for word_index, word in enumerate(plan.words):
        fragments = sorted(
            fragments_by_word.get(word_index, []), key=lambda value: value.fragment_index
        )
        if not fragments:
            raise SplitValidationError(f"Source word {word_index} was dropped")
        expected_count = fragments[0].fragment_count
        if expected_count != len(fragments):
            raise SplitValidationError(f"Source word {word_index} has missing fragments")
        if [fragment.fragment_index for fragment in fragments] != list(range(expected_count)):
            raise SplitValidationError(f"Source word {word_index} has duplicate/out-of-order fragments")
        if any(fragment.fragment_count != expected_count for fragment in fragments):
            raise SplitValidationError(f"Source word {word_index} has inconsistent fragment counts")
        if "".join(fragment.text for fragment in fragments) != word.text:
            raise SplitValidationError(f"Source word {word_index} text was changed")

    transcript_preserved = reconstruct_plan_transcript(plan) == plan.source_transcript
    if not transcript_preserved:
        raise SplitValidationError("Reconstructed transcript differs from source transcript")

    textless_parts = [part for part in plan.parts if not part.has_text]
    if textless_parts and not allow_textless_parts:
        raise SplitValidationError(
            f"Plan contains {len(textless_parts)} silence-only parts"
        )

    return SplitAudit(
        part_count=len(plan.parts),
        source_duration_sec=plan.source_duration,
        planned_duration_sec=planned_duration,
        duration_delta_sec=duration_delta,
        max_part_duration_sec=max(part.duration for part in plan.parts),
        textless_part_count=len(textless_parts),
        textless_duration_sec=sum(part.duration for part in textless_parts),
        transcript_preserved=transcript_preserved,
    )


def partition_transcript(
    source_text: str,
    weights: Sequence[float],
    *,
    require_text_for_positive_weight: bool = True,
) -> tuple[str, ...]:
    """Partition a supplied transcript across parts without dropping a token.

    This is intended for manual datasets: ASR timestamps choose the audio
    boundaries, while the user's supplied transcript remains authoritative.
    Zero-weight (for example silence-only) parts receive an empty transcript.
    """

    tokens = re.findall(r"\S+", source_text or "")
    if not tokens:
        raise TranscriptPartitionError("Source transcript is empty")
    if not weights:
        raise TranscriptPartitionError("No destination weights were supplied")
    if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
        raise TranscriptPartitionError("Transcript weights must be finite and non-negative")

    eligible = [index for index, weight in enumerate(weights) if float(weight) > 0]
    if not eligible:
        raise TranscriptPartitionError("At least one destination must have positive weight")
    allocations = _allocate_units(
        len(tokens),
        [float(weights[index]) for index in eligible],
        minimum_one=require_text_for_positive_weight,
    )

    result = [""] * len(weights)
    cursor = 0
    for index, token_count in zip(eligible, allocations):
        result[index] = " ".join(tokens[cursor : cursor + token_count])
        cursor += token_count
    if cursor != len(tokens):
        raise TranscriptPartitionError("Transcript partition did not consume every token")
    validate_transcript_partition(source_text, result)
    return tuple(result)


def validate_transcript_partition(source_text: str, child_texts: Sequence[str]) -> None:
    """Require child transcripts to reproduce the source whitespace-token sequence."""

    source_tokens = re.findall(r"\S+", source_text or "")
    child_tokens = [token for text in child_texts for token in re.findall(r"\S+", text or "")]
    if child_tokens != source_tokens:
        raise TranscriptPartitionError(
            "Child transcripts do not reproduce the source transcript exactly"
        )


def sample_ranges_for_plan(
    plan: SplitPlan,
    *,
    sample_rate: int,
    total_frames: int,
    source_origin_sec: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    """Convert shared time boundaries once so adjacent slices share exact frames."""

    if sample_rate <= 0:
        raise SplitPlanningError("sample_rate must be positive")
    if total_frames < 0:
        raise SplitPlanningError("total_frames must be non-negative")

    time_boundaries = [plan.parts[0].start] + [part.end for part in plan.parts]
    frame_boundaries = [
        int(round((boundary - source_origin_sec) * sample_rate))
        for boundary in time_boundaries
    ]
    if frame_boundaries[0] < 0 or frame_boundaries[-1] > total_frames:
        raise SplitPlanningError(
            "Split interval falls outside the source audio frame range"
        )
    if any(right <= left for left, right in zip(frame_boundaries, frame_boundaries[1:])):
        raise SplitPlanningError("At least one split is empty at the source sample rate")
    return tuple(zip(frame_boundaries, frame_boundaries[1:]))


def slice_audio_file(
    source_path: str | Path,
    output_dir: str | Path,
    plan: SplitPlan,
    *,
    prefix: str = "clip",
    source_origin_sec: float = 0.0,
    output_subtype: str | None = "PCM_16",
) -> tuple[SlicedAudio, ...]:
    """Materialize every plan part as a WAV using exact shared frame boundaries."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - portable app includes soundfile
        raise RuntimeError("soundfile is required to slice audio") from exc

    source = Path(source_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)

    outputs: list[SlicedAudio] = []
    with sf.SoundFile(str(source), mode="r") as audio:
        ranges = sample_ranges_for_plan(
            plan,
            sample_rate=audio.samplerate,
            total_frames=len(audio),
            source_origin_sec=source_origin_sec,
        )
        for part, (start_frame, end_frame) in zip(plan.parts, ranges):
            audio.seek(start_frame)
            frames = audio.read(end_frame - start_frame, dtype="float32", always_2d=True)
            if len(frames) != end_frame - start_frame:
                raise SplitPlanningError(
                    f"Short read for part {part.index}: expected {end_frame - start_frame}, "
                    f"received {len(frames)} frames"
                )
            output_path = destination / f"{prefix}_{part.index:04d}.wav"
            sf.write(
                str(output_path),
                frames,
                audio.samplerate,
                subtype=output_subtype,
                format="WAV",
            )
            outputs.append(
                SlicedAudio(
                    part=part,
                    path=output_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    sample_rate=audio.samplerate,
                )
            )
    return tuple(outputs)
