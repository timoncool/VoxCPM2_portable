from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

# The portable embedded Python uses an isolated ``._pth`` configuration and
# therefore does not automatically add the working directory to ``sys.path``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.audio_splitting import (
    SplitValidationError,
    TimedWord,
    TranscriptPartitionError,
    extract_absolute_words_from_timestamped_segment,
    partition_transcript,
    plan_no_drop_splits,
    reconstruct_plan_transcript,
    sample_ranges_for_plan,
    slice_audio_file,
    validate_split_plan,
    validate_transcript_partition,
)


class NoDropSplitPlanTests(unittest.TestCase):
    def test_onnx_vad_relative_token_times_become_absolute(self):
        result = SimpleNamespace(
            start=100.0,
            end=110.0,
            tokens=[" Hel", "lo", " world"],
            timestamps=[1.0, 2.0, 5.0],
        )
        words = extract_absolute_words_from_timestamped_segment(result)

        self.assertEqual([word.text for word in words], ["Hello", "world"])
        self.assertEqual(
            [(word.start, word.end) for word in words],
            [(101.0, 105.0), (105.0, 110.0)],
        )

    def test_natural_cuts_cover_leading_and_trailing_audio(self):
        words = [
            TimedWord("one", 1.0, 2.0),
            TimedWord("two", 2.4, 3.2),
            TimedWord("three", 6.0, 7.0),
            TimedWord("four", 9.0, 10.0),
        ]
        plan = plan_no_drop_splits(
            words,
            source_start=0.0,
            source_end=12.0,
            max_part_sec=5.0,
            target_part_sec=4.0,
        )

        audit = validate_split_plan(plan)
        self.assertAlmostEqual(plan.parts[0].start, 0.0)
        self.assertAlmostEqual(plan.parts[-1].end, 12.0)
        self.assertTrue(all(part.duration <= 5.0 for part in plan.parts))
        self.assertAlmostEqual(sum(part.duration for part in plan.parts), 12.0)
        self.assertEqual(reconstruct_plan_transcript(plan), "one two three four")
        self.assertTrue(audit.transcript_preserved)

    def test_single_long_word_is_fragmented_without_text_loss(self):
        plan = plan_no_drop_splits(
            [TimedWord("abcdefghij", 1.0, 21.0)],
            source_start=0.0,
            source_end=22.0,
            max_part_sec=5.0,
        )

        self.assertTrue(all(part.duration <= 5.0 for part in plan.parts))
        self.assertEqual(reconstruct_plan_transcript(plan), "abcdefghij")
        fragments = [fragment.text for part in plan.parts for fragment in part.fragments]
        self.assertEqual("".join(fragments), "abcdefghij")
        self.assertEqual(len(fragments), 4)

    def test_pathological_long_one_character_word_fails_instead_of_dropping(self):
        with self.assertRaises(TranscriptPartitionError):
            plan_no_drop_splits(
                [TimedWord("a", 0.0, 12.0)],
                source_start=0.0,
                source_end=12.0,
                max_part_sec=4.0,
            )

    def test_long_leading_and_trailing_gaps_remain_accounted_for(self):
        plan = plan_no_drop_splits(
            [TimedWord("voice", 6.0, 7.0)],
            source_start=0.0,
            source_end=14.0,
            max_part_sec=4.0,
        )

        audit = validate_split_plan(plan)
        self.assertGreater(audit.textless_part_count, 0)
        self.assertGreater(audit.textless_duration_sec, 0.0)
        self.assertAlmostEqual(audit.planned_duration_sec, 14.0)
        with self.assertRaises(SplitValidationError):
            validate_split_plan(plan, allow_textless_parts=False)

    def test_zero_duration_word_is_assigned_once(self):
        plan = plan_no_drop_splits(
            [TimedWord("marker", 2.0, 2.0)],
            source_start=0.0,
            source_end=4.0,
            max_part_sec=2.0,
        )
        fragments = [fragment for part in plan.parts for fragment in part.fragments]
        self.assertEqual(len(fragments), 1)
        self.assertEqual(reconstruct_plan_transcript(plan), "marker")


class TranscriptPartitionTests(unittest.TestCase):
    def test_manual_transcript_partition_preserves_every_token(self):
        source = "This is the exact user supplied transcript."
        children = partition_transcript(source, [2.0, 3.0, 1.0])
        validate_transcript_partition(source, children)
        self.assertEqual(
            [token for child in children for token in child.split()],
            source.split(),
        )
        self.assertTrue(all(children))

    def test_silence_weight_gets_no_text(self):
        children = partition_transcript("one two three", [0.0, 2.0, 1.0])
        self.assertEqual(children[0], "")
        validate_transcript_partition("one two three", children)

    def test_too_few_tokens_is_explicit(self):
        with self.assertRaises(TranscriptPartitionError):
            partition_transcript("one", [1.0, 1.0])


class AudioSlicingTests(unittest.TestCase):
    def test_shared_frame_boundaries_and_written_files_preserve_frames(self):
        sample_rate = 1000
        frames = np.linspace(-0.25, 0.25, 2300, dtype=np.float32)
        plan = plan_no_drop_splits(
            [TimedWord("alpha", 0.2, 1.1), TimedWord("beta", 1.3, 2.0)],
            source_start=0.0,
            source_end=2.3,
            max_part_sec=0.8,
            target_part_sec=0.7,
        )

        ranges = sample_ranges_for_plan(
            plan, sample_rate=sample_rate, total_frames=len(frames)
        )
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], len(frames))
        self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))
        self.assertEqual(sum(end - start for start, end in ranges), len(frames))

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.wav"
            output_dir = Path(temp_dir) / "parts"
            sf.write(source_path, frames, sample_rate, subtype="FLOAT")
            outputs = slice_audio_file(source_path, output_dir, plan, prefix="sample")

            self.assertEqual(len(outputs), len(plan.parts))
            self.assertEqual(sum(output.frames for output in outputs), len(frames))
            for output in outputs:
                self.assertTrue(output.path.is_file())
                self.assertEqual(sf.info(output.path).frames, output.frames)


if __name__ == "__main__":
    unittest.main()
