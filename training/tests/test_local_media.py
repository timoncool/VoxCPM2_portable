from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.local_media import (
    LocalMediaError,
    find_transcripts_file,
    read_transcripts,
    resolve_audio_files,
    resolve_pasted_path,
    resolve_single_media_file,
)


class PastedPathTests(unittest.TestCase):
    def test_matching_quotes_and_surrounding_whitespace_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "voice sample.wav"
            media.touch()

            resolved = resolve_pasted_path(f'  "{media}"  ')

            self.assertEqual(resolved, media.resolve())

    def test_typographic_quotes_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "voice.wav"
            media.touch()

            resolved = resolve_pasted_path(f"\N{LEFT DOUBLE QUOTATION MARK}{media}\N{RIGHT DOUBLE QUOTATION MARK}")

            self.assertEqual(resolved, media.resolve())

    def test_relative_path_is_resolved_against_explicit_base_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            media = base / "clips" / "voice.wav"
            media.parent.mkdir()
            media.touch()

            resolved = resolve_pasted_path(
                "clips\\voice.wav",
                base_dir=base,
            )

            self.assertEqual(resolved, media.resolve())

    def test_empty_and_missing_paths_have_english_errors(self):
        with self.assertRaisesRegex(LocalMediaError, "Enter a file or folder path"):
            resolve_pasted_path("  ")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(LocalMediaError, "Path does not exist"):
                resolve_pasted_path("missing.wav", base_dir=temp_dir)


class AudioPathTests(unittest.TestCase):
    def test_direct_supported_audio_file_returns_single_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "VOICE.FLAC"
            audio.touch()

            self.assertEqual(resolve_audio_files(audio), [audio.resolve()])

    def test_folder_returns_only_supported_audio_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            expected_names = ["alpha.MP3", "Beta.wav", "zeta.opus"]
            for name in reversed(expected_names):
                (folder / name).touch()
            (folder / "notes.txt").touch()
            (folder / "movie.mp4").touch()
            (folder / "nested").mkdir()
            (folder / "nested" / "ignored.wav").touch()

            resolved = resolve_audio_files(folder)

            self.assertEqual([path.name for path in resolved], expected_names)

    def test_auto_prepare_dataset_root_uses_audio_subfolder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            audio_folder = dataset / "audio"
            audio_folder.mkdir()
            first = audio_folder / "clip_0001.wav"
            second = audio_folder / "clip_0002.flac"
            first.touch()
            second.touch()
            (dataset / "transcripts.txt").write_text(
                "clip_0001.wav|First\nclip_0002.flac|Second\n",
                encoding="utf-8",
            )

            self.assertEqual(
                resolve_audio_files(dataset),
                [first.resolve(), second.resolve()],
            )

    def test_unsupported_file_and_empty_folder_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            text_file = folder / "notes.txt"
            text_file.touch()

            with self.assertRaisesRegex(LocalMediaError, "Unsupported audio file type"):
                resolve_audio_files(text_file)
            with self.assertRaisesRegex(LocalMediaError, "No supported audio files"):
                resolve_audio_files(folder)


class AutomaticMediaPathTests(unittest.TestCase):
    def test_direct_audio_or_video_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            audio = folder / "voice.m4a"
            video = folder / "recording.MKV"
            audio.touch()
            video.touch()

            self.assertEqual(resolve_single_media_file(audio), audio.resolve())
            self.assertEqual(resolve_single_media_file(video), video.resolve())

    def test_folder_with_exactly_one_media_file_returns_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            media = folder / "recording.webm"
            media.touch()
            (folder / "transcripts.txt").touch()

            self.assertEqual(resolve_single_media_file(folder), media.resolve())

    def test_folder_with_zero_or_multiple_media_files_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "notes.txt").touch()
            with self.assertRaisesRegex(LocalMediaError, "exactly one.*found 0"):
                resolve_single_media_file(folder)

            (folder / "one.wav").touch()
            (folder / "two.mp4").touch()
            with self.assertRaisesRegex(LocalMediaError, "exactly one.*found 2"):
                resolve_single_media_file(folder)

    def test_unsupported_direct_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            unsupported = Path(temp_dir) / "recording.aac"
            unsupported.touch()

            with self.assertRaisesRegex(LocalMediaError, "Unsupported media file type"):
                resolve_single_media_file(unsupported)


class TranscriptTests(unittest.TestCase):
    def test_reads_transcripts_beside_an_audio_file_and_strips_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            audio = folder / "voice.wav"
            audio.touch()
            transcript = folder / "transcripts.txt"
            transcript.write_text(
                "voice.wav|Hello from a local path.\n",
                encoding="utf-8-sig",
            )

            self.assertEqual(find_transcripts_file(audio), transcript.resolve())
            self.assertEqual(
                read_transcripts(audio),
                "voice.wav|Hello from a local path.\n",
            )

    def test_reads_transcripts_inside_selected_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            transcript = folder / "transcripts.txt"
            transcript.write_text("one.wav|First\ntwo.wav|Second", encoding="utf-8")

            self.assertEqual(read_transcripts(folder), "one.wav|First\ntwo.wav|Second")

    def test_audio_subfolder_finds_dataset_root_transcripts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            audio_folder = dataset / "audio"
            audio_folder.mkdir()
            transcript = dataset / "transcripts.txt"
            transcript.write_text("clip.wav|Dataset root\n", encoding="utf-8")

            self.assertEqual(find_transcripts_file(audio_folder), transcript.resolve())
            self.assertEqual(read_transcripts(audio_folder), "clip.wav|Dataset root\n")

    def test_missing_transcripts_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            audio = folder / "voice.wav"
            audio.touch()

            self.assertIsNone(find_transcripts_file(audio))
            self.assertIsNone(read_transcripts(folder))


if __name__ == "__main__":
    unittest.main()
