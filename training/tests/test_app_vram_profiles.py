from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class AppVramProfileTests(unittest.TestCase):
    def test_profile_contract_and_unrestricted_fast_mode(self):
        self.assertEqual(set(app.VRAM_PROFILES), {"8gb", "12gb", "16gb", "over_16gb"})

        low = app.VRAM_PROFILES["8gb"]
        self.assertEqual(low["batch_size"], 1)
        self.assertTrue(low["gradient_checkpointing"])
        self.assertEqual(low["max_sample_tokens"], 256)

        fast = app.VRAM_PROFILES["over_16gb"]
        self.assertEqual(fast["batch_size"], 2)
        self.assertFalse(fast["gradient_checkpointing"])
        self.assertEqual(fast["max_sample_tokens"], 0)
        self.assertEqual(fast["max_batch_tokens"], 0)
        self.assertEqual(fast["vram_safety_margin_mb"], 0)
        self.assertEqual(fast["max_audio_seconds"], 0.0)

    def test_portable_ffmpeg_layout_is_detected(self):
        ffmpeg = Path(app._ffmpeg_bin())
        self.assertTrue(ffmpeg.is_file())
        self.assertEqual(ffmpeg.parent, ROOT / "ffmpeg")

    def test_no_drop_segment_fallback_preserves_transcript(self):
        source = "one two three four five six seven"
        segments, audit = app._split_asr_segment_no_drop(
            seg_start=10.0,
            seg_end=31.0,
            seg_text=source,
            words=[],
            max_sec=6.0,
            target_max=5.0,
        )
        self.assertTrue(all(segment["end"] - segment["start"] <= 6.0 for segment in segments))
        self.assertEqual(" ".join(segment["text"] for segment in segments), source)
        self.assertAlmostEqual(audit["planned_duration_sec"], 21.0)
        self.assertTrue(all(segment["_exact_bounds"] for segment in segments))

    def test_named_gradio_training_routes_exist(self):
        demo = app.build_ui()
        config_text = str(demo.config)
        self.assertIn("auto_prepare_lora_dataset", config_text)
        self.assertIn("auto_train_lora", config_text)
        self.assertIn("train_lora", config_text)
        placeholders = {
            component.get("props", {}).get("placeholder")
            for component in demo.config.get("components", [])
        }
        self.assertIn(r"C:\path\to\recording.mp3", placeholders)
        self.assertIn(r"C:\path\to\audio_folder", placeholders)
        for label, value in app.VRAM_PROFILE_DROPDOWN_CHOICES:
            self.assertIsInstance(label, str)
            self.assertIn(label, config_text)
            self.assertIn(value, config_text)

    def test_local_folder_fallback_enters_the_normal_training_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "voice.wav").touch()
            (folder / "transcripts.txt").write_text(
                "voice.wav|Local path transcript\n",
                encoding="utf-8",
            )
            with mock.patch.object(app, "get_training_script", return_value=None):
                updates = list(app.train_lora(
                    "local_path_smoke",
                    None,
                    "",
                    32,
                    32,
                    100,
                    1e-4,
                    "8gb",
                    str(folder),
                ))

        self.assertIn("Local path resolved to 1 audio file", updates[0])
        self.assertIn("training/scripts/train_voxcpm_finetune.py was not found", updates[-1])


if __name__ == "__main__":
    unittest.main()
