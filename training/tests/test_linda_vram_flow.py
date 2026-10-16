from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


LINDA = ROOT / "LINDA.mp3"


class _ProgressRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, value, **kwargs):
        self.calls.append((value, kwargs))


class LindaVramFlowTests(unittest.TestCase):
    def test_auto_train_forwards_profile_to_config_and_subprocess(self):
        progress = _ProgressRecorder()
        fake_process = mock.Mock()
        fake_process.stdout = ["iter 1 / 1\n"]
        fake_process.returncode = 0
        fake_process.wait.return_value = 0

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            train_data = temporary / "train_data"
            lora_dir = temporary / "lora"
            manifest = train_data / "linda_flow" / "train.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                '{"audio": "LINDA.mp3", "text": "reference"}\n',
                encoding="utf-8",
            )
            train_script = temporary / "train_voxcpm_finetune.py"
            train_script.write_text("# command target only\n", encoding="utf-8")
            pretrained = temporary / "VoxCPM2"
            pretrained.mkdir()

            def prepared(name, files, transcripts, max_audio_seconds):
                self.assertEqual(name, "linda_flow")
                self.assertEqual([Path(item) for item in files], [LINDA])
                self.assertIn("LINDA.mp3|", transcripts)
                self.assertEqual(max_audio_seconds, 30.0)
                return manifest, 23

            with (
                mock.patch.object(app, "TRAIN_DATA_DIR", train_data),
                mock.patch.object(app, "LORA_DIR", lora_dir),
                mock.patch.object(app, "prepare_train_data", side_effect=prepared),
                mock.patch.object(app, "get_training_script", return_value=train_script),
                mock.patch.object(app, "release_training_gpu_memory", return_value="test models freed"),
                mock.patch.object(app, "persist_vram_profile", return_value="12gb") as persisted,
                mock.patch("huggingface_hub.snapshot_download", return_value=str(pretrained)),
                mock.patch("subprocess.Popen", return_value=fake_process) as popen,
            ):
                output = list(
                    app.maybe_auto_train(
                        True,
                        "linda flow",
                        [str(LINDA)],
                        "LINDA.mp3|Linda reference transcript",
                        32,
                        32,
                        100,
                        1e-4,
                        "12 GB",
                        progress=progress,
                    )
                )

            config_path = train_data / "linda_flow" / "train_config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["vram_profile"], "12gb")
            self.assertEqual(config["batch_size"], 1)
            self.assertEqual(config["max_sample_tokens"], 384)
            self.assertEqual(config["max_batch_tokens"], 384)
            self.assertTrue(config["gradient_checkpointing"])
            self.assertTrue(config["split_oversized_samples"])
            self.assertEqual(config["vram_safety_margin_mb"], 768)
            self.assertEqual(config["lora"]["r"], 32)
            self.assertEqual(config["lora"]["alpha"], 32)
            self.assertEqual(config["num_iters"], 100)
            persisted.assert_called_once_with("12gb")

            command = popen.call_args.args[0]
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-u",
                    str(train_script),
                    "--config_path",
                    str(config_path),
                ],
            )
            self.assertEqual(popen.call_args.kwargs["cwd"], str(app.TRAINING_DIR))
            self.assertTrue(any("VRAM profile: 12gb" in line for line in output))

    def test_config_builder_preserves_effective_batch_across_profiles(self):
        common = {
            "pretrained": "model",
            "manifest": "LINDA.mp3",
            "save_path": "lora/linda",
            "sample_count": 100,
            "r": 64,
            "alpha": 64,
            "steps": 250,
            "lr": 5e-5,
        }
        low = app.build_lora_training_config(**common, vram_profile="8gb")
        fast = app.build_lora_training_config(**common, vram_profile="17 GB")
        self.assertEqual(low["batch_size"] * low["grad_accum_steps"], 4)
        self.assertEqual(fast["batch_size"] * fast["grad_accum_steps"], 4)
        self.assertEqual((fast["lora"]["r"], fast["lora"]["alpha"]), (64, 64))
        self.assertEqual(fast["max_sample_tokens"], 0)
        self.assertFalse(fast["gradient_checkpointing"])


if __name__ == "__main__":
    unittest.main()
