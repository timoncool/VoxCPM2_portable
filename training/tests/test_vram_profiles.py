from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import vram_profiles as profiles


class VramProfilePolicyTests(unittest.TestCase):
    def test_exact_profile_boundaries(self):
        cases = (
            (7.5, "8gb"),
            (9.999, "8gb"),
            (10.0, "12gb"),
            (14.999, "12gb"),
            (15.0, "16gb"),
            (16.5, "16gb"),
            (16.501, "over_16gb"),
        )
        for amount, expected in cases:
            with self.subTest(amount=amount):
                self.assertEqual(profiles.profile_for_vram_gib(amount), expected)

    def test_manual_amount_units_and_profile_aliases(self):
        cases = {
            "8188 MB": "8gb",
            "10,0 GB": "12gb",
            "12288 MiB": "12gb",
            "16 GB": "16gb",
            "17": "over_16gb",
            "16gb+": "over_16gb",
            ">16GB": "over_16gb",
            "over-16gb": "over_16gb",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(profiles.normalize_vram_profile(value), expected)

    def test_invalid_manual_values_are_rejected(self):
        for value in (None, "", "banana", 0, -1, 7.49, math.nan, math.inf, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    profiles.normalize_vram_profile(value)

    def test_resolved_profile_is_a_private_copy(self):
        name, resolved = profiles.resolve_vram_profile("12 GB")
        self.assertEqual(name, "12gb")
        resolved["batch_size"] = 999
        self.assertEqual(profiles.VRAM_PROFILES["12gb"]["batch_size"], 1)

    def test_preference_round_trip_is_atomic_and_corruption_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "vram.json"
            selected = profiles.save_vram_preference(path, "12.5 GB")
            self.assertEqual(selected, "12gb")
            self.assertEqual(profiles.load_vram_preference(path, "8gb"), "12gb")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["manual_vram_gib"], 12.5)
            self.assertFalse(list(path.parent.glob("*.tmp")))

            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(profiles.load_vram_preference(path, "16gb"), "16gb")

    def test_lora_hyperparameters_reject_fractional_nonfinite_and_zero(self):
        self.assertEqual(
            profiles.validate_lora_hyperparameters("32", 64.0, "1000", "0.0001"),
            (32, 64, 1000, 0.0001),
        )
        invalid = (
            (0, 32, 1000, 1e-4),
            (32.5, 32, 1000, 1e-4),
            (32, -1, 1000, 1e-4),
            (32, 32, 0, 1e-4),
            (32, 32, 1000, 0),
            (32, 32, 1000, math.nan),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    profiles.validate_lora_hyperparameters(*values)

    def test_training_command_preserves_paths_as_distinct_arguments(self):
        command = profiles.build_training_command(
            Path("portable python/python.exe"),
            Path("training scripts/train.py"),
            Path("data with spaces/train.yaml"),
        )
        self.assertEqual(
            command,
            [
                "portable python\\python.exe",
                "-u",
                "training scripts\\train.py",
                "--config_path",
                "data with spaces\\train.yaml",
            ],
        )


if __name__ == "__main__":
    unittest.main()
