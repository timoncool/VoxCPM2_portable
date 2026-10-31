import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_voxcpm_finetune.py"
SPEC = importlib.util.spec_from_file_location("train_voxcpm_finetune", SCRIPT_PATH)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


def _tiny_minicpm_config():
    from voxcpm.modules.minicpm4.config import MiniCPM4Config, RopeScalingConfig

    rope = RopeScalingConfig(
        type="longrope",
        long_factor=[1.0, 1.0],
        short_factor=[1.0, 1.0],
        original_max_position_embeddings=16,
    )
    return MiniCPM4Config(
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-5,
        rope_scaling=rope,
        vocab_size=0,
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=8,
        scale_depth=1.0,
        rope_theta=10_000.0,
        kv_channels=4,
        no_rope=True,
    )


class TrainingMemoryTests(unittest.TestCase):
    def test_normalize_vram_profile(self):
        cases = {
            "8 GB": "8gb",
            "12gb": "12gb",
            "16gb": "16gb",
            "16gb_plus": "over_16gb",
            ">16GB": "over_16gb",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(TRAINING.normalize_vram_profile(value), expected)

    def test_over_16gb_explicit_zero_disables_limits_and_safety_gate(self):
        policy = TRAINING._resolve_memory_policy(
            profile="over_16gb",
            batch_size=2,
            max_batch_tokens=0,
            gradient_checkpointing=False,
            max_sample_tokens=0,
            vram_safety_margin_mb=0,
        )

        self.assertEqual(policy["max_sample_tokens"], 0)
        self.assertEqual(policy["effective_max_sample_tokens"], 0)
        self.assertEqual(policy["max_batch_tokens"], 0)
        self.assertEqual(policy["vram_safety_margin_mb"], 0)
        self.assertFalse(policy["gradient_checkpointing"])

    def test_profile_defaults_apply_when_optional_values_are_omitted(self):
        policy = TRAINING._resolve_memory_policy(
            profile="8gb",
            batch_size=1,
            max_batch_tokens=0,
            gradient_checkpointing=None,
            max_sample_tokens=-1,
            vram_safety_margin_mb=-1,
            split_oversized_samples=None,
        )
        self.assertEqual(policy["max_sample_tokens"], 256)
        self.assertEqual(policy["effective_max_sample_tokens"], 256)
        self.assertEqual(policy["max_batch_tokens"], 256)
        self.assertEqual(policy["vram_safety_margin_mb"], 512)
        self.assertTrue(policy["gradient_checkpointing"])
        self.assertTrue(policy["split_oversized_samples"])

    def test_invalid_memory_policy_values_are_rejected_early(self):
        base = {
            "profile": "8gb",
            "batch_size": 1,
            "max_batch_tokens": 256,
            "gradient_checkpointing": True,
            "max_sample_tokens": 256,
            "vram_safety_margin_mb": 512,
        }
        invalid_overrides = (
            {"batch_size": 0},
            {"max_batch_tokens": -1},
            {"max_sample_tokens": -2},
            {"vram_safety_margin_mb": -2},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                values = {**base, **override}
                with self.assertRaises(ValueError):
                    TRAINING._resolve_memory_policy(**values)

    def test_training_forward_uses_checkpointing_without_returning_kv_tensors(self):
        from voxcpm.modules.minicpm4 import MiniCPMModel

        TRAINING._install_memory_efficient_minicpm_forward()
        model = MiniCPMModel(_tiny_minicpm_config()).train()
        model.gradient_checkpointing_enable()
        inputs = torch.randn(2, 5, 8, requires_grad=True)

        output, caches = model(inputs, is_causal=True)
        output.square().mean().backward()

        self.assertEqual(caches, [])
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        model.eval()
        with torch.no_grad():
            _, inference_caches = model(inputs.detach(), is_causal=True)
        self.assertEqual(len(inference_caches), len(model.layers))

    def test_constructor_cache_suppression_is_temporary(self):
        from voxcpm.modules.minicpm4 import MiniCPMModel

        model = MiniCPMModel(_tiny_minicpm_config())
        original = MiniCPMModel.setup_cache
        with TRAINING._suppress_constructor_kv_caches():
            model.setup_cache(1, 8, "cpu", torch.float32)
            self.assertIsNone(model.kv_cache)
        self.assertIs(MiniCPMModel.setup_cache, original)

    def test_training_dtype_cast_keeps_only_lora_parameters_fp32(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(8, 8)
                self.backbone.weight.requires_grad = False
                self.backbone.bias.requires_grad = False
                self.lora_A = torch.nn.Parameter(torch.randn(2, 8))
                self.audio_vae = torch.nn.Linear(4, 4)

            @staticmethod
            def _dtype():
                return torch.bfloat16

        model = DummyModel()
        TRAINING._prepare_training_dtypes(model, keep_lora_fp32=True)

        self.assertEqual(model.backbone.weight.dtype, torch.bfloat16)
        self.assertEqual(model.lora_A.dtype, torch.float32)
        self.assertEqual(model.audio_vae.weight.dtype, torch.float32)

    def test_oversized_samples_are_rejected_instead_of_dropped(self):
        with self.assertRaisesRegex(ValueError, "refuses to drop or silently truncate"):
            TRAINING._raise_oversized_samples([100, 300], 256, True, "train.jsonl")


if __name__ == "__main__":
    unittest.main()
