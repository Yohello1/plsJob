from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pls_compression.evaluation import evaluate_checkpoint
from pls_compression.models import CompressionModel, save_checkpoint
from pls_compression.schema import ModelConfig

from .fixtures import write_session


class EvaluationTests(unittest.TestCase):
    def test_two_step_rollout_uses_saved_normalization_without_mutating_data(self):
        for variant in ("density", "density_velocity"):
            with self.subTest(variant=variant):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    data = root / "data"
                    data.mkdir()
                    session = write_session(data, "session", width=8, height=8, frames=5, variant=variant)
                    config = ModelConfig(
                        width=8,
                        height=8,
                        model_variant=variant,
                        latent_dim=8,
                        base_channels=2,
                        bottleneck_channels=4,
                        context_channels=2,
                    )
                    model = CompressionModel(config)
                    checkpoint = root / "model.pth"
                    save_checkpoint(
                        checkpoint,
                        model,
                        normalization={"density_scale": 2.0, "velocity_scale": 3.0},
                        schema=config.schema,
                    )
                    before = (session / "sim_data.bin").read_bytes()
                    result = evaluate_checkpoint(checkpoint, data, steps=2, skip=1, device="cpu")
                    after = (session / "sim_data.bin").read_bytes()
                    self.assertEqual(before, after)
                    self.assertEqual(result.variant, variant)
                    self.assertEqual(result.normalization.density_scale, 2.0)
                    self.assertEqual(result.normalization.velocity_scale, 3.0)
                    self.assertEqual(len(result.steps), 2)
                    self.assertEqual(tuple(result.targets[0].shape), (1, 1, 8, 8) if variant == "density" else (1, 3, 8, 8))
                    self.assertEqual(tuple(result.targets[1].shape), tuple(result.targets[0].shape))
                    self.assertTrue(torch.isfinite(result.predictions[1]).all())


if __name__ == "__main__":
    unittest.main()
