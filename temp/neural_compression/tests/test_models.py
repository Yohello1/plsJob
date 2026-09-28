from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pls_compression.models import CompressionModel, ModelConfig, build_model, load_checkpoint, load_model_checkpoint, save_checkpoint


class ModelTests(unittest.TestCase):
    def make_config(self, variant):
        return ModelConfig(
            width=16,
            height=16,
            model_variant=variant,
            latent_dim=16,
            base_channels=4,
            bottleneck_channels=8,
            context_channels=4,
        )

    def test_shapes_ranges_and_nonsquare_shape(self):
        for variant, channels in (("density", 1), ("density_velocity", 3)):
            config = self.make_config(variant).replace(width=20, height=12)
            model = CompressionModel(config)
            values = (
                torch.rand(2, 1, 12, 20),
                torch.rand(2, 2, 12, 20),
                torch.rand(2, 1, 12, 20),
                torch.rand(2, 2, 12, 20),
                torch.zeros(2, 1, 12, 20),
            )
            output = model(*values)
            self.assertEqual(tuple(output.shape), (2, channels, 12, 20))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output[:, 0:1] >= 0).all())
            self.assertTrue((output[:, 0:1] <= 1).all())
            if channels == 3:
                self.assertTrue((output[:, 1:3] >= -1).all())
                self.assertTrue((output[:, 1:3] <= 1).all())

    def test_checkpoint_contains_config_and_normalization_for_both_variants(self):
        for variant in ("density", "density_velocity"):
            with self.subTest(variant=variant):
                config = self.make_config(variant)
                model = CompressionModel(config)
                normalization = {"density_scale": 2.0, "velocity_scale": 3.0}
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pth"
                    save_checkpoint(path, model, normalization=normalization, schema=config.schema)
                    payload = load_checkpoint(path)
                    self.assertIn("state_dict", payload)
                    self.assertEqual(payload["model_config"]["model_variant"], variant)
                    self.assertEqual(payload["normalization"]["density_scale"], 2.0)
                    loaded = load_model_checkpoint(path, device="cpu")
                    self.assertEqual(loaded.config, config)
                    self.assertEqual(loaded.normalization.density_scale, 2.0)
                    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
                        self.assertTrue(torch.equal(expected, actual))

    def test_production_models_construct_without_forward(self):
        for variant in ("density", "density_velocity"):
            with self.subTest(variant=variant):
                model = build_model(variant)
                self.assertEqual(model.config.width, 400)
                self.assertEqual(model.config.height, 400)
                self.assertEqual(model.model_variant, variant)
                self.assertEqual(model.output_channels, 1 if variant == "density" else 3)

    def test_density_only_consumes_previous_velocity(self):
        config = self.make_config("density")
        model = CompressionModel(config)
        model.eval()
        p_d = torch.rand(1, 1, 16, 16)
        c_d = torch.rand(1, 1, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        zero_velocity = torch.zeros(1, 2, 16, 16)
        nonzero_velocity = torch.ones(1, 2, 16, 16)
        with torch.no_grad():
            first = model(p_d, zero_velocity, c_d, zero_velocity, mask)
            second = model(p_d, nonzero_velocity, c_d, nonzero_velocity, mask)
        self.assertFalse(torch.allclose(first, second))

    def test_dataparallel_prefix_is_removed(self):
        config = self.make_config("density")
        model = CompressionModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save(
                {
                    "model_config": config.to_dict(),
                    "state_dict": {f"module.{key}": value for key, value in model.state_dict().items()},
                },
                path,
            )
            loaded = load_model_checkpoint(path, device="cpu")
            self.assertEqual(loaded.model_variant, "density")
            self.assertTrue(all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
