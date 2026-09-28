from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pls_compression.schema import ModelConfig
from pls_compression.training import TrainingConfig, train_model

from .fixtures import write_session


class TrainingSmokeTests(unittest.TestCase):
    def test_one_batch_cpu_smoke_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            write_session(data, "one", width=8, height=8, frames=4)
            write_session(data, "two", width=8, height=8, frames=4)
            config = ModelConfig(
                width=8,
                height=8,
                model_variant="density",
                latent_dim=8,
                base_channels=2,
                bottleneck_channels=4,
                context_channels=2,
            )
            output = root / "output"
            result = train_model(
                config=TrainingConfig(
                    data_dir=data,
                    output_dir=output,
                    model_config=config,
                    epochs=1,
                    batch_size=1,
                    max_batches=1,
                    smoke=True,
                    skip_frames=1,
                    n_steps=1,
                    device="cpu",
                ),
            )
            self.assertTrue(result.checkpoint.is_file())
            self.assertEqual(len(result.history), 1)
            self.assertEqual(set(result.train_sessions).isdisjoint(result.validation_sessions), True)
            self.assertIn("val_zero_mse", result.history[0])
            self.assertTrue(result.checkpoint.parent.joinpath("run_config.json").is_file())

    def test_density_velocity_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            write_session(data, "one", width=8, height=8, frames=4, variant="density_velocity")
            write_session(data, "two", width=8, height=8, frames=4, variant="density_velocity")
            config = ModelConfig(
                width=8,
                height=8,
                model_variant="density_velocity",
                latent_dim=8,
                base_channels=2,
                bottleneck_channels=4,
                context_channels=2,
            )
            result = train_model(
                config=TrainingConfig(
                    data_dir=data,
                    output_dir=root / "output_velocity",
                    model_config=config,
                    epochs=1,
                    batch_size=1,
                    max_batches=1,
                    smoke=True,
                    skip_frames=1,
                    n_steps=1,
                    device="cpu",
                ),
            )
            self.assertTrue(result.checkpoint.is_file())
            self.assertEqual(result.model.model_variant, "density_velocity")
            self.assertEqual(len(result.history), 1)
            self.assertIn("val_mse", result.history[0])


if __name__ == "__main__":
    unittest.main()
