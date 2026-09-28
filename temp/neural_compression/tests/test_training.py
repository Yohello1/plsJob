from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pls_compression.schema import ModelConfig
from pls_compression.training import TrainingConfig, train_model

from .fixtures import write_session


def _small_config(variant="density"):
    return ModelConfig(
        width=8,
        height=8,
        model_variant=variant,
        latent_dim=8,
        base_channels=2,
        bottleneck_channels=4,
        context_channels=2,
    )


class CheckpointThrottleTests(unittest.TestCase):
    def _run(self, root, epochs, **overrides):
        data = root / "data"
        data.mkdir(exist_ok=True)
        write_session(data, "one", width=8, height=8, frames=4)
        write_session(data, "two", width=8, height=8, frames=4)
        config = _small_config()
        options = dict(
            data_dir=data,
            output_dir=root / "out",
            model_config=config,
            epochs=epochs,
            batch_size=1,
            max_batches=1,
            skip_frames=1,
            n_steps=1,
            device="cpu",
        )
        options.update(overrides)
        writes = []
        real_save = __import__("pls_compression.training", fromlist=["save_checkpoint"]).save_checkpoint

        def counting_save(path, *args, **kwargs):
            writes.append(Path(path))
            return real_save(path, *args, **kwargs)

        with patch("pls_compression.training.save_checkpoint", side_effect=counting_save):
            result = train_model(config=TrainingConfig(**options))
        return result, writes

    def test_default_writes_on_every_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, writes = self._run(root, epochs=3)
            self.assertTrue(result.checkpoint.is_file())
            # One write per improving epoch, never more than the epoch count.
            self.assertLessEqual(len(writes), 3)
            self.assertGreaterEqual(len(writes), 1)

    def test_save_every_throttles_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, throttled = self._run(root, epochs=6, save_every=6)
            self.assertTrue(result.checkpoint.is_file())
            self.assertEqual(len(throttled), 1)

    def test_min_delta_suppresses_marginal_writes_but_keeps_the_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, writes = self._run(root, epochs=5, min_delta=1e9)
            self.assertTrue(result.checkpoint.is_file())
            self.assertEqual(len(writes), 1)

    def test_reported_best_is_the_true_minimum_regardless_of_throttling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _ = self._run(root, epochs=4, min_delta=1e9)
            reported = [row["val_loss"] for row in result.history]
            self.assertAlmostEqual(result.best_validation_loss, min(reported), places=9)

    def test_invalid_throttle_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(data_dir="d", output_dir="o", min_delta=-1.0)
        with self.assertRaises(ValueError):
            TrainingConfig(data_dir="d", output_dir="o", save_every=0)


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
