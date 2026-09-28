from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pls_compression.dataset import SPHDataset
from pls_compression.schema import DataSchema

from .fixtures import write_session


class DatasetTests(unittest.TestCase):
    def test_indexing_shapes_and_partial_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), width=8, height=8, frames=5)
            schema = DataSchema(8, 8)
            dataset = SPHDataset(session, skip=1, n_steps=2, schema=schema, augment=False)
            self.assertEqual(len(dataset), 3)
            previous_density, previous_velocity, future_density, future_velocity, mask = dataset[0]
            self.assertEqual(tuple(previous_density.shape), (1, 8, 8))
            self.assertEqual(tuple(previous_velocity.shape), (2, 8, 8))
            self.assertEqual(tuple(future_density.shape), (2, 1, 8, 8))
            self.assertEqual(tuple(future_velocity.shape), (2, 2, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertEqual(dataset.readers[session.resolve()].frame_count, 5)
            self.assertEqual(dataset.readers[session.resolve()].trailing_bytes, len(b"partial"))
            dataset.close()
            self.assertEqual(dataset.readers, {})

    def test_augmentation_does_not_modify_binary_source(self):
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), width=8, height=8, frames=4)
            binary = session / "sim_data.bin"
            before = binary.read_bytes()
            dataset = SPHDataset(session, skip=1, n_steps=1, schema=DataSchema(8, 8), augment=True)
            with patch("pls_compression.dataset.random.random", side_effect=(0.9, 0.9)):
                values = dataset[0]
            self.assertEqual(binary.read_bytes(), before)
            self.assertEqual(tuple(values[2].shape), (1, 1, 8, 8))
            dataset.close()

    def test_metadata_mismatch_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), width=8, height=8, frames=4)
            with self.assertRaisesRegex(ValueError, "width 8 != 16"):
                SPHDataset(session, schema=DataSchema(16, 8))
            (session / "metadata.json").write_text(json.dumps({"width": 8, "height": 8, "model_variant": "density_velocity"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_variant"):
                SPHDataset(session, schema=DataSchema(8, 8))

    def test_normalization_scales_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            session = write_session(Path(directory), width=8, height=8, frames=4)
            dataset = SPHDataset(
                session,
                skip=1,
                n_steps=1,
                schema=DataSchema(8, 8),
                normalization=(2.0, 3.0),
            )
            raw_density, raw_velocity, _ = dataset.load_frame_data(session, 0)
            density, velocity, _, _, _ = dataset[0]
            self.assertTrue(torch.allclose(density, raw_density * 2.0))
            self.assertTrue(torch.allclose(velocity, raw_velocity * 3.0))
            dataset.close()


if __name__ == "__main__":
    unittest.main()
