from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from pls_compression.dataset import SessionReader
from pls_compression.schema import DataSchema, canonical_model_variant


SPH_ROOT = Path(__file__).resolve().parents[2] / "sph"
BINARIES = {
    "density": SPH_ROOT / "draw2-density-only",
    "density_velocity": SPH_ROOT / "draw2-density-velocity",
}


@unittest.skipUnless(all(path.is_file() and os.access(path, os.X_OK) for path in BINARIES.values()), "native simulator binaries are not built")
class NativeMetadataContractTests(unittest.TestCase):
    def test_real_binaries_match_python_frame_contract(self):
        for variant, binary in BINARIES.items():
            with self.subTest(variant=variant):
                with tempfile.TemporaryDirectory() as directory:
                    environment = os.environ.copy()
                    environment["SPH_DATA_ROOT"] = directory
                    result = subprocess.run(
                        [str(binary), "--headless", "--frames", "2", "--fluid", "120", "120", "24", "24"],
                        cwd=SPH_ROOT,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=120,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    sessions = [path for path in Path(directory).iterdir() if (path / "sim_data.bin").is_file()]
                    self.assertEqual(len(sessions), 1)
                    session = sessions[0]
                    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
                    expected_label = "density-only" if variant == "density" else "density-velocity"
                    self.assertEqual(metadata["model_variant"], expected_label)
                    self.assertEqual(canonical_model_variant(metadata["model_variant"]), variant)
                    self.assertEqual(metadata["width"], 400)
                    self.assertEqual(metadata["height"], 400)
                    self.assertEqual(metadata["fields"], ["density", "velocity_x", "velocity_y", "obstacle_mask"])
                    self.assertEqual(metadata["dtype"], "float32")
                    reader = SessionReader(session, DataSchema(400, 400, model_variant=variant))
                    try:
                        self.assertEqual(reader.frame_count, 2)
                        density, velocity, mask = reader.read_frame(0)
                        self.assertEqual(tuple(density.shape), (1, 400, 400))
                        self.assertEqual(tuple(velocity.shape), (2, 400, 400))
                        self.assertEqual(tuple(mask.shape), (1, 400, 400))
                        self.assertGreater(int(torch.count_nonzero(velocity)), 0)
                    finally:
                        reader.close()


if __name__ == "__main__":
    unittest.main()
