from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pls_compression.cli import main
from compressor2 import main as density_velocity_main


class CliTests(unittest.TestCase):
    def test_train_cli_selects_requested_variant(self):
        result = SimpleNamespace(to_dict=lambda: {})
        with patch("pls_compression.cli.train_model", return_value=result) as train:
            with contextlib.redirect_stdout(io.StringIO()):
                status = main([
                    "--data-dir",
                    "data",
                    "--model-variant",
                    "density_velocity",
                    "--epochs",
                    "1",
                ])
        self.assertEqual(status, 0)
        config = train.call_args.kwargs["config"]
        self.assertEqual(config.model_config.model_variant, "density_velocity")

    def test_legacy_wrapper_default_and_explicit_override(self):
        result = SimpleNamespace(to_dict=lambda: {})
        with patch("pls_compression.cli.train_model", return_value=result) as train:
            with contextlib.redirect_stdout(io.StringIO()):
                status = density_velocity_main([
                    "--data-dir",
                    "data",
                    "--model-variant",
                    "density",
                ])
        self.assertEqual(status, 0)
        config = train.call_args.kwargs["config"]
        self.assertEqual(config.model_config.model_variant, "density")

    def test_check_failure_returns_nonzero(self):
        with patch("pls_compression.cli.evaluate_checkpoint", side_effect=ValueError("bad checkpoint")):
            with contextlib.redirect_stderr(io.StringIO()):
                status = main([
                    "check",
                    "--model",
                    "missing.pth",
                    "--data-dir",
                    "missing-data",
                ])
        self.assertNotEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
