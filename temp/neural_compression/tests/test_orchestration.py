from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pls_compression.orchestration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EFFECTIVE_BATCH_SIZE,
    PACKAGE_ROOT,
    ActiveLearningConfig,
    build_parser,
    main,
    run_active_learning,
    run_simulations,
    simulation_binary,
    simulation_build_target,
)


class OrchestrationTests(unittest.TestCase):
    def test_variant_specific_targets_and_binaries(self):
        self.assertEqual(simulation_build_target("density"), "draw2-density-only")
        self.assertEqual(simulation_build_target("density_velocity"), "draw2-density-velocity")
        self.assertEqual(simulation_binary("density").name, "draw2-density-only")
        self.assertEqual(simulation_binary("density_velocity").name, "draw2-density-velocity")

    def test_simulation_command_receives_variant_frames_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "attempts",
                runs_per_cycle=1,
                model_variant="density_velocity",
                frames_per_run=7,
                simulation_seed=11,
                simulation_command=[str(PACKAGE_ROOT / "spawn_random.sh")],
            )
            with patch("pls_compression.orchestration.subprocess.run") as run:
                run_simulations(config, cycle=3)
            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertEqual(command[0], str(PACKAGE_ROOT / "spawn_random.sh"))
            self.assertEqual(command[command.index("--variant") + 1], "density-velocity")
            self.assertEqual(command[command.index("--frames") + 1], "7")
            self.assertEqual(command[command.index("--seed") + 1], str(11 + 3 * 1000003))
            self.assertEqual(environment["SPH_MODEL_VARIANT"], "density-velocity")
            self.assertEqual(environment["SPH_BUILD_TARGET"], "draw2-density-velocity")
            self.assertEqual(environment["SPH_DATA_ROOT"], str(config.data_dir))

    def test_simulation_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "attempts",
                runs_per_cycle=1,
                simulation_command=[str(PACKAGE_ROOT / "spawn_random.sh")],
            )
            failure = subprocess.CalledProcessError(9, ["spawn_random.sh"])
            with patch("pls_compression.orchestration.subprocess.run", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_simulations(config, cycle=1)


class TrainingFlagTests(unittest.TestCase):
    def test_defaults_batch_eight_effective_thirty_two(self):
        config = ActiveLearningConfig(data_dir="d", output_dir="o")
        self.assertEqual(config.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.effective_batch_size, DEFAULT_EFFECTIVE_BATCH_SIZE)
        self.assertEqual(config.effective_batch_size, 32)
        self.assertEqual(config.skip_frames, 10)

    def test_training_hyperparameters_reach_the_cycle_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                batch_size=4,
                effective_batch_size=16,
                skip_frames=3,
                n_steps=2,
                learning_rate=1e-3,
                validation_fraction=0.25,
                num_workers=3,
                model_filename="custom.pth",
            )
            with patch("pls_compression.orchestration.train_model") as train:
                train.return_value.checkpoint = root / "out" / "cycle_1" / "custom.pth"
                run_active_learning(config)
            training = train.call_args.kwargs["config"]
            self.assertEqual(training.batch_size, 4)
            self.assertEqual(training.effective_batch_size, 16)
            self.assertEqual(training.skip_frames, 3)
            self.assertEqual(training.n_steps, 2)
            self.assertEqual(training.learning_rate, 1e-3)
            self.assertEqual(training.validation_fraction, 0.25)
            self.assertEqual(training.num_workers, 3)
            self.assertEqual(training.model_filename, "custom.pth")
            self.assertEqual(training.output_dir, root / "out" / "cycle_1")

    def test_explicit_training_config_is_overridden_not_dropped(self):
        from pls_compression.training import TrainingConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                batch_size=6,
                skip_frames=7,
                training_config=TrainingConfig(
                    data_dir=root / "other",
                    output_dir=root / "elsewhere",
                    batch_size=1,
                    skip_frames=10,
                ),
            )
            with patch("pls_compression.orchestration.train_model") as train:
                train.return_value.checkpoint = root / "out" / "cycle_1" / "best_model.pth"
                run_active_learning(config)
            training = train.call_args.kwargs["config"]
            self.assertEqual(training.batch_size, 6)
            self.assertEqual(training.skip_frames, 7)
            self.assertEqual(training.data_dir, root / "data")
            self.assertEqual(training.output_dir, root / "out" / "cycle_1")

    def test_invalid_hyperparameters_are_rejected(self):
        for overrides in (
            {"batch_size": 0},
            {"batch_size": 8, "effective_batch_size": 4},
            {"skip_frames": 0},
            {"n_steps": 0},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"validation_fraction": 0.0},
            {"validation_fraction": 1.0},
            {"num_workers": -1},
            {"noise_std": -0.1},
            {"gradient_clip_norm": 0.0},
            {"model_filename": ""},
            {"phase": "nonsense"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    ActiveLearningConfig(data_dir="d", output_dir="o", **overrides)

    def test_parser_exposes_training_flags(self):
        args = build_parser().parse_args(
            [
                "--batch-size", "16",
                "--effective-batch-size", "64",
                "--skip-frames", "3",
                "--learning-rate", "1e-4",
                "--gradient-clip-norm", "1.0",
                "--noise-std", "0.05",
                "--num-workers", "2",
                "--model-filename", "run.pth",
                "--no-resume",
            ]
        )
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.effective_batch_size, 64)
        self.assertEqual(args.skip_frames, 3)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.gradient_clip_norm, 1.0)
        self.assertEqual(args.noise_std, 0.05)
        self.assertEqual(args.num_workers, 2)
        self.assertEqual(args.model_filename, "run.pth")
        self.assertFalse(args.resume)


class ResumeAcrossCyclesTests(unittest.TestCase):
    def test_second_cycle_resumes_from_the_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "out" / "cycle_1" / "best_model.pth"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"stub")
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                cycles=2,
                runs_per_cycle=0,
                simulation_command=["true"],
            )
            with patch("pls_compression.orchestration.run_simulations"), patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                train.side_effect = lambda config, resume_path=None: type(
                    "Result",
                    (),
                    {"checkpoint": Path(resume_path) if resume_path else root / "out" / "cycle_1" / "best_model.pth"},
                )()
                run_active_learning(config)
            self.assertEqual(train.call_count, 2)
            first_call, second_call = train.call_args_list
            self.assertIsNone(first_call.kwargs["resume_path"])
            self.assertEqual(second_call.kwargs["resume_path"], first)

    def test_missing_checkpoint_does_not_break_the_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                cycles=2,
                runs_per_cycle=0,
                simulation_command=["true"],
            )
            with patch("pls_compression.orchestration.run_simulations"), patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                train.return_value = type(
                    "Result", (), {"checkpoint": root / "out" / "cycle_1" / "never_written.pth"}
                )()
                run_active_learning(config)
            self.assertIsNone(train.call_args_list[1].kwargs["resume_path"])

    def test_no_resume_restarts_from_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "out" / "cycle_1" / "best_model.pth"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"stub")
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                cycles=2,
                runs_per_cycle=0,
                resume=False,
                simulation_command=["true"],
            )
            with patch("pls_compression.orchestration.run_simulations"), patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                train.return_value = type("Result", (), {"checkpoint": first})()
                run_active_learning(config)
            for call in train.call_args_list:
                self.assertIsNone(call.kwargs["resume_path"])


class PhaseTests(unittest.TestCase):
    def test_generate_phase_does_not_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                phase="generate",
                runs_per_cycle=0,
                simulation_command=["true"],
            )
            with patch("pls_compression.orchestration.run_simulations") as sim, patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                run_active_learning(config)
            sim.assert_called_once()
            train.assert_not_called()

    def test_train_phase_does_not_simulate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ActiveLearningConfig(
                data_dir=root / "data",
                output_dir=root / "out",
                phase="train",
                runs_per_cycle=0,
                simulation_command=["true"],
            )
            with patch("pls_compression.orchestration.run_simulations") as sim, patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                train.return_value = type("Result", (), {"checkpoint": root / "c.pth"})()
                run_active_learning(config)
            sim.assert_not_called()
            train.assert_called_once()

    def test_main_generate_phase_reports_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            session = data / "20260101_000000_aaaa"
            session.mkdir(parents=True)
            (session / "sim_data.bin").write_bytes(b"\0" * (4 * 8 * 8 * 4 * 3))
            (session / "metadata.json").write_text(
                '{"model_variant": "density-only", "width": 8, "height": 8, '
                '"fields": ["density", "velocity_x", "velocity_y", "obstacle_mask"], "dtype": "float32"}',
                encoding="utf-8",
            )
            with patch("pls_compression.orchestration.run_simulations"), patch(
                "pls_compression.orchestration.train_model"
            ) as train:
                exit_code = main(
                    [
                        "--phase", "generate",
                        "--data-dir", str(data),
                        "--output-dir", str(root / "out"),
                        "--width", "8",
                        "--height", "8",
                    ]
                )
            self.assertEqual(exit_code, 0)
            train.assert_not_called()

    def test_skip_sim_means_train_not_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = {}

            def fake_train(config, resume_path=None):
                captured["config"] = config
                return type("Result", (), {"checkpoint": root / "c.pth"})()

            with patch("pls_compression.orchestration.run_simulations") as sim, patch(
                "pls_compression.orchestration.train_model", side_effect=fake_train
            ):
                main(
                    [
                        "--phase", "train",
                        "--data-dir", str(root / "data"),
                        "--output-dir", str(root / "out"),
                        "--width", "8",
                        "--height", "8",
                        "--latent-dim", "8",
                        "--smoke",
                        "--skip-sim",
                    ]
                )
            sim.assert_not_called()
            self.assertIn("config", captured)

    def test_skip_sim_with_generate_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                main(
                    [
                        "--phase", "generate",
                        "--data-dir", str(root / "data"),
                        "--output-dir", str(root / "out"),
                        "--skip-sim",
                    ]
                )


class SpawnScriptTests(unittest.TestCase):
    def test_script_resolves_sibling_sph_and_is_deterministic(self):
        script = PACKAGE_ROOT / "spawn_random.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sph = root / "sph"
            sph.mkdir()
            data = root / "data"
            binary = sph / "draw2-density-only"
            binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\"\nprintf '%s\\n' \"$SPH_DATA_ROOT\"\nprintf '%s\\n' \"$SPH_MODEL_VARIANT\"\n", encoding="utf-8")
            binary.chmod(0o755)
            environment = os.environ.copy()
            environment.update({"SPH_ROOT": str(sph), "SPH_DATA_ROOT": str(data), "SPH_SCENARIO_SEED": "23"})
            environment.pop("SPH_MODEL_VARIANT", None)
            environment.pop("SPH_VARIANT", None)
            environment.pop("SPH_BUILD_TARGET", None)
            environment.pop("SPH_BINARY", None)
            environment.pop("SPH_DRAW2", None)
            first = subprocess.run([str(script), "2", "--variant", "density-only"], env=environment, text=True, capture_output=True, check=True)
            second = subprocess.run([str(script), "2", "--variant", "density-only"], env=environment, text=True, capture_output=True, check=True)
            self.assertEqual(first.stdout, second.stdout)
            self.assertIn("--headless", first.stdout)
            self.assertIn("density-only", first.stdout)
            self.assertIn(str(data.resolve()), first.stdout)


if __name__ == "__main__":
    unittest.main()
