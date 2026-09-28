from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pls_compression.orchestration import PACKAGE_ROOT
from pls_compression.pipeline import (
    DataBudget,
    build_orchestration_argv,
    build_parser,
    compute_data_budget,
    describe_budget,
    main,
    nearest_existing_directory,
    resolution_warning,
)

CLEAN_ENVIRONMENT = {
    key: value
    for key, value in os.environ.items()
    if key
    not in {
        "VARIANT", "DATA_DIR", "OUTPUT_DIR", "FRAMES_PER_RUN", "RUNS_PER_CYCLE", "MAX_PARALLEL",
        "CYCLES", "EPOCHS", "BATCH_SIZE", "EFFECTIVE_BATCH", "SKIP_FRAMES", "LEARNING_RATE",
        "VALIDATION_FRACTION", "NUM_WORKERS", "MODEL_FILENAME", "MIN_DELTA", "SAVE_EVERY",
        "KEEP_LAST_CHECKPOINTS", "DEVICE", "MAX_SESSIONS", "SIMULATION_SEED", "SEED", "WIDTH",
        "HEIGHT", "LATENT_DIM", "MAX_BATCHES", "SMOKE", "SPH_ROOT", "NO_RESUME", "PRUNE",
        "SIM_WIDTH", "SIM_HEIGHT",
    }
}


def _argv(*arguments: str, environment: dict | None = None) -> list[str]:
    with patch.dict(os.environ, {**CLEAN_ENVIRONMENT, **(environment or {})}):
        return build_orchestration_argv(build_parser().parse_args(list(arguments)))


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class ParserDefaultsTests(unittest.TestCase):
    def test_defaults_match_the_documented_values(self):
        argv = _argv("all")
        self.assertEqual(_value(argv, "--batch-size"), "8")
        self.assertEqual(_value(argv, "--effective-batch-size"), "32")
        self.assertEqual(_value(argv, "--skip-frames"), "10")
        self.assertEqual(_value(argv, "--learning-rate"), "5e-4")
        self.assertEqual(_value(argv, "--validation-fraction"), "0.1")
        self.assertEqual(_value(argv, "--frames-per-run"), "100")
        self.assertEqual(_value(argv, "--runs-per-cycle"), "1")
        self.assertEqual(_value(argv, "--latent-dim"), "1024")
        self.assertEqual(_value(argv, "--min-delta"), "0")
        self.assertEqual(_value(argv, "--save-every"), "1")
        self.assertEqual(_value(argv, "--keep-last-checkpoints"), "0")
        self.assertEqual(_value(argv, "--variant"), "density")
        self.assertEqual(_value(argv, "--phase"), "all")
        self.assertEqual(_value(argv, "--data-dir"), str(PACKAGE_ROOT / "data"))
        self.assertEqual(_value(argv, "--output-dir"), str(PACKAGE_ROOT / "attempts"))

    def test_every_documented_environment_variable_is_read(self):
        # A variable that is not wired to an option would be silently ignored.
        overrides = {
            "VARIANT": "density_velocity",
            "DATA_DIR": "/tmp/d",
            "OUTPUT_DIR": "/tmp/o",
            "FRAMES_PER_RUN": "42",
            "RUNS_PER_CYCLE": "6",
            "MAX_PARALLEL": "3",
            "CYCLES": "4",
            "EPOCHS": "7",
            "BATCH_SIZE": "16",
            "EFFECTIVE_BATCH": "64",
            "SKIP_FRAMES": "3",
            "LEARNING_RATE": "1e-4",
            "VALIDATION_FRACTION": "0.25",
            "NUM_WORKERS": "5",
            "MODEL_FILENAME": "run.pth",
            "MIN_DELTA": "0.01",
            "SAVE_EVERY": "5",
            "KEEP_LAST_CHECKPOINTS": "3",
            "DEVICE": "cpu",
            "MAX_SESSIONS": "9",
            "SIMULATION_SEED": "11",
            "SEED": "13",
            "WIDTH": "512",
            "HEIGHT": "256",
            "LATENT_DIM": "64",
            "MAX_BATCHES": "2",
            "SPH_ROOT": "/tmp/sph",
        }
        argv = _argv("all", environment=overrides)
        for flag, expected in (
            ("--variant", "density_velocity"), ("--data-dir", "/tmp/d"), ("--output-dir", "/tmp/o"),
            ("--frames-per-run", "42"), ("--runs-per-cycle", "6"), ("--max-parallel", "3"),
            ("--cycles", "4"), ("--epochs", "7"), ("--batch-size", "16"),
            ("--effective-batch-size", "64"), ("--skip-frames", "3"), ("--learning-rate", "1e-4"),
            ("--validation-fraction", "0.25"), ("--num-workers", "5"), ("--model-filename", "run.pth"),
            ("--min-delta", "0.01"), ("--save-every", "5"), ("--keep-last-checkpoints", "3"),
            ("--device", "cpu"), ("--max-sessions", "9"), ("--simulation-seed", "11"), ("--seed", "13"),
            ("--width", "512"), ("--height", "256"), ("--latent-dim", "64"), ("--max-batches", "2"),
            ("--sph-root", "/tmp/sph"),
        ):
            with self.subTest(flag=flag):
                self.assertEqual(_value(argv, flag), expected)

    def test_environment_flags_are_read(self):
        argv = _argv("all", environment={"SMOKE": "1", "PRUNE": "1", "NO_RESUME": "1"})
        self.assertIn("--smoke", argv)
        self.assertIn("--prune", argv)
        self.assertIn("--no-resume", argv)
        self.assertEqual(_argv("all"), [a for a in _argv("all")])

    def test_command_line_overrides_the_environment(self):
        argv = _argv("all", "--epochs", "99", environment={"EPOCHS": "7"})
        self.assertEqual(_value(argv, "--epochs"), "99")

    def test_values_are_passed_through_as_text(self):
        # Strings are forwarded unconverted so orchestration owns the parsing;
        # this is why "0" must not become "0.0" on the way through.
        self.assertEqual(_value(_argv("all"), "--min-delta"), "0")
        self.assertEqual(_value(_argv("all", "--min-delta", "0.50"), "--min-delta"), "0.50")

    def test_sim_resolution_environment_is_read(self):
        argv = _argv("generate", environment={"SIM_WIDTH": "800", "SIM_HEIGHT": "600"})
        # SIM_* are not forwarded to orchestration; they only feed the budget.
        self.assertNotIn("--sim-width", argv)
        self.assertNotIn("800", argv)


class PhaseArgumentTests(unittest.TestCase):
    def test_train_appends_skip_sim(self):
        self.assertIn("--skip-sim", _argv("train"))
        self.assertNotIn("--skip-sim", _argv("all"))
        self.assertNotIn("--skip-sim", _argv("generate"))

    def test_phase_defaults_to_all(self):
        self.assertEqual(_value(_argv(), "--phase"), "all")

    def test_invalid_phase_is_rejected(self):
        with patch.dict(os.environ, CLEAN_ENVIRONMENT):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["bogus"])

    def test_optional_arguments_are_appended_in_a_stable_order(self):
        argv = _argv("train", "--smoke", "--prune", "--no-resume", "--max-batches", "3")
        tail = argv[argv.index("--max-batches") :]
        self.assertEqual(tail, ["--max-batches", "3", "--smoke", "--prune", "--no-resume", "--skip-sim"])


class BudgetTests(unittest.TestCase):
    def test_frame_and_session_sizes_use_the_simulator_resolution(self):
        budget = compute_data_budget("/tmp", frames_per_run=100, runs_per_cycle=1, cycles=1, sim_width=400, sim_height=400)
        self.assertEqual(budget.frame_bytes, 4 * 400 * 400 * 4)
        self.assertEqual(budget.session_bytes, 100 * budget.frame_bytes)
        self.assertEqual(budget.session_mib, 244)

    def test_model_size_does_not_affect_the_budget(self):
        small = compute_data_budget("/tmp", 100, 1, 1, 400, 400)
        # width/height are not even arguments here, which is the point: the
        # simulator resolution is what the data will actually occupy.
        self.assertEqual(small.total_bytes, small.session_bytes)

    def test_total_scales_with_runs_and_cycles(self):
        one = compute_data_budget("/tmp", 10, 1, 1, 400, 400)
        many = compute_data_budget("/tmp", 10, 4, 3, 400, 400)
        self.assertEqual(many.total_bytes, one.total_bytes * 12)

    def test_probe_walks_up_to_an_existing_directory(self):
        probe = nearest_existing_directory("/tmp/definitely/not/here/yet")
        self.assertTrue(probe.is_dir())
        self.assertEqual(probe, Path("/tmp"))

    def test_exceeds_free_space(self):
        budget = DataBudget(
            frame_bytes=1, session_bytes=10, total_bytes=100, free_bytes=50,
            probe="/tmp", frames_per_run=1, runs_per_cycle=1, cycles=1,
        )
        self.assertTrue(budget.exceeds_free_space)
        budget = DataBudget(
            frame_bytes=1, session_bytes=10, total_bytes=100, free_bytes=1000,
            probe="/tmp", frames_per_run=1, runs_per_cycle=1, cycles=1,
        )
        self.assertFalse(budget.exceeds_free_space)

    def test_undeterminable_free_space_does_not_exceed(self):
        budget = DataBudget(
            frame_bytes=1, session_bytes=10, total_bytes=100, free_bytes=None,
            probe="/tmp", frames_per_run=1, runs_per_cycle=1, cycles=1,
        )
        self.assertFalse(budget.exceeds_free_space)

    def test_budget_description_format(self):
        budget = DataBudget(
            frame_bytes=1, session_bytes=244 * 1048576, total_bytes=1, free_bytes=58 * 1073741824,
            probe="/tmp", frames_per_run=100, runs_per_cycle=1, cycles=1,
        )
        self.assertEqual(
            describe_budget(budget),
            "data budget: 1 cycles x 1 runs x 100 frames = ~244 MiB (58 GiB free on /tmp)",
        )

    def test_needed_gib_rounds_up(self):
        budget = DataBudget(
            frame_bytes=1, session_bytes=1, total_bytes=239 * 1073741824, free_bytes=1,
            probe="/tmp", frames_per_run=1, runs_per_cycle=1, cycles=1,
        )
        self.assertEqual(budget.needed_gib, 240)


class ResolutionWarningTests(unittest.TestCase):
    def test_no_warning_when_they_match(self):
        self.assertIsNone(resolution_warning(400, 400, 400, 400))

    def test_warning_names_both_resolutions(self):
        message = resolution_warning(8, 8, 400, 400)
        self.assertIsNotNone(message)
        self.assertIn("8x8", message)
        self.assertIn("400x400", message)


class MainTests(unittest.TestCase):
    def _run(self, *arguments, environment=None, exit_code=None):
        with patch.dict(os.environ, {**CLEAN_ENVIRONMENT, **(environment or {})}), patch(
            "pls_compression.pipeline.orchestration_main", return_value=0
        ) as orchestrator:
            with patch("sys.stdout"), patch("sys.stderr"):
                code = main(list(arguments))
        if exit_code is not None:
            self.assertEqual(code, exit_code)
        return code, orchestrator

    def test_generate_phase_forwards_to_orchestration(self):
        code, orchestrator = self._run("generate", "--data-dir", "/tmp/d", exit_code=0)
        argv = orchestrator.call_args.args[0]
        self.assertEqual(_value(argv, "--phase"), "generate")
        self.assertNotIn("--skip-sim", argv)

    def test_oversized_generation_is_refused(self):
        code, orchestrator = self._run("generate", "--frames-per-run", "99999999", exit_code=1)
        orchestrator.assert_not_called()

    def test_train_phase_skips_the_resolution_warning(self):
        with patch.dict(os.environ, CLEAN_ENVIRONMENT), patch(
            "pls_compression.pipeline.orchestration_main", return_value=0
        ):
            with patch("sys.stdout"), patch("sys.stderr") as stderr:
                main(["train", "--width", "8", "--height", "8"])
        self.assertEqual(stderr.write.call_count, 0)

    def test_generate_phase_emits_the_resolution_warning(self):
        with patch.dict(os.environ, CLEAN_ENVIRONMENT), patch(
            "pls_compression.pipeline.orchestration_main", return_value=0
        ):
            with patch("sys.stdout"), patch("sys.stderr") as stderr:
                main(["generate", "--width", "8", "--height", "8"])
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("model is 8x8", message)
        self.assertIn("400x400", message)

    def test_orchestration_errors_become_exit_two(self):
        with patch.dict(os.environ, CLEAN_ENVIRONMENT), patch(
            "pls_compression.pipeline.orchestration_main", side_effect=ValueError("boom")
        ):
            with patch("sys.stdout"), patch("sys.stderr"):
                self.assertEqual(main(["train"]), 2)

    def test_invalid_numeric_option_is_reported(self):
        code, orchestrator = self._run("generate", "--frames-per-run", "abc", exit_code=2)
        orchestrator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
