from __future__ import annotations

import unittest

import torch

from pls_compression.losses import LossConfig, hybrid_loss
from pls_compression.metrics import compute_metrics, identity_baseline_mse, ssim, ssim_fluid, ssim_map, zero_baseline_mse


class LossAndMetricTests(unittest.TestCase):
    def test_all_background_loss_has_finite_gradients(self):
        prediction = torch.zeros(2, 1, 4, 4, requires_grad=True)
        target = torch.zeros(2, 1, 4, 4)
        loss = hybrid_loss(prediction, target, LossConfig())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_density_velocity_loss_uses_all_channels(self):
        prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
        target = torch.zeros(1, 3, 4, 4)
        target[:, 0] = 0.5
        target[:, 1:] = 0.25
        loss = hybrid_loss(prediction, target, LossConfig(fluid_weight=2.0, mass_weight=1.0))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_metrics_and_baselines(self):
        target = torch.ones(2, 1, 2, 2)
        prediction = torch.zeros_like(target)
        previous = torch.full_like(target, 0.5)
        metrics = compute_metrics(prediction, target, previous)
        self.assertAlmostEqual(metrics["mse"], 1.0)
        self.assertAlmostEqual(metrics["zero_mse"], 1.0)
        self.assertAlmostEqual(metrics["identity_mse"], 0.25)
        self.assertAlmostEqual(float(zero_baseline_mse(target)), 1.0)
        self.assertAlmostEqual(float(identity_baseline_mse(target, previous)), 0.25)


class StructuralSimilarityTests(unittest.TestCase):
    def test_identical_images_score_one(self):
        for shape in ((1, 1, 64, 64), (2, 3, 32, 32), (1, 1, 8, 8), (1, 1, 2, 2)):
            with self.subTest(shape=shape):
                value = torch.rand(*shape)
                self.assertAlmostEqual(float(ssim(value, value)), 1.0, places=5)

    def test_matches_independent_reference(self):
        # Wang et al. 2004 with an 11x11 Gaussian (sigma 1.5), data range 1.0.
        window, sigma, data_range = 11, 1.5, 1.0
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        coordinates = torch.arange(window, dtype=torch.float64) - (window - 1) / 2
        kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, window, window)

        def reference(prediction, target):
            totals = []
            for index in range(prediction.shape[0]):
                for channel in range(prediction.shape[1]):
                    left = prediction[index, channel].double()
                    right = target[index, channel].double()
                    filtered = lambda value: torch.nn.functional.conv2d(
                        value.reshape(1, 1, *value.shape), kernel, padding=window // 2
                    )
                    mean_left, mean_right = filtered(left), filtered(right)
                    product = mean_left * mean_right
                    variance_left = filtered(left * left) - product
                    variance_right = filtered(right * right) - product
                    covariance = filtered(left * right) - product
                    luminance = (2 * mean_left * mean_right + c1) / (
                        mean_left**2 + mean_right**2 + c1
                    )
                    structure = (2 * covariance + c2) / (variance_left + variance_right + c2)
                    totals.append((luminance * structure).mean())
            return torch.stack(totals).mean().item()

        for shape in ((1, 1, 64, 64), (2, 1, 32, 32), (1, 3, 40, 40)):
            with self.subTest(shape=shape):
                prediction = torch.rand(*shape)
                target = torch.rand(*shape)
                self.assertAlmostEqual(float(ssim(prediction, target)), reference(prediction, target), places=5)

    def test_symmetric_bounded_and_decreasing_in_noise(self):
        torch.manual_seed(11)
        prediction = torch.rand(2, 1, 32, 32)
        self.assertAlmostEqual(float(ssim(prediction, prediction)), 1.0, places=5)
        target = torch.rand(2, 1, 32, 32)
        self.assertAlmostEqual(float(ssim(prediction, target)), float(ssim(target, prediction)), places=6)
        noisy = [float(ssim(prediction, prediction + torch.randn_like(prediction) * scale)) for scale in (0.05, 0.25)]
        self.assertLess(noisy[0], 1.0)
        self.assertLess(noisy[1], noisy[0])
        self.assertGreaterEqual(noisy[1], -1.0)

    def test_valid_padding_crops_the_border(self):
        prediction = torch.rand(1, 1, 32, 32)
        target = torch.rand(1, 1, 32, 32)
        self.assertEqual(tuple(ssim_map(prediction, target).shape), (1, 32, 32))
        self.assertEqual(tuple(ssim_map(prediction, target, padding="valid").shape), (1, 22, 22))

    def test_differentiable_and_dtype_preserving(self):
        prediction = torch.rand(1, 3, 24, 24, requires_grad=True)
        value = ssim(prediction, torch.rand(1, 3, 24, 24))
        value.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertEqual(ssim_map(torch.rand(1, 1, 8, 8).double(), torch.rand(1, 1, 8, 8).double()).dtype, torch.float64)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            ssim(torch.rand(2, 4), torch.rand(2, 4))
        with self.assertRaises(ValueError):
            ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), data_range=0.0)
        with self.assertRaises(ValueError):
            ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), window_size=4)
        with self.assertRaises(ValueError):
            ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), sigma=0.0)
        with self.assertRaises(ValueError):
            ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), padding="reflect")
        with self.assertRaises(ValueError):
            ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 9, 8))

    def test_compute_metrics_reports_ssim_per_channel_and_density(self):
        prediction = torch.rand(2, 3, 16, 16)
        target = torch.rand(2, 3, 16, 16)
        metrics = compute_metrics(prediction, target)
        self.assertAlmostEqual(metrics["ssim"], float(ssim(prediction, target)), places=6)
        self.assertAlmostEqual(
            metrics["ssim_density"],
            float(ssim(prediction[:, 0:1], target[:, 0:1])),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["ssim_fluid"],
            float(ssim_fluid(prediction, target)),
            places=6,
        )


class FluidMaskedSimilarityTests(unittest.TestCase):
    # The field size and blob size are chosen so fluid covers ~0.6% of the
    # image, close to the fraction a real 400x400 SPH frame sees. That ratio is
    # what makes the unmasked mean uninformative in the first place.
    SIZE = 256
    BLOB = 20

    @classmethod
    def _field(cls, drop=False, shift=0, blur=0, background_patch=False):
        target = torch.full((1, 1, cls.SIZE, cls.SIZE), 0.01)
        target[0, 0, 100 : 100 + cls.BLOB, 110 : 110 + cls.BLOB] = 1.0
        prediction = target.clone()
        if drop:
            prediction[0, 0, 100 : 100 + cls.BLOB, 110 : 110 + cls.BLOB] = 0.0
        if shift:
            prediction[0, 0, 100 : 100 + cls.BLOB, 110 : 110 + cls.BLOB] = 0.0
            offset = 100 + shift
            side = 110 + shift
            prediction[0, 0, offset : offset + cls.BLOB, side : side + cls.BLOB] = 1.0
        if blur:
            prediction = torch.nn.functional.avg_pool2d(prediction, blur, 1, blur // 2)
        if background_patch:
            prediction[0, 0, 0:20, 0:20] = 0.4
        return prediction, target

    def test_masking_recovers_discrimination_on_sparse_fields(self):
        # The unmasked mean is dominated by background agreement and rates every
        # one of these failures above 0.98, which is why the masked score exists.
        _, target = self._field()
        cases = (
            ("dropped", self._field(drop=True)[0], 0.1),
            ("shifted", self._field(shift=6)[0], 0.2),
            ("blurred", self._field(blur=9)[0], 0.5),
        )
        for name, prediction, ceiling in cases:
            with self.subTest(case=name):
                self.assertGreater(float(ssim(prediction, target)), 0.98)
                self.assertLess(float(ssim_fluid(prediction, target)), ceiling)

    def test_masked_score_orders_failures_while_unmasked_cannot(self):
        _, target = self._field()
        dropped = float(ssim_fluid(self._field(drop=True)[0], target))
        blurred = float(ssim_fluid(self._field(blur=9)[0], target))
        self.assertLess(dropped, blurred)
        unmasked = [float(ssim(prediction, target)) for prediction in (target, self._field(drop=True)[0])]
        self.assertGreater(unmasked[0] - unmasked[1], 0.0)
        self.assertLess(unmasked[0] - unmasked[1], 0.05)

    def test_perfect_prediction_scores_one(self):
        prediction, target = self._field()
        self.assertAlmostEqual(float(ssim_fluid(prediction, target)), 1.0, places=5)

    def test_background_only_target_scores_zero_without_nan(self):
        background = torch.full((1, 1, 32, 32), 0.01)
        value = ssim_fluid(background, background)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(float(value), 0.0)

    def test_background_error_is_ignored(self):
        # Fluid is exactly right and only empty background is wrong: the masked
        # score stays at 1.0 while the unmasked score drops.
        prediction, target = self._field(background_patch=True)
        self.assertAlmostEqual(float(ssim_fluid(prediction, target)), 1.0, places=5)
        self.assertLess(float(ssim(prediction, target)), 1.0)

    def test_differentiable(self):
        prediction = torch.rand(1, 1, 32, 32, requires_grad=True)
        target = torch.full((1, 1, 32, 32), 0.01)
        target[0, 0, 8:24, 8:24] = 0.9
        ssim_fluid(prediction, target).backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
