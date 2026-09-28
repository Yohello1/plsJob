from __future__ import annotations

import unittest

from pls_compression.schema import (
    DEFAULT_MODEL_VARIANT,
    DTYPE,
    FIELDS,
    HEIGHT,
    WIDTH,
    DataSchema,
    ModelConfig,
    NormalizationMetadata,
    canonical_model_variant,
)


class SchemaTests(unittest.TestCase):
    def test_production_constants_and_schema(self):
        self.assertEqual(WIDTH, 400)
        self.assertEqual(HEIGHT, 400)
        self.assertEqual(FIELDS, ("density", "velocity_x", "velocity_y", "obstacle_mask"))
        self.assertEqual(DTYPE.__name__, "float32")
        schema = DataSchema()
        self.assertEqual(schema.frame_bytes, 4 * 400 * 400 * 4)
        self.assertEqual(schema.model_variant, DEFAULT_MODEL_VARIANT)
        self.assertEqual(schema.output_channels, 1)

    def test_derived_model_dimensions_and_variants(self):
        config = ModelConfig(width=20, height=24, model_variant="density_velocity", latent_dim=12)
        self.assertEqual(config.bottleneck_size, (3, 3))
        self.assertEqual(config.output_channels, 3)
        self.assertEqual(canonical_model_variant("density+velocity"), "density_velocity")
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)

    def test_normalization_metadata_round_trip(self):
        metadata = NormalizationMetadata.from_maxima(0.2, 7.5)
        self.assertAlmostEqual(metadata.density_scale, 5.0)
        self.assertAlmostEqual(metadata.velocity_scale, 1.0 / 7.5)
        self.assertEqual(NormalizationMetadata.from_dict(metadata.to_dict()), metadata)

    def test_invalid_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            DataSchema(fields=("density",))
        with self.assertRaises(ValueError):
            DataSchema(dtype="float64")
        with self.assertRaises(ValueError):
            ModelConfig(model_variant="unknown")


if __name__ == "__main__":
    unittest.main()
