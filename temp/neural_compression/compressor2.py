import sys

from pls_compression.cli import main as _cli_main
from pls_compression.models import FullModel, build_model, get_coord_grid, load_checkpoint, save_checkpoint
from pls_compression.schema import DataSchema, ModelConfig, NormalizationMetadata
from pls_compression.training import TrainingConfig, train, train_model


def main(argv=None):
    values = [] if argv is None else list(argv)
    return _cli_main(["train", *values], default_variant="density_velocity")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
