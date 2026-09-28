import sys

from pls_compression.cli import main as _cli_main
from pls_compression.dataset import SPHDataset, compute_global_stats, discover_sessions
from pls_compression.losses import hybrid_loss
from pls_compression.metrics import compute_metrics
from pls_compression.models import FullModel, build_model, get_coord_grid, load_checkpoint, save_checkpoint
from pls_compression.schema import BUFFER_HEIGHT, BUFFER_WIDTH, FIELDS, LATENT_DIM, MODEL_VARIANT, WIDTH, HEIGHT, DataSchema, ModelConfig, NormalizationMetadata
from pls_compression.training import TrainingConfig, find_max_batch_size, get_global_stats, train, train_model


def main(argv=None):
    values = [] if argv is None else list(argv)
    return _cli_main(["train", *values], default_variant="density")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
