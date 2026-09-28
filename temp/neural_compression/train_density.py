import sys

from pls_compression.cli import main as _cli_main


def main(argv=None):
    values = [] if argv is None else list(argv)
    return _cli_main(["train", *values], default_variant="density")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
