import sys

from pls_compression.cli import main as _cli_main


def sanity_check(argv=None):
    values = [] if argv is None else list(argv)
    return _cli_main(["check", *values])


def check_model(argv=None):
    return sanity_check(argv)


def main(argv=None):
    return sanity_check(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
