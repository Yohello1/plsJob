import sys

from pls_compression.orchestration import main as _main


def main(argv=None):
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
