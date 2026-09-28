#!/bin/sh
# Thin wrapper. All logic lives in pls_compression/pipeline.py, reached through
# active_train_parallel.py. Use that directly if you prefer not to have a shell
# in the path.
#
#   ./active_train_parallel.sh generate [options]
#   ./active_train_parallel.sh train    [options]
#   ./active_train_parallel.sh all      [options]
#
# Run with --help for the options and the environment variables they read.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$SCRIPT_DIR/active_train_parallel.py" "$@"
