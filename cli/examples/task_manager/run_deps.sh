#!/bin/sh
set -e

CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Dependency tree from entry point: handle_request"
$CARGO deps handle_request src/
