#!/bin/sh
set -e

CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Directory summary (with file prioritization)"
$CARGO src/

echo ""
echo "==> Single file parse"
$CARGO src/handlers/create.rs

echo ""
echo "==> Stats table"
$CARGO src/ --stats

echo ""
echo "==> Named node lookup: validate_task"
$CARGO src/ --name validate_task
