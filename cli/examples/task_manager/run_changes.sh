#!/bin/sh
set -e

CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Commit history"
$CARGO changes list

echo ""
echo "==> Graph delta for last commit"
$CARGO changes diff --last 1
