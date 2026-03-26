#!/bin/sh
set -e

REPO_URL="https://github.com/fayekelmith/task-manager-example"
REPO_DIR="$(mktemp -d)/task-manager-example"
CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Cloning example repo..."
git clone --quiet "$REPO_URL" "$REPO_DIR"

echo ""
echo "==> Directory summary (with file prioritization)"
$CARGO "$REPO_DIR/src/"

echo ""
echo "==> Single file parse"
$CARGO "$REPO_DIR/src/handlers/create.rs"

echo ""
echo "==> Stats table"
$CARGO "$REPO_DIR/src/" --stats

echo ""
echo "==> Named node lookup: validate_task"
$CARGO "$REPO_DIR/src/" --name validate_task

rm -rf "$(dirname "$REPO_DIR")"
