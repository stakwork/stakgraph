#!/bin/sh
set -e

REPO_URL="https://github.com/fayekelmith/task-manager-example"
REPO_DIR="$(mktemp -d)/task-manager-example"
CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Cloning example repo..."
git clone --quiet "$REPO_URL" "$REPO_DIR"

echo ""
echo "==> Dependency tree from entry point: handle_request"
$CARGO deps handle_request "$REPO_DIR/src/"

rm -rf "$(dirname "$REPO_DIR")"
