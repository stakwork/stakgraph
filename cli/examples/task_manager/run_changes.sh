#!/bin/sh
set -e

# The `changes` commands require a real git repo to inspect.
# We use a dedicated example repo with a stable two-commit history,
# similar to how standalone/tests/graph_updates.rs pins before_commit/after_commit.
#
# Repo:   https://github.com/fayekelmith/task-manager-example
# commit1 (before): 29d6de9146861faf1329679bc3b4435c714ebeb3  initial CRUD handlers
# commit2 (after):  a2f171e7fe383794083c82fed05c55d3ef6ca3a7  auth + db split + validator refactor
#
# Alternative: if you have a local clone already, set REPO_DIR to its path
# and skip the clone/cleanup steps below.

REPO_URL="https://github.com/fayekelmith/task-manager-example"
REPO_DIR="$(mktemp -d)/task-manager-example"
CARGO="cargo run --manifest-path ../../Cargo.toml --"

echo "==> Cloning example repo..."
git clone --quiet "$REPO_URL" "$REPO_DIR"

echo ""
echo "==> Commit history"
(cd "$REPO_DIR" && $CARGO changes list)

echo ""
echo "==> Graph delta for last commit"
(cd "$REPO_DIR" && $CARGO changes diff --last 1)

# Clean up the temporary clone
rm -rf "$(dirname "$REPO_DIR")"
