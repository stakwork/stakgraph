#!/usr/bin/env bash
set -eu

##############################################################################
# Stakgraph CLI Install Script
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/stakwork/stakgraph/refs/heads/main/install.sh | bash
#
# Environment variables:
#   STAKGRAPH_VERSION - Version to install (default: latest)
#   STAKGRAPH_BIN_DIR - Install directory (default: $HOME/.local/bin)
##############################################################################

# Check dependencies
if ! command -v curl >/dev/null 2>&1; then
  echo "Error: 'curl' is required"
  exit 1
fi

# Variables
REPO="stakwork/stakgraph"
VERSION="${STAKGRAPH_VERSION:-latest}"
INSTALL_DIR="${STAKGRAPH_BIN_DIR:-$HOME/.local/bin}"

# Detect OS/Architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux) ;;
  darwin) ;;
  mingw*|msys*|cygwin*) OS="windows" ;;
  *)
    echo "Error: Unsupported OS '$OS'"
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64) ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *)
    echo "Error: Unsupported architecture '$ARCH'"
    exit 1
    ;;
esac

# Build download URL and file info
if [ "$OS" = "darwin" ]; then
  TARGET="$ARCH-apple-darwin"
  FILE="stakgraph-$TARGET.tar.gz"
  BINARY="stakgraph"
elif [ "$OS" = "windows" ]; then
  TARGET="x86_64-pc-windows-msvc"
  FILE="stakgraph-$TARGET.zip"
  BINARY="stakgraph.exe"
else
  TARGET="$ARCH-unknown-linux-musl"
  FILE="stakgraph-$TARGET.tar.gz"
  BINARY="stakgraph"
fi

if [ "$VERSION" = "latest" ]; then
  URL="https://github.com/$REPO/releases/latest/download/$FILE"
else
  URL="https://github.com/$REPO/releases/download/$VERSION/$FILE"
fi

# Download and extract
curl -sLf "$URL" -o "$FILE" || { echo "Error: Download failed"; exit 1; }

# echo "Extracting..."
if [ "$OS" = "windows" ]; then
  unzip -q "$FILE"
else
  tar -xzf "$FILE"
fi

# Install
mkdir -p "$INSTALL_DIR"
echo "Installing to $INSTALL_DIR/$BINARY"
mv "$BINARY" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$BINARY"
rm -f "$FILE"

echo "✓ stakgraph installed!"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo ""
  echo "Add to your PATH:"
  echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi