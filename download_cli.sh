#!/usr/bin/env bash
set -eu

##############################################################################
# Stakgraph CLI Install Script
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/YOUR_REPO/main/install.sh | bash
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
REPO="YOUR_ORG/YOUR_REPO"  # Update this!
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
  TARGET="x86_64-pc-windows-gnu"
  FILE="stakgraph-$TARGET.zip"
  BINARY="stakgraph.exe"
else
  TARGET="$ARCH-unknown-linux-gnu"
  FILE="stakgraph-$TARGET.tar.gz"
  BINARY="stakgraph"
fi

URL="https://github.com/$REPO/releases/${VERSION/#latest/latest/download}${VERSION/#latest//download/$VERSION}/$FILE"

# Download and extract
echo "Downloading $FILE..."
curl -sLf "$URL" -o "$FILE" || { echo "Error: Download failed"; exit 1; }

echo "Extracting..."
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

echo ""
echo "✓ Stakgraph installed!"

# Check PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo ""
  echo "Add to your PATH:"
  echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi