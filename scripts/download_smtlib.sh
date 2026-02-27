#!/usr/bin/env bash
# Download, decompress, extract, and clean up one SMT-LIB 25 logic division from Zenodo.
# Usage: ./download_smtlib25_logic.sh <LOGIC_NAME>
# Example: ./download_smtlib25_logic.sh QF_NIA

set -e

ZENODO_RECORD="${ZENODO_RECORD:-15493090}"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../smtlib"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <LOGIC_NAME>" >&2
  echo "Example: $0 QF_NIA" >&2
  exit 1
fi

LOGIC="$1"
ARCHIVE="${LOGIC}.tar.zst"
TARFILE="${LOGIC}.tar"
DOWNLOAD_URL="${BASE_URL}/${ARCHIVE}?download=1"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading ${ARCHIVE}..."
wget -O "$ARCHIVE" "$DOWNLOAD_URL"

echo "Decompressing..."
if command -v zstd &>/dev/null; then
  zstd -df "$ARCHIVE" -o "$TARFILE"
else
  # Fallback: use Python zstandard (pip install zstandard)
  PYTHON="${PYTHON:-python3}"
  if [[ -x "${SCRIPT_DIR}/../.venv/bin/python3" ]]; then
    PYTHON="${SCRIPT_DIR}/../.venv/bin/python3"
  fi
  if ! "$PYTHON" -c "
import zstandard as zstd
with open('$ARCHIVE', 'rb') as f_in, open('$TARFILE', 'wb') as f_out:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(f_in, f_out)
" 2>/dev/null; then
    echo "Error: zstd not found and Python zstandard not available." >&2
    echo "Install one of: (1) zstd binary, or (2) pip install zstandard" >&2
    exit 1
  fi
fi

echo "Extracting..."
tar -xf "$TARFILE" --no-same-owner

echo "Removing archives to free space..."
rm -f "$TARFILE" "$ARCHIVE"

echo "Done. Benchmarks are under ${TARGET_DIR}/"
