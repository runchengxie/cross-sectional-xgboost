#!/usr/bin/env bash
set -euo pipefail

# Project root: parent directory of this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults (can be overridden by args)
NAME="${1:-cross-sectional-machine-learning}"
OUT_DIR="${2:-"$ROOT/.."}"

# Timestamped output to avoid overwriting and to keep history
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_PATH="${OUT_DIR}/${NAME}_${STAMP}.zip"

# Exclusion list (7z format)
EXCLUDE_FILE="${ROOT}/scripts/7z_exclusion_list.txt"

cd "$ROOT"

# Ensure 7z exists
command -v 7z >/dev/null 2>&1 || {
  echo "7z not found. Install it with:"
  echo "  sudo apt update && sudo apt install -y p7zip-full"
  exit 1
}

# Create ZIP in the parent directory so we don't accidentally include the ZIP itself.
# -xr@ reads a recursive exclude list from file
# -x! excludes the exclude list file itself (optional but tidy)
7z a -tzip -mx=9 "$ZIP_PATH" . \
  -xr@"$EXCLUDE_FILE" \
  -x!"scripts/7z_exclusion_list.txt"

# Basic integrity test of the archive
7z t "$ZIP_PATH" >/dev/null

# Write a checksum file for verification after transfer
sha256sum "$ZIP_PATH" | tee "${ZIP_PATH}.sha256" >/dev/null

echo "Created: $ZIP_PATH"
echo "SHA256 : ${ZIP_PATH}.sha256"
