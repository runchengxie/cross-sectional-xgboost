#!/usr/bin/env bash
set -euo pipefail

# Project root: parent directory of this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults (can be overridden by args)
DEFAULT_NAME="$(basename "$ROOT")"
NAME="${1:-$DEFAULT_NAME}"
OUT_DIR="${2:-"$ROOT/.."}"

# Timestamped output to avoid overwriting and to keep history
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_PATH="${OUT_DIR}/${NAME}_${STAMP}.zip"

# Built-in exclusions (migrated from project_tools/7z_exclusion_list.txt)
EXCLUDES=(
  "__pycache__"
  ".pytest_cache"
  "*.pyc"
  "*.pyo"
  "*.pyd"
  ".coverage"
  "htmlcov"
  ".venv"
  ".git"
)

cd "$ROOT"

# Ensure 7z exists
command -v 7z >/dev/null 2>&1 || {
  echo "7z not found. Install it with:"
  echo "  sudo apt update && sudo apt install -y p7zip-full"
  exit 1
}

# Ensure output directory exists and is writable
mkdir -p "$OUT_DIR"
if [[ ! -w "$OUT_DIR" ]]; then
  echo "Output directory is not writable: $OUT_DIR"
  exit 1
fi

# Create ZIP in the parent directory so we don't accidentally include the ZIP itself.
# -xr! applies recursive exclusion patterns
SEVEN_Z_ARGS=(a -tzip -mx=9 "$ZIP_PATH" .)
for pattern in "${EXCLUDES[@]}"; do
  SEVEN_Z_ARGS+=("-xr!${pattern}")
done
7z "${SEVEN_Z_ARGS[@]}"

# Basic integrity test of the archive
7z t "$ZIP_PATH" >/dev/null

# Write a checksum file for verification after transfer
sha256sum "$ZIP_PATH" | tee "${ZIP_PATH}.sha256" >/dev/null

echo "Created: $ZIP_PATH"
echo "SHA256 : ${ZIP_PATH}.sha256"
