#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[security] Running dependency vulnerability audit..."
uv run --locked --no-default-groups --group security pip-audit

echo "[security] Generating SBOM at ./sbom.json..."
uv run --locked --no-default-groups --group security \
  cyclonedx-py environment --output-file sbom.json --of JSON

echo "[security] Supply-chain checks complete."
