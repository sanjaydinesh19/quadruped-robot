#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Convert the quadruped URDF to a USD asset consumable by Isaac Lab.
#
# Prerequisites:
#   1. Isaac Lab installed to ~/IsaacLab  (or set ISAACLAB_PATH)
#   2. conda env 'isaaclab' active
#   3. URDF already generated:  python scripts/generate_urdf.py
#
# Usage:
#   ./scripts/convert_to_usd.sh
#
# Output:
#   assets/quadruped/quadruped.usd   ← referenced by quadruped_env_cfg.py
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
ISAACLAB="$ISAACLAB_PATH/isaaclab.sh"

URDF="$REPO_ROOT/src/ros2/quadruped_description/urdf/quadruped.urdf"
OUTPUT_DIR="$REPO_ROOT/assets/quadruped"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$ISAACLAB" ]; then
  echo "[ERROR] Isaac Lab not found at $ISAACLAB_PATH"
  echo "        Install it first (see README or scripts/install_isaaclab.md)"
  echo "        or set:  export ISAACLAB_PATH=/path/to/IsaacLab"
  exit 1
fi

if [ ! -f "$URDF" ]; then
  echo "[ERROR] URDF not found: $URDF"
  echo "        Generate it first:  python scripts/generate_urdf.py"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[convert_to_usd] URDF  : $URDF"
echo "[convert_to_usd] Output: $OUTPUT_DIR/quadruped.usd"
echo "[convert_to_usd] Running Isaac Lab URDF importer (headless)..."

# --merge-joints: fuses the four fixed foot joints into their shin links,
#   reducing articulation complexity without losing collision geometry.
#   (Isaac Lab 4.5 renamed --merge-fixed-joints → --merge-joints)
"$ISAACLAB" -p "$ISAACLAB_PATH/scripts/tools/convert_urdf.py" \
  "$URDF" \
  "$OUTPUT_DIR" \
  --merge-joints \
  --headless

echo "[convert_to_usd] Done → $OUTPUT_DIR/quadruped.usd"
