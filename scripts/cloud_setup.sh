#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# One-shot setup for quadruped RL training on RunPod.
#
# PREREQUISITE: start a RunPod pod using the container image
#   nvcr.io/nvidia/isaac-lab:2.3.2
# (NOT the raw Isaac Sim image — the isaac-lab image has compatible physx.fabric)
#
# Then SSH into the pod and run:
#   cd /workspace && git clone <your-repo-url> Quadruped
#   cd Quadruped && bash scripts/cloud_setup.sh
#
# What it does:
#   1. Clones Isaac Lab main → /workspace/isaaclab  (skips if already present)
#   2. Links /isaac-sim into the Isaac Lab tree
#   3. Installs Isaac Lab Python extensions into the Kit Python
#   4. Generates the URDF
#   5. Converts URDF → USD
#   6. Prints the training command
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"

echo "════════════════════════════════════════════════════════"
echo " Quadruped cloud setup  (isaac-lab:2.3.2 container)"
echo " Repo    : $REPO_ROOT"
echo " IsaacLab: $ISAACLAB_PATH"
echo "════════════════════════════════════════════════════════"

# ── 1. Clone Isaac Lab ────────────────────────────────────────────────────────
if [ -d "$ISAACLAB_PATH/.git" ]; then
  echo "[1/4] Isaac Lab already cloned — skipping"
else
  echo "[1/4] Cloning Isaac Lab (main)..."
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi

# ── 2. Link /isaac-sim into the Isaac Lab tree ───────────────────────────────
# The container ships Isaac Sim at /isaac-sim; Isaac Lab expects it at
# <isaaclab>/_isaac_sim.  A symlink is all that's needed.
if [ -e "$ISAACLAB_PATH/_isaac_sim" ]; then
  echo "[2/4] _isaac_sim symlink already exists — skipping"
else
  echo "[2/4] Linking /isaac-sim → $ISAACLAB_PATH/_isaac_sim"
  ln -s /isaac-sim "$ISAACLAB_PATH/_isaac_sim"
fi

# ── 3. Install Isaac Lab extensions into the Kit Python ──────────────────────
echo "[3/4] Installing Isaac Lab extensions (uses Kit Python, ~5 min)..."
cd "$ISAACLAB_PATH"
./isaaclab.sh --install

# ── 4. Generate URDF + convert to USD ────────────────────────────────────────
cd "$REPO_ROOT"

echo "[4/4a] Generating URDF..."
"$ISAACLAB_PATH/_isaac_sim/python.sh" scripts/generate_urdf.py

echo "[4/4b] Converting URDF → USD (Isaac Sim starts headless, ~3 min)..."
export ISAACLAB_PATH
bash scripts/convert_to_usd.sh

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo " Start training:"
echo "   cd $REPO_ROOT"
echo "   $ISAACLAB_PATH/isaaclab.sh -p scripts/train_rl.py \\"
echo "     --num_envs 2048 --livestream 1"
echo ""
echo " Visualise (RunPod HTTP proxy — no SSH tunnel needed):"
echo "   https://<pod-id>-49100.proxy.runpod.net/streaming/client"
echo "════════════════════════════════════════════════════════"
