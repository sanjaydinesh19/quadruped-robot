#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# One-shot cloud instance setup for quadruped RL training.
# Tested on RunPod PyTorch 2.8.0 (Ubuntu 24.04, CUDA 12.8).
#
# Run once on a fresh cloud instance:
#   bash scripts/cloud_setup.sh
#
# What it does:
#   1. Installs Miniconda
#   2. Creates 'isaaclab' conda env (Python 3.10)
#   3. Clones Isaac Lab → ~/IsaacLab
#   4. Installs Isaac Sim 4.5 + Isaac Lab extensions
#   5. Generates the URDF
#   6. Converts URDF → USD
#   7. Prints the training command
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="$HOME/IsaacLab"

echo "════════════════════════════════════════════════════════"
echo " Quadruped cloud setup"
echo " Repo   : $REPO_ROOT"
echo " Isaac  : $ISAACLAB_PATH"
echo "════════════════════════════════════════════════════════"

# ── 1. Miniconda ──────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "[1/6] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm /tmp/miniconda.sh
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda init bash
else
  echo "[1/6] Miniconda already installed — skipping"
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# ── 2. Conda env ──────────────────────────────────────────────────────────────
if conda env list | grep -qw "isaaclab"; then
  echo "[2/6] Conda env 'isaaclab' already exists — skipping"
else
  echo "[2/6] Creating conda env (Python 3.10)..."
  conda create -y -n isaaclab python=3.10
fi
conda activate isaaclab

# ── 3. Clone Isaac Lab ────────────────────────────────────────────────────────
if [ -d "$ISAACLAB_PATH" ]; then
  echo "[3/6] Isaac Lab already cloned — skipping"
else
  echo "[3/6] Cloning Isaac Lab..."
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi

# ── 4. Install Isaac Sim + Isaac Lab ─────────────────────────────────────────
if python -c "import isaacsim" &>/dev/null; then
  echo "[4/6] Isaac Sim already installed — skipping"
else
  echo "[4/6] Installing Isaac Sim 4.5.0 (~20 GB download, ~20 min)..."
  pip install "isaacsim[all]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
  echo "[4/6] Installing Isaac Lab extensions..."
  cd "$ISAACLAB_PATH" && ./isaaclab.sh --install
  conda activate isaaclab
fi

# ── 5. Generate URDF ──────────────────────────────────────────────────────────
echo "[5/6] Generating URDF..."
cd "$REPO_ROOT"
python scripts/generate_urdf.py

# ── 6. Convert URDF → USD ─────────────────────────────────────────────────────
echo "[6/6] Converting URDF → USD (Isaac Sim starts headless, ~3 min)..."
export ISAACLAB_PATH="$ISAACLAB_PATH"
bash scripts/convert_to_usd.sh

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo " Start training (with livestream viewport):"
echo "   ~/IsaacLab/isaaclab.sh -p scripts/train_rl.py \\"
echo "     --num_envs 2048 --livestream 1"
echo ""
echo " Then on your LOCAL machine, open an SSH tunnel:"
echo "   ssh root@213.192.2.68 -p 40177 -i ~/.ssh/id_ed25519 -L 49100:localhost:49100"
echo " And open: http://localhost:49100/streaming/client"
echo "════════════════════════════════════════════════════════"
