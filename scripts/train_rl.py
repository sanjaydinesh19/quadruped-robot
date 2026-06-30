#!/usr/bin/env python3
"""
Quadruped locomotion RL training script (RSL-RL / PPO).

MUST be launched via Isaac Lab's Python wrapper so that Isaac Sim
is initialised before any omni.* imports happen:

    ~/IsaacLab/isaaclab.sh -p scripts/train_rl.py [options]

Options
-------
    --num_envs  N      Number of parallel environments (default: 32 for RTX 3050)
                       Use 2048–4096 on a cloud A100 for full-speed training.
    --headless         Run without the viewport (faster; use for cloud training)
    --resume           Resume from the latest checkpoint in logs/rsl_rl/
    --max_iterations N Override training length (default from PPO config: 3000)

Examples
--------
    # Local development (3050 4GB, with viewer)
    ~/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 32

    # Cloud / headless full training run
    ~/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 2048 --headless
"""
from __future__ import annotations

import argparse
import sys
import os

# ── Step 1: parse args + launch Isaac Sim BEFORE any omni.* imports ──────────
# AppLauncher must be the very first thing that touches the Omniverse runtime.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quadruped locomotion RL training")
parser.add_argument("--num_envs",       type=int,  default=32)
parser.add_argument("--resume",         action="store_true")
parser.add_argument("--max_iterations", type=int,  default=None)
# NOTE: RTX 3050 Ti (4 GB VRAM) cannot run the Isaac Sim viewport — the ray-tracing
# TLAS alone needs ~7.5 GB. Always pass --headless on this machine.
# Remove --headless when running on a cloud GPU with 8 GB+ VRAM.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Step 2: all other imports (safe after Omniverse is up) ───────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Register our gymnasium environment and get the config
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.simulation.isaac_lab  # noqa: F401 — triggers gym.register side-effect

from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg
from src.simulation.isaac_lab.agents.rsl_rl_ppo_cfg import QuadrupedPPORunnerCfg

# ── Step 3: build environment ─────────────────────────────────────────────────
env_cfg = QuadrupedFlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs

env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# ── Step 4: configure runner ──────────────────────────────────────────────────
runner_cfg = QuadrupedPPORunnerCfg()
if args_cli.max_iterations is not None:
    runner_cfg.max_iterations = args_cli.max_iterations

log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "rsl_rl")
os.makedirs(log_dir, exist_ok=True)

runner = OnPolicyRunner(
    env,
    runner_cfg.to_dict(),
    log_dir=log_dir,
    device="cuda:0",
)

# ── Step 5: optionally resume ─────────────────────────────────────────────────
if args_cli.resume:
    # Loads the latest checkpoint from log_dir automatically
    runner.load(log_dir)
    print(f"[train_rl] Resumed from {log_dir}")

# ── Step 6: train ─────────────────────────────────────────────────────────────
print(f"[train_rl] Starting training: {runner_cfg.max_iterations} iterations, "
      f"{args_cli.num_envs} envs, device=cuda:0")
print(f"[train_rl] Logs → {log_dir}")

runner.learn(
    num_learning_iterations=runner_cfg.max_iterations,
    init_at_random_ep_len=True,
)

# ── Cleanup ───────────────────────────────────────────────────────────────────
env.close()
simulation_app.close()
