#!/usr/bin/env python3
"""
Load a trained checkpoint and run the quadruped policy in the viewer.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/play_rl.py --checkpoint logs/rsl_rl/<run>/model_3000.pt

The viewer lets you watch the robot walk. Use the mouse to orbit the camera.
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained quadruped policy")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint file")
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel robots to display")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force viewer on (never headless for play)
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.simulation.isaac_lab  # noqa: F401

from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg
from src.simulation.isaac_lab.agents.rsl_rl_ppo_cfg import QuadrupedPPORunnerCfg

# Build env with viewer
env_cfg = QuadrupedFlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 60.0   # longer episodes for watching

env = ManagerBasedRLEnv(cfg=env_cfg)

# Load policy
runner_cfg = QuadrupedPPORunnerCfg()
log_dir = os.path.dirname(args_cli.checkpoint)
runner = RslRlOnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.device)

# Run
obs, _ = env.reset()
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, _, _, _ = env.step(actions)

env.close()
simulation_app.close()
