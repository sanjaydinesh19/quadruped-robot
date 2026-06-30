#!/usr/bin/env python3
"""
Load a trained checkpoint and run the quadruped policy.

Viewer mode (local, needs display):
    ~/IsaacLab/isaaclab.sh -p scripts/play_rl.py \
        --checkpoint logs/rsl_rl/<run>/model_3000.pt

Video mode (cloud / headless — saves MP4):
    ~/IsaacLab/isaaclab.sh -p scripts/play_rl.py \
        --checkpoint logs/rsl_rl/<run>/model_3000.pt \
        --headless --video --video_length 500

Download the video:
    scp <user>@ssh.runpod.io:/workspace/Quadruped/videos/*.mp4 .
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
parser.add_argument("--video", action="store_true",
                    help="Record video to file (enables headless rendering)")
parser.add_argument("--video_length", type=int, default=500,
                    help="Number of steps to record (default: 500 ≈ 10 s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Viewer is only useful without --video; --headless must be set by the user
# when running on a remote pod with --video.
if not args_cli.video:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.simulation.isaac_lab  # noqa: F401

from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg
from src.simulation.isaac_lab.agents.rsl_rl_ppo_cfg import QuadrupedPPORunnerCfg

# ── Build env ─────────────────────────────────────────────────────────────────
env_cfg = QuadrupedFlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 60.0

env = ManagerBasedRLEnv(cfg=env_cfg)

if args_cli.video:
    video_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    print(f"[play_rl] Recording {args_cli.video_length} steps → {video_dir}/")

# ── Load policy (same API adapter as train_rl.py) ────────────────────────────
runner_cfg = QuadrupedPPORunnerCfg()
runner_cfg_dict = runner_cfg.to_dict()

_actor  = runner_cfg_dict.get("actor")  or {}
_critic = runner_cfg_dict.get("critic") or {}
_dist   = _actor.get("distribution_cfg") or {}
runner_cfg_dict["policy"] = {
    "class_name":        "ActorCritic",
    "actor_hidden_dims":  _actor.get("hidden_dims", [512, 256, 128]),
    "critic_hidden_dims": _critic.get("hidden_dims", [512, 256, 128]),
    "activation":         _actor.get("activation", "elu"),
    "init_noise_std": (_dist.get("init_std") if isinstance(_dist, dict) else 1.0) or 1.0,
}
_VALID_ALG_KWARGS = {
    "class_name", "num_learning_epochs", "num_mini_batches", "learning_rate",
    "schedule", "gamma", "lam", "entropy_coef", "desired_kl", "max_grad_norm",
    "value_loss_coef", "use_clipped_value_loss", "clip_param",
    "normalize_advantage_per_mini_batch",
}
if isinstance(runner_cfg_dict.get("algorithm"), dict):
    runner_cfg_dict["algorithm"] = {
        k: v for k, v in runner_cfg_dict["algorithm"].items()
        if k in _VALID_ALG_KWARGS
    }

log_dir = os.path.dirname(args_cli.checkpoint)
runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device="cuda:0")
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.device)

# ── Run ───────────────────────────────────────────────────────────────────────
obs, _ = env.reset()
step = 0
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, _, _, _ = env.step(actions)
    step += 1
    if args_cli.video and step >= args_cli.video_length:
        break

env.close()
simulation_app.close()
