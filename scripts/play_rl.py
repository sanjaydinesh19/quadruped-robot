#!/usr/bin/env python3
"""
Load a trained checkpoint and run the quadruped policy.

Viewer mode (local, needs display):
    ~/IsaacLab/isaaclab.sh -p scripts/play_rl.py \
        --checkpoint logs/rsl_rl/model_3000.pt

Video mode (cloud / headless — saves MP4 then download with scp):
    ~/IsaacLab/isaaclab.sh -p scripts/play_rl.py \
        --checkpoint logs/rsl_rl/model_3000.pt \
        --headless --video --video_length 500

    scp <user>@ssh.runpod.io:/workspace/Quadruped/videos/play.mp4 .
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
                    help="Record video to file (use with --headless on cloud)")
parser.add_argument("--video_length", type=int, default=500,
                    help="Number of steps to record (default: 500 ≈ 10 s at 50 Hz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Video mode needs the offscreen renderer; viewer mode forces headless off.
if args_cli.video:
    args_cli.enable_cameras = True
else:
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.simulation.isaac_lab  # noqa: F401

from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg
from src.simulation.isaac_lab.agents.rsl_rl_ppo_cfg import QuadrupedPPORunnerCfg
from src.simulation.isaac_lab.chase_camera import update_chase_camera
from src.simulation.isaac_lab.rsl_rl_adapter import build_runner_cfg_dict

# ── Build env ─────────────────────────────────────────────────────────────────
env_cfg = QuadrupedFlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.episode_length_s = 60.0

render_mode = "rgb_array" if args_cli.video else None
env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
env = RslRlVecEnvWrapper(env)   # adds get_observations() that OnPolicyRunner expects

video_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
if args_cli.video:
    os.makedirs(video_dir, exist_ok=True)
    print(f"[play_rl] Will record {args_cli.video_length} steps → {video_dir}/play.mp4")

# ── Load policy (same API adapter as train_rl.py) ────────────────────────────
runner_cfg = QuadrupedPPORunnerCfg()
runner_cfg_dict = build_runner_cfg_dict(runner_cfg)

log_dir = os.path.dirname(args_cli.checkpoint)
runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=log_dir, device="cuda:0")
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.device)

# ── Run ───────────────────────────────────────────────────────────────────────
obs, _ = env.reset()   # (obs_dict TensorDict, info)
frames = []
step = 0

while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, _, _ = env.step(actions)   # RslRlVecEnvWrapper: (obs, rew, done, extras)

    if args_cli.video:
        update_chase_camera(env)   # follows env 0 — the first of --num_envs robots
        frame = env.unwrapped.render()   # bypass wrapper; ManagerBasedRLEnv returns (H,W,3)
        if frame is not None:
            frames.append(frame)

    step += 1
    if args_cli.video and step >= args_cli.video_length:
        break

# ── Save video ────────────────────────────────────────────────────────────────
if args_cli.video and frames:
    try:
        import imageio
        video_path = os.path.join(video_dir, "play.mp4")
        imageio.mimsave(video_path, frames, fps=50)
        print(f"[play_rl] Saved {len(frames)} frames → {video_path}")
    except ImportError:
        # Fall back to saving individual PNG frames if imageio is missing
        for i, f in enumerate(frames):
            from PIL import Image
            Image.fromarray(f).save(os.path.join(video_dir, f"frame_{i:05d}.png"))
        print(f"[play_rl] imageio not found — saved {len(frames)} PNGs to {video_dir}/")

env.close()
simulation_app.close()
