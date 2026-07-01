#!/usr/bin/env python3
"""
Render static preview shots of the robot standing in the sim.

Needs a GPU with enough VRAM to open an Isaac Sim viewport/camera (the
ray-tracing TLAS alone needs ~7.5 GB) — this reliably segfaults on the local
RTX 3050 Ti 4 GB laptop even fully headless, the same reason train_rl.py and
watch_rl.py only ever run --num_envs 32 locally without cameras. Run this on
the RunPod cloud GPU instead, the same way as watch_rl.py:

    ~/IsaacLab/isaaclab.sh -p scripts/render_robot_preview.py --headless

Then copy the PNGs back, e.g.:
    scp -P <port> root@<pod-ip>:/workspace/Quadruped/renders/*.png .
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render static preview shots of the quadruped")
parser.add_argument("--output_dir", type=str, default="renders",
                     help="Directory to write PNGs to (relative to repo root)")
parser.add_argument("--settle_steps", type=int, default=40,
                     help="Sim steps to hold the standing pose before capturing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import src.simulation.isaac_lab  # noqa: F401,E402
from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg  # noqa: E402

# (eye, lookat) — tuned for the ~0.4 x 0.2 x 0.08 m body at standing height.
_VIEWS = {
    "three_quarter": ((1.1, 1.1, 0.75), (0.0, 0.0, 0.28)),
    "side":          ((0.0, 1.15, 0.35), (0.0, 0.0, 0.28)),
    "front_closeup": ((0.55, 0.02, 0.32), (0.0, 0.0, 0.28)),
    "leg_closeup":   ((0.35, 0.35, 0.15), (0.183, 0.10, 0.05)),
}


def main() -> None:
    cfg = QuadrupedFlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.viewer.resolution = (1600, 900)

    env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array")
    env.reset()

    zero_action = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(args_cli.settle_steps):
        env.step(zero_action)

    out_dir = os.path.join(REPO_ROOT, args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for name, (eye, lookat) in _VIEWS.items():
        env.sim.set_camera_view(eye=eye, target=lookat)
        for _ in range(3):
            env.sim.render()
        frame = env.unwrapped.render()
        path = os.path.join(out_dir, f"robot_{name}.png")
        Image.fromarray(frame).save(path)
        print(f"[render_robot_preview] saved {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
