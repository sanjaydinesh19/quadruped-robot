#!/usr/bin/env python3
"""
Live MJPEG preview of the currently-training policy — no per-checkpoint
record/rename/serve/download cycle needed.

Runs as its own process alongside train_rl.py. Polls logs/rsl_rl/ for the
newest checkpoint, hot-reloads it, and streams the rollout as MJPEG over
plain HTTP so you can watch gait quality change live in a browser tab.

Isaac Sim's native livestream (WebRTC) needs a UDP media channel, and RunPod
pods don't forward UDP — that path just won't connect on a RunPod pod. MJPEG
over HTTP only needs TCP, so it works through the same "Expose HTTP Ports"
proxy already used for TensorBoard.

Usage (cloud, alongside a running train_rl.py):
    ~/IsaacLab/isaaclab.sh -p scripts/watch_rl.py --headless --enable_cameras

Then expose the port (default 6007) the same way as TensorBoard's 6006 and
open:
    https://<pod-id>-6007.proxy.runpod.net/
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live-stream the current training policy")
parser.add_argument("--port", type=int, default=6007,
                     help="HTTP port to serve the MJPEG stream on")
parser.add_argument("--poll_interval", type=float, default=30.0,
                     help="Seconds between checks for a newer checkpoint")
parser.add_argument("--fps", type=float, default=15.0,
                     help="Stream frame rate (lower = less bandwidth)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# MJPEG streaming needs the offscreen renderer, and this process is meant to
# run headless alongside the (also headless) training process.
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.simulation.isaac_lab  # noqa: F401 — triggers gym.register side-effect

from src.simulation.isaac_lab.quadruped_env_cfg import QuadrupedFlatEnvCfg
from src.simulation.isaac_lab.agents.rsl_rl_ppo_cfg import QuadrupedPPORunnerCfg
from src.simulation.isaac_lab.rsl_rl_adapter import build_runner_cfg_dict, latest_checkpoint

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "rsl_rl")

# ── Build a single-env viewer environment ────────────────────────────────────
env_cfg = QuadrupedFlatEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 60.0

env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
env = RslRlVecEnvWrapper(env)

runner_cfg_dict = build_runner_cfg_dict(QuadrupedPPORunnerCfg())
runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=LOG_DIR, device="cuda:0")

# ── Shared state between the sim loop (main thread) and the HTTP server threads ──
_state: dict = {"policy": None, "checkpoint": None, "frame": None}
_lock = threading.Lock()


def _reload_if_newer() -> None:
    path = latest_checkpoint(LOG_DIR)
    if path is None or path == _state["checkpoint"]:
        return
    runner.load(path)
    _state["policy"] = runner.get_inference_policy(device=env.device)
    _state["checkpoint"] = path
    print(f"[watch_rl] now watching {os.path.basename(path)}")


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass  # keep stdout free of per-frame request logs

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _lock:
                    frame = _state["frame"]
                if frame is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                time.sleep(1.0 / args_cli.fps)
        except (BrokenPipeError, ConnectionResetError):
            pass


server = ThreadingHTTPServer(("0.0.0.0", args_cli.port), _MJPEGHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"[watch_rl] streaming on http://0.0.0.0:{args_cli.port}/ — "
      f"expose this port on RunPod the same way as TensorBoard's 6006")
print(f"[watch_rl] watching {LOG_DIR} for checkpoints (poll every {args_cli.poll_interval:.0f}s)")

# ── Main sim loop — must stay on the main thread for Isaac Sim ──────────────
obs, _ = env.reset()
last_poll = 0.0
last_frame_t = 0.0
frame_period = 1.0 / args_cli.fps

while simulation_app.is_running():
    now = time.time()
    if now - last_poll >= args_cli.poll_interval:
        _reload_if_newer()
        last_poll = now

    if _state["policy"] is None:
        time.sleep(0.5)
        continue

    with torch.inference_mode():
        actions = _state["policy"](obs)
    obs, _, _, _ = env.step(actions)

    if now - last_frame_t >= frame_period:
        frame = env.unwrapped.render()
        if frame is not None:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=80)
            with _lock:
                _state["frame"] = buf.getvalue()
        last_frame_t = now

env.close()
simulation_app.close()
