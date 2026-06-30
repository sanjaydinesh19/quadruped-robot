# Quadruped Robot

A 5 kg, 12-DOF quadruped robot built simulation-first using NVIDIA Isaac Lab.
The goal is full RL-trained locomotion in simulation before any hardware is touched.

## Robot Specification

| Parameter | Value |
|---|---|
| Body mass | ~5 kg total |
| DOF | 12 (3 per leg: hip abduction, thigh, knee) |
| Actuators | Quasi-Direct Drive (QDD) |
| Use case | Terrain inspection |
| High-level compute | Raspberry Pi 5 8 GB |
| Real-time compute | Teensy 4.1 @ 1 kHz |
| Depth sensor | Intel RealSense D435i |
| Foot contact | FSR sensors ×4 |

## Stack

| Layer | Tool |
|---|---|
| RL training | NVIDIA Isaac Lab (RSL-RL / PPO) |
| Physics sim | NVIDIA Isaac Sim (PhysX GPU) |
| Visualisation | RViz2 |
| Middleware | ROS2 Jazzy |
| Language | Python 3.12 / 3.10 (Isaac Lab env) |
| Dev tooling | Claude Code + MCP server |

## Repository Layout

```
config/
  robot_params.yaml          # single source of truth — all geometry, mass, joint limits

src/
  kinematics/
    leg.py                   # analytical FK + IK for a single 3-DOF leg
  simulation/
    isaac_lab/
      quadruped_env_cfg.py   # full ManagerBasedRLEnvCfg (scene, obs, rewards, events)
      agents/
        rsl_rl_ppo_cfg.py    # PPO hyperparameters (RSL-RL)
  ros2/
    quadruped_description/   # ROS2 package: URDF, RViz2 launch, display config

assets/
  quadruped/                 # generated USD asset (output of convert_to_usd.sh)

scripts/
  generate_urdf.py           # reads robot_params.yaml → writes quadruped.urdf
  convert_to_usd.sh          # URDF → USD for Isaac Lab
  train_rl.py                # PPO training entry point
  play_rl.py                 # load checkpoint + run policy in viewer

tests/
  unit/                      # pytest unit tests (no simulator required)
```

## Quickstart

### 1. Python dependencies

```bash
pip install -e ".[dev]"
```

### 2. Generate the URDF

Edit `config/robot_params.yaml` to change any physical parameter, then regenerate:

```bash
python scripts/generate_urdf.py
```

### 3. Visualise in RViz2

```bash
# Install ROS2 tools (once)
sudo apt install ros-jazzy-joint-state-publisher ros-jazzy-joint-state-publisher-gui

colcon build --packages-select quadruped_description --symlink-install
source install/setup.bash
ros2 launch quadruped_description display.launch.py
```

Use the joint slider GUI to manually drive all 12 joints.

### 4. Install Isaac Lab (one-time, ~20 min)

Isaac Lab requires Python 3.10 via conda:

```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
cd ~/Projects/IsaacLab
./isaaclab.sh --install

# Point our scripts at your Isaac Lab location
echo 'export ISAACLAB_PATH=~/Projects/IsaacLab' >> ~/.bashrc
source ~/.bashrc
```

### 5. Convert URDF → USD

```bash
conda activate isaaclab
./scripts/convert_to_usd.sh
# Output: assets/quadruped/quadruped.usd
```

### 6. Train

```bash
# Local RTX 3050 4 GB (development)
~/Projects/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 32

# Cloud A100 (full training run, headless)
~/Projects/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 2048 --headless

# Resume from checkpoint
~/Projects/IsaacLab/isaaclab.sh -p scripts/train_rl.py --num_envs 32 --resume
```

Logs and checkpoints are saved to `logs/rsl_rl/`.

### 7. Play a trained policy

```bash
~/Projects/IsaacLab/isaaclab.sh -p scripts/play_rl.py \
  --checkpoint logs/rsl_rl/quadruped_flat/<run>/model_3000.pt
```

### 8. Unit tests

```bash
pytest tests/unit/    # no simulator required
```

## RL Environment

| Property | Value |
|---|---|
| Observation space | 48-dim (proprioceptive only) |
| Action space | 12-dim joint position offsets |
| Physics rate | 200 Hz |
| Policy rate | 50 Hz |
| Episode length | 20 s |
| Default parallel envs | 32 (local) / 2048 (cloud A100) |

**Reward shaping:** primary objective is tracking commanded (vx, vy, ωz) velocity. Penalties discourage bouncing, high energy use, shin/thigh ground contacts, and falling.

**Domain randomisation:** floor friction, base mass ±0.5 kg, random episode reset, random pushes every 10–15 s.

## Status

| Component | State |
|---|---|
| Robot spec | Locked |
| Parametric URDF | Done |
| FK / IK | Done, tested |
| RViz2 visualisation | Done |
| Isaac Lab env (flat terrain) | Done |
| USD asset | Pending (needs Isaac Lab install) |
| RL training (flat terrain) | Pending |
| Rough terrain curriculum | Not started |
| ROS2 controllers | Not started |
| Hardware bring-up | Not started |
