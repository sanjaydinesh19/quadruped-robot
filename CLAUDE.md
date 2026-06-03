# Quadruped Robot — Claude Code Project Guide

## Project Overview
A quadruped (4-legged) robot project targeting full simulation before hardware.
Stack: Python 3.12, ROS2 Jazzy, Isaac Sim, Gazebo Harmonic, MCP server for Claude tooling.

## Repository Layout
```
src/
  kinematics/        # Forward/inverse kinematics (analytical + numerical)
  dynamics/          # Rigid-body dynamics, contact models
  control/
    gait/            # Gait planners (trot, walk, bound, stand)
    balance/         # Whole-body control, balance recovery
  simulation/
    isaac_sim/       # NVIDIA Isaac Sim integration
    gazebo/          # Gazebo Harmonic integration
  ros2/
    quadruped_description/   # URDF / meshes
    quadruped_bringup/       # Launch files
    quadruped_control/       # ROS2 controllers
config/              # Robot parameters, joint limits, gains
tests/               # pytest unit + integration tests
mcp_server/          # MCP server exposing sim/robot tools to Claude
docs/                # Architecture, design decisions
scripts/             # Dev utilities
```

## Conventions
- Python: PEP 8, type hints everywhere, no implicit Any
- All physical quantities carry units in variable names: `pos_m`, `vel_mps`, `angle_rad`
- Coordinate frame: body frame = robot's torso, world frame = gravity-aligned NED
- Joint numbering: FL=0, FR=1, RL=2, RR=3 (Front-Left, Front-Right, Rear-Left, Rear-Right)
- Each leg: hip (abduction), thigh (flexion), knee (flexion) — 3 DOF per leg, 12 DOF total

## Key Entry Points
- `scripts/run_sim.sh` — launch Gazebo or Isaac Sim
- `mcp_server/server.py` — MCP server (Claude tooling)
- `src/kinematics/leg.py` — single-leg FK/IK
- `src/control/gait/trot.py` — trot gait planner

## Testing
```bash
pytest tests/unit/         # fast, no sim required
pytest tests/integration/  # requires Gazebo or Isaac Sim running
```

## ROS2 Workspace
```bash
colcon build --symlink-install
source install/setup.bash
```

## MCP Server
```bash
cd mcp_server && python server.py
```
Exposes tools: `compute_ik`, `compute_fk`, `launch_sim`, `get_robot_state`.
