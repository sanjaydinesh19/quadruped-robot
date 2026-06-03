# Quadruped Robot

A quadruped (4-legged) robot project built simulation-first — kinematics, dynamics, and motion control validated in software before any hardware is touched.

## Stack

| Layer | Tool |
|---|---|
| Simulation | NVIDIA Isaac Sim, Gazebo Harmonic |
| Middleware | ROS2 Jazzy |
| Language | Python 3.12 |
| Dev tooling | Claude Code + MCP server |

## Repository Layout

```
src/
  kinematics/     # Forward / inverse kinematics
  dynamics/       # Rigid-body dynamics, contact models
  control/        # Gait planners, whole-body control
  simulation/     # Isaac Sim & Gazebo integrations
  ros2/           # ROS2 packages (URDF, controllers, bringup)
config/           # Robot parameters, joint limits, gains
mcp_server/       # MCP server exposing sim tools to Claude Code
tests/            # Unit and integration tests
docs/             # Architecture and design notes
```

## Quickstart

```bash
# Install Python dependencies
pip install -e ".[dev]"

# Run unit tests (no simulator required)
pytest tests/unit/

# Start the MCP server (Claude Code integration)
python mcp_server/server.py
```

## Status

Early stage — simulation scaffold in progress. Hardware spec TBD.
