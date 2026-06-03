#!/usr/bin/env python3
"""
Quadruped MCP Server — exposes robot/sim tools to Claude Code.

Run:  python mcp_server/server.py
Then register in .claude/settings.json under mcpServers.
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import mcp.server.stdio
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: mcp package not found. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

server = Server("quadruped")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="compute_fk",
            description=(
                "Compute forward kinematics for one leg. "
                "Returns the foot position in body frame."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "leg": {
                        "type": "string",
                        "enum": ["FL", "FR", "RL", "RR"],
                        "description": "Leg identifier",
                    },
                    "joint_angles_rad": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[hip, thigh, knee] joint angles in radians",
                    },
                },
                "required": ["leg", "joint_angles_rad"],
            },
        ),
        Tool(
            name="compute_ik",
            description=(
                "Compute inverse kinematics for one leg. "
                "Returns joint angles given a desired foot position in body frame."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "leg": {
                        "type": "string",
                        "enum": ["FL", "FR", "RL", "RR"],
                    },
                    "foot_pos_m": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[x, y, z] foot position in body frame (metres)",
                    },
                },
                "required": ["leg", "foot_pos_m"],
            },
        ),
        Tool(
            name="launch_sim",
            description="Launch the Gazebo or Isaac Sim simulation environment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "simulator": {
                        "type": "string",
                        "enum": ["gazebo", "isaac_sim"],
                        "description": "Which simulator to launch",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Run without GUI (default: false)",
                        "default": False,
                    },
                },
                "required": ["simulator"],
            },
        ),
        Tool(
            name="get_robot_state",
            description=(
                "Query the current robot state from the running simulation. "
                "Returns joint positions, velocities, body pose, and foot contacts."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "compute_fk":
        return await _compute_fk(arguments)
    elif name == "compute_ik":
        return await _compute_ik(arguments)
    elif name == "launch_sim":
        return await _launch_sim(arguments)
    elif name == "get_robot_state":
        return await _get_robot_state(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _compute_fk(args: dict) -> list[TextContent]:
    try:
        from src.kinematics.leg import LegKinematics
        leg_id = args["leg"]
        angles = args["joint_angles_rad"]
        kin = LegKinematics(leg_id)
        pos = kin.forward(angles)
        result = {"leg": leg_id, "foot_pos_m": pos.tolist()}
    except ImportError:
        result = {"error": "Kinematics module not yet implemented. See src/kinematics/leg.py"}
    except Exception as e:
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _compute_ik(args: dict) -> list[TextContent]:
    try:
        from src.kinematics.leg import LegKinematics
        leg_id = args["leg"]
        foot_pos = args["foot_pos_m"]
        kin = LegKinematics(leg_id)
        angles = kin.inverse(foot_pos)
        result = {"leg": leg_id, "joint_angles_rad": angles.tolist()}
    except ImportError:
        result = {"error": "Kinematics module not yet implemented. See src/kinematics/leg.py"}
    except Exception as e:
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _launch_sim(args: dict) -> list[TextContent]:
    simulator = args.get("simulator", "gazebo")
    headless = args.get("headless", False)
    result = {
        "status": "not_implemented",
        "message": (
            f"Launcher for '{simulator}' not wired up yet. "
            f"See src/simulation/{simulator}/ for integration stubs."
        ),
        "headless": headless,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _get_robot_state(args: dict) -> list[TextContent]:
    result = {
        "status": "not_implemented",
        "message": "Connect to a running ROS2 /joint_states and /odom topic first.",
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
