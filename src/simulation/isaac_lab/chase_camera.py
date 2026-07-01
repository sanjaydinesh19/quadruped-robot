"""
Chase-camera helper for headless MJPEG/video capture scripts.

watch_rl.py and play_rl.py's render camera otherwise sits wherever
QuadrupedFlatEnvCfg.viewer.eye/lookat puts it — fixed, robot-position-
independent. That's fine for a robot that stays near the origin, but a
policy that's actually tracking velocity commands can walk far enough
during a 20s+ episode to leave that fixed frame entirely (this is what
happened watching an iteration-278 checkpoint: the robot hadn't crashed,
it had just walked out of view). Recomputing eye/target from the robot's
current position every frame keeps it in view regardless of how far it
walks, using the same relative viewing angle/distance the fixed camera
used to have.
"""
from __future__ import annotations

_EYE_OFFSET = (3.5, 3.5, 2.0)
_TARGET_Z_OFFSET = 0.15


def update_chase_camera(env, env_index: int = 0) -> None:
    """Recenter the render camera on env_index's robot."""
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[env_index]
    eye = (
        float(pos[0]) + _EYE_OFFSET[0],
        float(pos[1]) + _EYE_OFFSET[1],
        float(pos[2]) + _EYE_OFFSET[2],
    )
    target = (float(pos[0]), float(pos[1]), float(pos[2]) + _TARGET_Z_OFFSET)
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)
