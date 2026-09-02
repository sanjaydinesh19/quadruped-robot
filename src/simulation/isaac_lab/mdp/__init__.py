"""Reward terms not available in isaaclab.envs.mdp under isaaclab 2.3.x.

Both are verbatim vendored ports from isaaclab_tasks' locomotion velocity
task. V6 removed this package's custom additions (feet_air_time_variance,
stand_still_joint_deviation_l1, and the command_ranges_curriculum module) —
see quadruped_env_cfg.py's module docstring for why.
"""
from .rewards import feet_air_time, feet_slide

__all__ = ["feet_air_time", "feet_slide"]
