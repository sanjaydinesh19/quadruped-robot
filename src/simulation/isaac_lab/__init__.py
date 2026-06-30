"""
Isaac Lab environment for quadruped locomotion.

Importing this module registers the environment with gymnasium so it can be
launched via:
    gym.make("Isaac-Quadruped-Flat-v0")
or via Isaac Lab's training scripts with --task Isaac-Quadruped-Flat-v0.
"""
import gymnasium as gym

from .quadruped_env_cfg import QuadrupedFlatEnvCfg

gym.register(
    id="Isaac-Quadruped-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"cfg": QuadrupedFlatEnvCfg()},
    disable_env_checker=True,
)
