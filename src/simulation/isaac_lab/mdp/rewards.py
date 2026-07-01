"""
Gait-quality reward terms.

isaaclab 2.3.x's core `isaaclab.envs.mdp` has no feet_air_time / feet_slide —
those live in `isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards`,
a task-specific extension module, not a pip-installable core dependency we can
rely on being importable. Vendored here (unmodified logic) from
isaac-sim/IsaacLab v2.3.2 so the reward terms don't depend on isaaclab_tasks
being on sys.path.

Without feet_air_time / feet_slide, nothing in the reward function distinguishes
an actual walking gait from sliding the body forward while a foot stays planted
("gliding") — velocity tracking alone rewards both equally.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    Rewards the agent for taking steps longer than `threshold`, so it lifts its
    feet off the ground instead of sliding them. Zero reward for near-zero
    velocity commands (standing still shouldn't be penalised for short air time).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding on the ground.

    Norm of foot linear velocity, gated by a binary contact flag — only
    penalises velocity while the foot is actually touching the ground. This is
    the term that directly punishes "gliding" (planted foot, body sliding).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from the default pose only when barely commanded to move.

    Applying joint_deviation_l1 unconditionally (the previous config) fights the
    leg swing a real gait needs, biasing the policy toward minimal joint motion —
    i.e. toward gliding instead of stepping. Gating it on the command magnitude
    keeps it as a "don't fidget while standing" term instead.
    """
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
