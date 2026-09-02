"""
Velocity-command curriculum.

isaaclab main (post-2.3.2) ships a generic `modify_env_param`/`modify_term_cfg`
curriculum helper for exactly this kind of scheduled range change, but its
availability in the pinned `nvcr.io/nvidia/isaac-lab:2.3.2` container is
unverified — this project has already been bitten twice by API drift between
what's importable in a given isaaclab/isaaclab_rl release and what the pinned
container actually ships (see rsl_rl_adapter.py, mdp/rewards.py). Implemented
here instead using only `ManagerBase.get_term()` and `env.common_step_counter`,
both long-stable core APIs already relied on elsewhere in this file/package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_ranges_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    initial_ranges: dict[str, tuple[float, float]],
    final_ranges: dict[str, tuple[float, float]],
    warmup_steps: int,
    ramp_steps: int,
) -> torch.Tensor:
    """Linearly widen a UniformVelocityCommand's sampling ranges over training.

    Motivated by unitreerobotics/unitree_rl_lab's `lin_vel_cmd_levels` curriculum
    for Go2 (narrow commands — reported ±0.1 m/s — ramped toward a full-range
    limit as training progresses), which is a more principled fix for the exact
    V3 creep failure than V4's static range widening:

      - Too narrow from iteration 0 (V3's ±0.5 m/s): a planted-feet creep can
        track it almost perfectly, so there is never a tracking-reward gradient
        toward stepping (see README's V3 root-cause analysis, point 2).
      - Too wide from iteration 0: a random-init policy cannot track *any*
        velocity, so early PPO gets a noisy, low-magnitude tracking signal on
        top of an already-hard exploration problem, before it has learned even
        basic balance.

    Starting narrow (inside the range creeping can still track) gives early
    training a learnable signal — walk-or-stand-still are both fine early on —
    then ramping to the full ±1.0 m/s / ±1.0 rad/s range (which creeping
    provably cannot track, per the same root-cause analysis) only once the
    policy already has some locomotion competence, so the "must step to keep
    tracking" pressure arrives after exploration has found stepping is
    possible, not before.

    `warmup_steps`/`ramp_steps` are in units of `env.common_step_counter`
    (policy steps, incremented once per `env.step()` call regardless of
    `num_envs` — NOT PPO iterations). With `num_steps_per_env=24`
    (rsl_rl_ppo_cfg.py), 1 PPO iteration ≈ 24 steps.

    Applied globally (keyed on `env.common_step_counter`, ignoring `env_ids`)
    rather than per-env-performance-gated like unitree_rl_lab's version — same
    ramp for every env, avoids needing an extra per-env tracking-error buffer.
    Simpler to verify correct; a per-env-performance gate is a reasonable
    follow-up once V5 data exists to show whether the fixed ramp under- or
    over-shoots.
    """
    command_term = env.command_manager.get_term(command_name)
    progress = min(max((env.common_step_counter - warmup_steps) / max(ramp_steps, 1), 0.0), 1.0)

    for axis, (lo1, hi1) in final_ranges.items():
        lo0, hi0 = initial_ranges[axis]
        new_range = (lo0 + (lo1 - lo0) * progress, hi0 + (hi1 - hi0) * progress)
        setattr(command_term.cfg.ranges, axis, new_range)

    return torch.tensor(progress)
