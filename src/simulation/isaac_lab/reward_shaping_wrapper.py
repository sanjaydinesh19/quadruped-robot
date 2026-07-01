"""
Reproduces legged_gym's `only_positive_rewards` convention, which every proven
reference implementation (ETH legged_gym, Isaac Lab's own ANYmal/A1/Go1
configs) uses and Isaac Lab's manager-based RewardManager has no built-in
equivalent for.

Mechanism: clamp the summed per-step task/shaping reward to >= 0, then add a
large, uncapped penalty on top for a *real* fall (not an episode timeout).
Without the floor, every stability penalty in RewardsCfg (height, orientation,
torques, ...) still applies in full to a clumsy, imperfect walking attempt,
while a robot that simply never moves pays almost none of them — so an early,
uncoordinated step can score worse per-step than standing rigidly still. That
is a documented RL-locomotion failure mode (survival-bonus local minima), and
matches exactly what this project's training runs show: episode length and
reward both jump hard once the policy learns not to fall, then plateau flat
with no further progress toward actual walking. The floor removes the
incentive to avoid movement altogether; the uncapped fall penalty keeps
falling clearly worse than any amount of clumsy stepping.
"""
from __future__ import annotations

import torch

# Starting point, not a literature value — legged_gym's own `termination` reward
# scale is disabled (0.0) in its base config and only enabled at small
# magnitudes (roughly -1 to -5) in downstream configs that use it, scaled to
# that project's per-step reward range. This project's observed per-episode
# reward has been sitting around 1.5-2.5 (see TensorBoard Train/mean_reward),
# so -10 is a clear, unambiguous penalty relative to that scale without being
# so large it dominates the advantage estimate. Re-tune after watching how the
# next run's reward curve responds.
DEFAULT_FALL_PENALTY = -10.0


class RewardShapingWrapper:
    """Wraps an RslRlVecEnvWrapper to apply the only-positive-rewards floor."""

    def __init__(self, env, fall_penalty: float = DEFAULT_FALL_PENALTY) -> None:
        self.env = env
        self.fall_penalty = fall_penalty

    def step(self, actions):
        obs, rew, dones, extras = self.env.step(actions)
        rew = rew.clamp(min=0.0)
        time_outs = extras.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.bool)
        real_falls = dones.bool() & ~time_outs.bool()
        rew = rew + real_falls.to(rew.dtype) * self.fall_penalty
        return obs, rew, dones, extras

    def __getattr__(self, name):
        return getattr(self.env, name)
