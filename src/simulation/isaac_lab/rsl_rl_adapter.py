"""
Adapter between isaaclab_rl's actor/critic RSL-RL config and the container's
rsl_rl runner, which expects a single "policy" key (ActorCritic) and a
narrower set of algorithm kwargs.

Needed because isaaclab_rl 0.5.x split RslRlPpoActorCriticCfg into separate
actor/critic RslRlMLPModelCfg objects, but the pinned container's rsl_rl
(OnPolicyRunner) predates that split. Shared by train_rl.py, play_rl.py, and
watch_rl.py so the API-mismatch fixes only need to happen in one place.
"""
from __future__ import annotations

from typing import Any

_VALID_ALG_KWARGS = {
    "class_name", "num_learning_epochs", "num_mini_batches", "learning_rate",
    "schedule", "gamma", "lam", "entropy_coef", "desired_kl", "max_grad_norm",
    "value_loss_coef", "use_clipped_value_loss", "clip_param",
    "normalize_advantage_per_mini_batch",
}


def build_runner_cfg_dict(runner_cfg: Any) -> dict:
    """Convert a QuadrupedPPORunnerCfg into the dict shape OnPolicyRunner expects."""
    runner_cfg_dict = runner_cfg.to_dict()

    actor = runner_cfg_dict.get("actor") or {}
    critic = runner_cfg_dict.get("critic") or {}
    dist = actor.get("distribution_cfg") or {}
    runner_cfg_dict["policy"] = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": actor.get("hidden_dims", [512, 256, 128]),
        "critic_hidden_dims": critic.get("hidden_dims", [512, 256, 128]),
        "activation": actor.get("activation", "elu"),
        "init_noise_std": (dist.get("init_std") if isinstance(dist, dict) else 1.0) or 1.0,
    }

    if isinstance(runner_cfg_dict.get("algorithm"), dict):
        runner_cfg_dict["algorithm"] = {
            k: v for k, v in runner_cfg_dict["algorithm"].items()
            if k in _VALID_ALG_KWARGS
        }

    return runner_cfg_dict
