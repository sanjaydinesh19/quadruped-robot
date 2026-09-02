"""
Adapter between isaaclab_rl's RSL-RL runner config and the container's
rsl_rl.runners.OnPolicyRunner, which reads a plain dict with "policy" and
"algorithm" keys off `train_cfg`.

Earlier versions of this file (and agents/rsl_rl_ppo_cfg.py) assumed the
pinned container shipped isaaclab_rl 0.5.x+'s split actor/critic API
(RslRlMLPModelCfg) paired with an rsl_rl runner that predated the split, and
manually reconstructed a "policy" dict from separate "actor"/"critic" dicts
to bridge the two. That assumption was never actually checked against the
container and was wrong: the container pins isaaclab_rl==0.4.7, which still
uses the older monolithic RslRlPpoActorCriticCfg (see rsl_rl_ppo_cfg.py) —
found only once `from isaaclab_rl.rsl_rl import RslRlMLPModelCfg` failed at
import time with `ImportError: cannot import name 'RslRlMLPModelCfg'`. With
the monolithic config, `runner_cfg.to_dict()` already produces a correctly
shaped "policy" key natively — no actor/critic merging needed anymore.

What's still real and still handled here: RslRlPpoAlgorithmCfg carries fields
(rnd_cfg, symmetry_cfg, optimizer, share_cnn_encoders, ...) that this
container's exact pinned rsl_rl PPO constructor may not accept as kwargs, so
the algorithm dict is still filtered defensively. Shared by train_rl.py,
play_rl.py, and watch_rl.py so any future API-mismatch fix only needs to
happen in one place.
"""
from __future__ import annotations

import glob
import os
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

    if isinstance(runner_cfg_dict.get("algorithm"), dict):
        runner_cfg_dict["algorithm"] = {
            k: v for k, v in runner_cfg_dict["algorithm"].items()
            if k in _VALID_ALG_KWARGS
        }

    return runner_cfg_dict


def latest_checkpoint(log_dir: str) -> str | None:
    """Path to the most recently *written* checkpoint in log_dir, or None.

    OnPolicyRunner.load() needs a specific file, not a directory — passing
    log_dir straight through raises IsADirectoryError. Picking by mtime
    rather than the highest iteration number in the filename matters too: a
    fresh run restarts its iteration counter from 0 in the same log_dir, so
    an old run's high-numbered checkpoint (e.g. model_2999.pt) would
    otherwise outrank the new run's actual latest by filename alone.
    """
    checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)
