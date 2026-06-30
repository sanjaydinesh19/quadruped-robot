"""
RSL-RL PPO configuration for the quadruped flat-terrain task.

RSL-RL is the on-policy RL library that ships with Isaac Lab (originally from
ETH Zurich's Robotic Systems Lab — hence the name). It implements PPO with
a few locomotion-specific tweaks.

Tuned for:
  - Local RTX 3050 4GB:  num_envs=32,  batch≈768  steps
  - Cloud A100 80GB:     num_envs=4096, batch≈98 304 steps  (change in train.py)
"""
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class QuadrupedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training runner — controls logging, checkpointing, and iteration count."""

    num_steps_per_env = 24      # rollout length per environment per update
    max_iterations    = 3000    # total gradient steps (≈ 20–30 min on A100)
    save_interval     = 200     # save checkpoint every N iterations
    experiment_name   = "quadruped_flat"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Three hidden layers; the 512→256→128 funnel is standard for locomotion.
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipping=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,     # mini-batch size = (num_envs × num_steps) / 4
        learning_rate=1.0e-3,
        schedule="adaptive",    # adjusts LR to keep KL divergence near desired_kl
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
