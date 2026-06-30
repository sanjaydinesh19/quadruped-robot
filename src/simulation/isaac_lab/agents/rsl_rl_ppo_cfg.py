"""
RSL-RL PPO configuration for the quadruped flat-terrain task (rsl_rl >= 4.0.0 API).

isaaclab_rl 0.5.x / rsl_rl 4.x replaced the monolithic RslRlPpoActorCriticCfg
with separate RslRlMLPModelCfg for actor and critic, and removed use_clipping /
clip_param from the algorithm config in favour of use_clipped_value_loss.
"""
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class QuadrupedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training runner — controls logging, checkpointing, and iteration count."""

    num_steps_per_env: int = 24
    max_iterations: int = 3000
    save_interval: int = 200
    experiment_name: str = "quadruped_flat"
    empirical_normalization: bool = False

    # Map env observation groups → algorithm observation sets.
    # Our env exposes a single "policy" group; both actor and critic see it.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # Actor: stochastic MLP with Gaussian output (mean + fixed std).
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    # Critic: deterministic value MLP (no distribution needed).
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
