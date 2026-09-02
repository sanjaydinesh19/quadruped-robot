"""
RSL-RL PPO configuration for the quadruped flat-terrain task.

Targets the API actually shipped in the pinned container
(nvcr.io/nvidia/isaac-lab:2.3.2 → isaaclab_rl==0.4.7): a single monolithic
RslRlPpoActorCriticCfg (one policy config carrying both actor_hidden_dims
and critic_hidden_dims), not the later split RslRlMLPModelCfg actor/critic
pair from isaaclab_rl 0.5.x+.

This file previously assumed the split API and failed at import time on the
real container:
    ImportError: cannot import name 'RslRlMLPModelCfg' from 'isaaclab_rl.rsl_rl'
That assumption was never checked against the actual pinned container, only
inferred. Corrected here directly against isaaclab_tasks'
config/go2/agents/rsl_rl_ppo_cfg.py read at the v2.3.2 tag specifically (not
main, which is a different, newer API) — same reference Isaac Lab config this
project's reward/PPO values have been cross-checked against elsewhere.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadrupedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Training runner — controls logging, checkpointing, and iteration count."""

    num_steps_per_env: int = 24
    max_iterations: int = 3000
    save_interval: int = 200
    experiment_name: str = "quadruped_flat"
    empirical_normalization: bool = False

    # Single monolithic actor+critic MLP (0.4.7 API has no separate actor/
    # critic model classes — both hidden-dim lists live on one config).
    # [128,128,128] matches Isaac Lab's own Go2Flat/A1Flat/AnymalCFlat configs
    # (their *rough*-terrain configs use the larger [512,256,128] — we're
    # flat-only, so the smaller/faster-converging network is the right match).
    # Go2's own v2.3.2 reference config doesn't set obs_groups either — the
    # framework fills it in from the env's observation groups when absent.
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
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
