"""
🔥 Flamewall Simple Local Training Script 🔥
No distributed training, no Redis - just pure PPO training!
"""

import torch
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from rlgym.utils.reward_functions import CombinedReward

from training.reward import FlamewallRewardFunction

def create_flamewall_env():
    """Create RLGym environment with hivemind rewards"""
    
    return Match(
        team_size=3,  # 3v3 for hivemind training
        tick_skip=8,
        reward_function=FlamewallRewardFunction(),
        terminal_conditions=[TimeoutCondition(300), GoalScoredCondition()],
        obs_builder=AdvancedObs(),
        action_parser=DiscreteAction(),
        state_setter=DefaultState(),
    )

def train_flamewall(
    total_timesteps=1_000_000,
    save_freq=50_000,
    model_path="src_flamewall/flamewall-model.pt",
    checkpoint_dir="./checkpoints/"
):
    """
    Train Flamewall with hivemind coordination rewards
    
    Args:
        total_timesteps: How many steps to train (default 1M)
        save_freq: Save checkpoint every N steps
        model_path: Path to existing model to continue training
        checkpoint_dir: Where to save checkpoints
    """
    
    print("🔥 FLAMEWALL HIVEMIND TRAINING 🔥")
    print(f"Training for {total_timesteps:,} timesteps")
    print(f"Checkpoints every {save_freq:,} steps → {checkpoint_dir}")
    print()
    
    # Create environment
    print("Creating RLGym environment with hivemind rewards...")
    env = create_flamewall_env()
    
    # Wrap for monitoring
    env = VecMonitor(env)
    env = VecCheckNan(env)
    
    # Create or load model
    try:
        print(f"Loading existing model from {model_path}...")
        # Note: For PyTorch JIT models, you'll need to convert first
        # For now, start fresh with Stable-Baselines3
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-5,
            n_steps=4096,
            batch_size=256,
            n_epochs=30,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log="./tb_logs/flamewall/"
        )
        print("Created new PPO model (SB3 format)")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-5,
            n_steps=4096,
            batch_size=256,
            n_epochs=30,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log="./tb_logs/flamewall/"
        )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="flamewall",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Train!
    print()
    print("🔥 Starting training... 🔥")
    print("Watch for spacing rewards increasing and stack penalties decreasing!")
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        tb_log_name="flamewall_hivemind"
    )
    
    # Save final model
    final_path = f"{checkpoint_dir}/flamewall_final.zip"
    model.save(final_path)
    print()
    print(f"🔥 Training complete! Final model saved to {final_path}")
    print()
    print("To use this model:")
    print(f"1. Copy {final_path} to models/")
    print("2. Update src_flamewall/agent.py to load SB3 models")
    print("3. Test in RLBot GUI!")
    
    return model

if __name__ == "__main__":
    import os
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Start training!
    train_flamewall(
        total_timesteps=500_000,  # Start with 500K steps (~few hours)
        save_freq=25_000,         # Save every 25K steps
        checkpoint_dir="./checkpoints/hivemind/"
    )
