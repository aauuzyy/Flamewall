"""
RLGym Training Script for Flamewall Bot  
This script sets up and trains a reinforcement learning agent using RLGym-Tools and Stable-Baselines3

Note: This uses RocketSim (included in rlgym-tools) which simulates Rocket League physics.
No need to have Rocket League running!
"""

try:
    from rlgym_tools import rocket_league
    HAVE_RLGYM = True
    print("✓ RLGym-tools imported successfully")
except ImportError:
    print("ERROR: rlgym-tools not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rlgym-tools"])
    print("Please run the script again after installation completes.")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
import numpy as np
import os

def create_rlgym_env():
    """
    Create and configure the RLGym environment using RocketSim
    """
    print("Creating RLGym environment with RocketSim...")
    
    # Create environment using rlgym_tools
    env = rocket_league.make(
        team_size=1,  # 1v1 for simpler learning
        tick_skip=8,  # Speed up simulation
        spawn_opponents=True,
    )
    
    print(f"✓ Environment created")
    print(f"  - Team size: 1v1")
    print(f"  - Tick skip: 8")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    return env

def train():
    """
    Main training function
    """
    # Create directories for saving models and logs
    models_dir = "models/PPO"
    logs_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create the environment
    print("Creating RLGym environment...")
    env = create_rlgym_env()
    
    # Wrap environment with monitoring and normalization
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecCheckNan(env)  # Check for NaN values
    
    # Create the PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=5e-5,
        n_steps=4096,  # Steps per update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient for exploration
    )
    
    # Create checkpoint callback to save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Save every 100k steps
        save_path=models_dir,
        name_prefix="flamewall_rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Train the model
    print("Starting training...")
    print("You can monitor training with: tensorboard --logdir=logs")
    
    try:
        model.learn(
            total_timesteps=10_000_000,  # 10 million steps
            callback=checkpoint_callback,
            log_interval=10,
            tb_log_name="flamewall_ppo",
            reset_num_timesteps=False,
        )
        
        # Save final model
        model.save(f"{models_dir}/flamewall_final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        print(f"Training complete! Model saved to {models_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(f"{models_dir}/flamewall_interrupted_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        print(f"Model saved to {models_dir}")
    
    finally:
        env.close()

if __name__ == "__main__":
    train()
