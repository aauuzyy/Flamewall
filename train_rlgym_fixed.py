"""
Working RLGym Training Script for Flamewall Bot
Uses RLGym 2.0 with RocketSim physics simulator
"""

import os
import numpy as np

print("="*70)
print(" Flamewall RLGym Training with Python 3.10")
print("="*70)
print()

# Import required packages
try:
    import rlgym.api
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.action_parsers import LookupTableAction
    from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward, VelocityTowardsBallReward
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    print("✓ RLGym imported successfully")
except ImportError as e:
    print(f"✗ Error importing RLGym: {e}")
    print("\nTrying alternative approach...")
    
    # Fallback to basic gym environment
    import gym
    from stable_baselines3 import PPO
    
    print("\n⚠️  Using basic demonstration environment")
    print("For full RocketLeague training, ensure rlgym-tools is properly installed")
    print()
    
    # Create a simple demo environment
    class SimpleDemoEnv(gym.Env):
        def __init__(self):
            super().__init__()
            from gym import spaces
            self.action_space = spaces.Discrete(8)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
            self.reset()
        
        def reset(self):
            self.state = np.random.uniform(-1, 1, 12).astype(np.float32)
            self.steps = 0
            return self.state
        
        def step(self, action):
            self.steps += 1
            self.state += np.random.uniform(-0.1, 0.1, 12).astype(np.float32)
            self.state = np.clip(self.state, -1, 1)
            reward = np.random.uniform(0, 1)
            done = self.steps >= 1000
            return self.state, reward, done, {}
        
        def render(self, mode='human'):
            pass
    
    print("Creating demo environment...")
    env = SimpleDemoEnv()
    
    models_dir = "models/demo_ppo"
    logs_dir = "logs/demo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"✓ Demo environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print()
    
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)
    print("✓ Model created")
    print()
    
    print("="*70)
    print("Starting DEMO training (10,000 steps)...")
    print("This is a simplified demo. For real Rocket League training,")
    print("see the RLGym documentation at https://rlgym.org")
    print("="*70)
    print()
    
    try:
        model.learn(total_timesteps=10000, tb_log_name="demo_training")
        model.save(f"{models_dir}/demo_model")
        print()
        print("="*70)
        print("✓ Demo training complete!")
        print(f"  Model saved to: {models_dir}/demo_model.zip")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        model.save(f"{models_dir}/demo_model_interrupted")
        print(f"Model saved to: {models_dir}/demo_model_interrupted.zip")
    
    import sys
    sys.exit(0)

# If we get here, RLGym imported successfully
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

print("✓ Stable-Baselines3 imported")
print()

def create_rlgym_env():
    """
    Create RLGym environment with RocketSim
    """
    print("Creating RLGym environment...")
    
    # Create the engine (RocketSim physics)
    engine = RocketSimEngine()
    
    # Create reward function
    reward_fn = CombinedReward(
        (GoalReward(), TouchReward(), VelocityTowardsBallReward()),
        (10.0, 0.5, 0.1)
    )
    
    # Create done conditions
    done = AnyCondition(
        GoalCondition(),
        TimeoutCondition(timeout=300)  # 300 ticks = ~10 seconds
    )
    
    # Create observation builder
    obs_builder = DefaultObs(zero_padding=11)
    
    # Create action parser
    action_parser = LookupTableAction()
    
    # Create state mutator
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    
    # Create the environment
    env = rlgym.api.RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=done,
        truncation_cond=done,
        transition_engine=engine,
        renderer=None  # No rendering for faster training
    )
    
    print(f"✓ RLGym environment created")
    print(f"  Team size: 1v1")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    return env

def train():
    """
    Main training function
    """
    models_dir = "models/rlgym_ppo"
    logs_dir = "logs/rlgym"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print()
    print("="*70)
    print(" Starting RLGym Training")
    print("="*70)
    print()
    
    # Create environment
    env = create_rlgym_env()
    
    # Wrap with monitor
    env = VecMonitor(env)
    
    print()
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
    )
    print("✓ PPO model created")
    print()
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="flamewall_rlgym",
    )
    
    print("="*70)
    print(" Training Configuration")
    print("="*70)
    print(f"  Total timesteps: 1,000,000")
    print(f"  Checkpoint every: 50,000 steps")
    print(f"  Learning rate: 5e-5")
    print(f"  Batch size: 256")
    print()
    print("Monitor training with: tensorboard --logdir=logs/rlgym")
    print("Press Ctrl+C to stop training early")
    print("="*70)
    print()
    
    try:
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            log_interval=10,
            tb_log_name="flamewall_rlgym_ppo",
        )
        
        model.save(f"{models_dir}/flamewall_rlgym_final")
        print()
        print("="*70)
        print("✓ Training complete!")
        print(f"  Final model saved to: {models_dir}/flamewall_rlgym_final.zip")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        model.save(f"{models_dir}/flamewall_rlgym_interrupted")
        print(f"Model saved to: {models_dir}/flamewall_rlgym_interrupted.zip")

if __name__ == "__main__":
    train()
