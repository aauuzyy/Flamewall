"""
Simplified RL Training Script for Flamewall Bot
This version works with Python 3.9 and provides a basic training example

NOTE: For full RLGym training, upgrade to Python 3.10+
This script demonstrates the concept without requiring the latest RLGym version.
"""

print("="*60)
print(" Flamewall Simple RL Training")
print("="*60)
print()

# Check Python version
import sys
print(f"Python version: {sys.version}")
print()

if sys.version_info < (3, 10):
    print("⚠️  WARNING: Python 3.10+ recommended for full RLGym support")
    print("   This script provides a simplified training example.")
    print("   See SETUP_NOTE.md for upgrade instructions.")
    print()

# Try to import required packages
missing_packages = []

try:
    import numpy as np
    print("✓ NumPy installed")
except ImportError:
    missing_packages.append("numpy")
    print("✗ NumPy not installed")

try:
    from stable_baselines3 import PPO
    print("✓ Stable-Baselines3 installed")
except ImportError:
    missing_packages.append("stable-baselines3")
    print("✗ Stable-Baselines3 not installed")

try:
    import torch
    print("✓ PyTorch installed")
except ImportError:
    missing_packages.append("torch")
    print("✗ PyTorch not installed")

print()

if missing_packages:
    print(f"❌ Missing packages: {', '.join(missing_packages)}")
    print("\nInstall with: pip install " + " ".join(missing_packages))
    sys.exit(1)

print("="*60)
print(" Creating Simple Training Environment")
print("="*60)
print()

# Create a simple gym environment as an example
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

class SimpleRocketLeagueEnv(gym.Env):
    """
    A simplified Rocket League environment for demonstration
    
    This is a basic example showing the structure.
    For real training, use RLGym with Python 3.10+
    """
    def __init__(self):
        super().__init__()
        
        # Action space: throttle, steer, jump, boost (simplified)
        self.action_space = spaces.Discrete(8)
        
        # Observation space: car pos (3), car vel (3), ball pos (3), ball vel (3) = 12 values
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(12,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        # Random starting positions
        self.car_pos = np.random.uniform(-1, 1, 3).astype(np.float32)
        self.car_vel = np.zeros(3, dtype=np.float32)
        self.ball_pos = np.random.uniform(-0.5, 0.5, 3).astype(np.float32)
        self.ball_vel = np.zeros(3, dtype=np.float32)
        self.steps = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get current observation"""
        return np.concatenate([
            self.car_pos,
            self.car_vel,
            self.ball_pos,
            self.ball_vel
        ]).astype(np.float32)
    
    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1
        
        # Simplified physics (just for demonstration)
        # In real RLGym, this would be actual Rocket League physics
        
        # Update car based on action
        if action == 1:  # Forward
            self.car_vel += np.array([0.1, 0, 0])
        elif action == 2:  # Left
            self.car_vel += np.array([0, 0.1, 0])
        elif action == 3:  # Right
            self.car_vel += np.array([0, -0.1, 0])
        
        # Apply velocity
        self.car_pos += self.car_vel * 0.1
        self.car_pos = np.clip(self.car_pos, -1, 1)
        
        # Apply friction
        self.car_vel *= 0.9
        
        # Ball physics (simple)
        self.ball_vel *= 0.95
        self.ball_pos += self.ball_vel * 0.1
        
        # Calculate reward based on distance to ball
        distance_to_ball = np.linalg.norm(self.car_pos - self.ball_pos)
        reward = -distance_to_ball  # Reward being close to ball
        
        # Check if done
        done = self.steps >= 1000
        
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        pass

print("✓ Simple environment created")
print()
print("="*60)
print(" Training Demo Bot")
print("="*60)
print()
print("This is a DEMONSTRATION showing the training structure.")
print("It uses a simplified physics model, not actual Rocket League.")
print()
print("For real Rocket League training:")
print("  1. Upgrade to Python 3.10+")
print("  2. Run: pip install rlgym-tools")
print("  3. Run: python train_rlgym.py")
print()
print("="*60)
print()

response = input("Continue with demo training? (y/n): ")
if response.lower() != 'y':
    print("Training cancelled.")
    sys.exit(0)

print()
print("Starting demo training...")
print()

# Create directories
models_dir = "models/demo"
logs_dir = "logs/demo"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Create environment
env = SimpleRocketLeagueEnv()

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=logs_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
)

print()
print("Training for 50,000 steps (this is just a demo)...")
print()

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    name_prefix="demo_model",
)

try:
    # Train
    model.learn(
        total_timesteps=50000,
        callback=checkpoint_callback,
        tb_log_name="demo_ppo",
    )
    
    # Save final model
    model.save(f"{models_dir}/demo_final_model")
    
    print()
    print("="*60)
    print(" Demo Training Complete!")
    print("="*60)
    print()
    print(f"✓ Model saved to: {models_dir}/demo_final_model.zip")
    print(f"✓ Logs saved to: {logs_dir}")
    print()
    print("This was a DEMO with simplified physics.")
    print()
    print("Next steps for REAL training:")
    print("  1. Upgrade to Python 3.10 or 3.11")
    print("  2. pip install rlgym-tools")
    print("  3. python train_rlgym.py")
    print()
    print("See SETUP_NOTE.md for details!")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted!")
    model.save(f"{models_dir}/demo_interrupted_model")
    print(f"Model saved to: {models_dir}/demo_interrupted_model.zip")
