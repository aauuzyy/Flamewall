"""
Flamewall 3v3 Training Script
Just run: py -3.10 train.py

Super simple - no complicated config!
"""
import os
import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs  
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence
from rlgym.api import RLGymAPI
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

print("🚀 Creating simple 3v3 environment...")

# Create a simple environment
def make_env():
    return RLGym(
        state_mutator=KickoffMutator(),
        obs_builder=DefaultObs(zero_padding=11),
        action_parser=LookupTableAction(),
        reward_fn=GoalReward(),  # Simple: reward for goals
        transition_engine=RocketSimEngine(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(timeout_seconds=300),
    )

env = make_env()
sb3_env = RLGymAPI(env)
sb3_env = VecMonitor(sb3_env)
sb3_env = VecNormalize(sb3_env, norm_obs=False, norm_reward=True, gamma=0.99)

# Create models directory
os.makedirs("models", exist_ok=True)

# Save every 100K steps
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./models/",
    name_prefix="flamewall"
)

print("🧠 Creating PPO agent...")
model = PPO(
    "MlpPolicy",
    sb3_env,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda",
    learning_rate=5e-5,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
)

print("\n🎮 Starting training!")
print("📊 Checkpoints every 100K steps → ./models/")
print("📈 Tensorboard: tensorboard --logdir=./logs/")
print("⏹️  Press Ctrl+C to stop\n")

try:
    model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    print("\n✅ Training complete!")
    model.save("models/flamewall_final")
except KeyboardInterrupt:
    print("\n⏹️  Stopped by user")
    model.save("models/flamewall_interrupted")
    print("💾 Model saved!")
