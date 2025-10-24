"""
Real RLGym Training with RocketSim Physics
Trains a Rocket League bot using reinforcement learning
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from gymnasium import spaces

print("="*70)
print(" Flamewall - Real RLGym Training")
print("="*70)
print()

# Import RLGym components
print("Loading RLGym components...")
import rlgym.api
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator

print("✓ All imports successful")
print()

class GymnasiumWrapper(gym.Env):
    """Wraps RLGym environment as Gymnasium environment for SB3"""
    def __init__(self, rlgym_env):
        super().__init__()
        self.env = rlgym_env
        # Get action and observation spaces for agent 0
        act_space = self.env.action_space(agent=0)
        obs_space = self.env.observation_space(agent=0)
        
        # Convert from RLGym format to Gymnasium spaces
        # Action space: ('discrete', n) -> Discrete(n)
        if act_space[0] == 'discrete':
            self.action_space = spaces.Discrete(act_space[1])
        else:
            raise ValueError(f"Unsupported action space type: {act_space[0]}")
        
        # Observation space: ('real', n) -> Box
        if obs_space[0] == 'real':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_space[1],), dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported observation space type: {obs_space[0]}")
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs_dict = self.env.reset()
        # RLGym returns a dict with agent IDs as keys
        # Get the first agent's observation
        if isinstance(obs_dict, dict):
            obs = list(obs_dict.values())[0]  # Get first agent's obs
        else:
            obs = obs_dict
        return obs, {}
    
    def step(self, action):
        # Step the environment with a dict of actions
        # RLGym expects dict with agent IDs, action must be np array with shape (1,)
        action = np.array([action])  # Convert to expected shape
        action_dict = {agent_id: action for agent_id in self.env.agents}
        step_result = self.env.step(action_dict)
        
        # Extract results for first agent
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = step_result
            obs = list(obs_dict.values())[0]
            reward = list(rew_dict.values())[0]
            terminated = list(term_dict.values())[0]
            truncated = list(trunc_dict.values())[0]
            info = list(info_dict.values())[0] if info_dict else {}
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            # Old format: returns dicts but only 4 elements (obs, reward, done, info)
            obs_dict, rew_dict, done_dict, info_dict = step_result
            obs = list(obs_dict.values())[0]
            reward = float(list(rew_dict.values())[0])
            terminated = bool(list(done_dict.values())[0])
            truncated = False
            info = list(info_dict.values())[0] if info_dict else {}
            # Ensure info is a dict
            if not isinstance(info, dict):
                info = {}
        else:
            raise ValueError(f"Unexpected step result: {len(step_result)} elements")
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

def create_rlgym_env():
    """
    Create RLGym environment with RocketSim physics
    """
    print("Creating RLGym environment with RocketSim...")
    
    # Physics engine
    engine = RocketSimEngine()
    
    # Reward: Encourage goals and ball touches
    reward_fn = CombinedReward(
        (GoalReward(), 10.0),   # Reward for scoring (weight=10.0)
        (TouchReward(), 0.5),   # Reward for touching ball (weight=0.5)
    )
    
    # End episode on goal or timeout
    done_condition = AnyCondition(
        GoalCondition(),
        TimeoutCondition(timeout_seconds=10.0)  # 10 seconds max per episode
    )
    
    # What the agent observes
    obs_builder = DefaultObs(zero_padding=11)
    
    # How actions are interpreted
    action_parser = LookupTableAction()
    
    # How to reset/initialize episodes
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),  # 1v1
        KickoffMutator()  # Start from kickoff
    )
    
    # Create the environment
    env = rlgym.api.RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=done_condition,
        truncation_cond=done_condition,
        transition_engine=engine,
        renderer=None  # No rendering = faster training
    )
    
    print(f"✓ Environment created successfully!")
    print(f"  Team size: 1v1")
    print(f"  Physics: RocketSim (real Rocket League physics)")
    
    # Wrap for Gymnasium compatibility
    env = GymnasiumWrapper(env)
    
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    return env

def train():
    """
    Main training function
    """
    # Setup directories
    models_dir = "models/rlgym_real"
    logs_dir = "logs/rlgym_real"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print()
    print("="*70)
    print(" Training Configuration")
    print("="*70)
    print()
    
    # Create environment
    env = create_rlgym_env()
    
    # No need to wrap - RLGym works directly with SB3
    print()
    print("Creating PPO agent...")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        device="cpu",  # Use CPU for better stability with RL
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration
    )
    
    print("✓ PPO agent created")
    print()
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="flamewall",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    print("="*70)
    print(" Starting Training")
    print("="*70)
    print(f"  Total timesteps: 1,000,000")
    print(f"  Checkpoint every: 50,000 steps")
    print(f"  Device: CPU")
    print()
    print("Monitor progress:")
    print(f"  tensorboard --logdir={logs_dir}")
    print()
    print("Press Ctrl+C anytime to stop and save progress")
    print("="*70)
    print()
    
    try:
        # Train!
        model.learn(
            total_timesteps=1_000_000,
            callback=checkpoint_callback,
            log_interval=10,
            tb_log_name="flamewall_ppo",
            progress_bar=True,
        )
        
        # Save final model
        final_path = f"{models_dir}/flamewall_final"
        model.save(final_path)
        
        print()
        print("="*70)
        print(" Training Complete! 🎉")
        print("="*70)
        print(f"  Final model: {final_path}.zip")
        print(f"  Checkpoints: {models_dir}/")
        print(f"  Logs: {logs_dir}/")
        print()
        print("Next steps:")
        print("  1. Test your model in RLBot")
        print("  2. Continue training from checkpoint")
        print("  3. Adjust rewards for better behavior")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n")
        print("="*70)
        print(" Training Interrupted")
        print("="*70)
        interrupted_path = f"{models_dir}/flamewall_interrupted"
        model.save(interrupted_path)
        print(f"  Model saved: {interrupted_path}.zip")
        print(f"  You can resume training from this checkpoint")
        print("="*70)
    
    finally:
        env.close()

if __name__ == "__main__":
    train()
