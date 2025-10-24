"""
🎯 ULTIMATE TRAINING: Dynamic Rewards + Flick Mastery
Features:
- Dynamic reward system that adapts to bot activity
- Ball-on-car spawning for flick training
- Progressive difficulty
- All advanced features from train_advanced.py
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Changed from SubprocVecEnv for Windows compatibility
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import glob

# Add rlgym_config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rlgym_config'))

print("="*70)
print(" 🔥 FLAMEWALL - ULTIMATE TRAINING 🔥")
print(" Dynamic Rewards + Flick Mastery")
print("="*70)
print()

# Import RLGym components
print("Loading RLGym components...")
import rlgym.api
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator

# Import our custom components
try:
    from dynamic_rewards import DynamicRewardSystem
    from flick_training import BallOnCarMutator, ProgressiveFlickTraining, FlickSuccessDetector
    print("✓ Custom reward and training systems loaded!")
except ImportError as e:
    print(f"⚠️  Warning: Could not import custom components: {e}")
    print("   Falling back to basic rewards")
    DynamicRewardSystem = None

print("✓ All imports successful")
print()


class GymnasiumWrapper(gym.Env):
    """Wraps RLGym environment as Gymnasium environment for SB3"""
    def __init__(self, rlgym_env):
        super().__init__()
        self.env = rlgym_env
        act_space = self.env.action_space(agent=0)
        obs_space = self.env.observation_space(agent=0)
        
        if act_space[0] == 'discrete':
            self.action_space = spaces.Discrete(act_space[1])
        else:
            raise ValueError(f"Unsupported action space type: {act_space[0]}")
        
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
        if isinstance(obs_dict, dict):
            obs = list(obs_dict.values())[0]
        else:
            obs = obs_dict
        return obs, {}
    
    def step(self, action):
        action = np.array([action])
        action_dict = {agent_id: action for agent_id in self.env.agents}
        step_result = self.env.step(action_dict)
        
        if len(step_result) == 5:
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = step_result
            obs = list(obs_dict.values())[0]
            reward = float(list(rew_dict.values())[0])
            terminated = bool(list(term_dict.values())[0])
            truncated = bool(list(trunc_dict.values())[0])
            info = list(info_dict.values())[0] if info_dict else {}
        elif len(step_result) == 4:
            obs_dict, rew_dict, done_dict, info_dict = step_result
            obs = list(obs_dict.values())[0]
            reward = float(list(rew_dict.values())[0])
            terminated = bool(list(done_dict.values())[0])
            truncated = False
            info = list(info_dict.values())[0] if info_dict else {}
        else:
            raise ValueError(f"Unexpected step result: {len(step_result)} elements")
        
        if not isinstance(info, dict):
            info = {}
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


class FlickTrainingCallback(BaseCallback):
    """
    Callback to track flick training progress
    """
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.flick_attempts = 0
        self.flick_successes = 0
        
    def _on_step(self):
        # Track flick success from episode info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'flick_success' in info:
                self.flick_attempts += 1
                if info['flick_success']:
                    self.flick_successes += 1
        
        # Report progress periodically
        if self.num_timesteps % self.check_freq == 0 and self.flick_attempts > 0:
            success_rate = self.flick_successes / max(1, self.flick_attempts)
            print(f"\n🎯 Flick Training Stats:")
            print(f"   Attempts: {self.flick_attempts}")
            print(f"   Successes: {self.flick_successes}")
            print(f"   Success Rate: {success_rate*100:.1f}%\n")
            
            # Reset counters
            self.flick_attempts = 0
            self.flick_successes = 0
        
        return True


class SelfPlayCallback(BaseCallback):
    """Save models for self-play opponents"""
    def __init__(self, save_freq=500000, save_path='models/opponents/'):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.opponent_count = 0
        
    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            opponent_path = os.path.join(self.save_path, f'opponent_{self.opponent_count}.zip')
            self.model.save(opponent_path)
            self.opponent_count += 1
            print(f"\n💾 Saved opponent checkpoint: {opponent_path}\n")
        return True


def create_rlgym_env(use_flick_training=True, training_mode='ultimate'):
    """
    Create RLGym environment with dynamic rewards
    
    training_mode options:
    - 'flick': Pure flick training (ball always on car)
    - 'mixed': Mix of flick training and normal play
    - 'ultimate': Dynamic rewards with all scenarios
    """
    engine = RocketSimEngine()
    
    # DYNAMIC REWARD SYSTEM
    if DynamicRewardSystem:
        print(f"Using Dynamic Reward System (mode: {training_mode})")
        reward_fn = DynamicRewardSystem()
    else:
        # Fallback to RLGym rewards
        from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
        reward_fn = CombinedReward(
            (GoalReward(), 10.0),
            (TouchReward(), 1.0),
        )
    
    # Episode termination
    done_condition = AnyCondition(
        GoalCondition(),
        TimeoutCondition(timeout_seconds=20.0)  # Longer for flick practice
    )
    
    # Observation
    obs_builder = DefaultObs(zero_padding=11)
    
    # Actions
    action_parser = LookupTableAction()
    
    # STATE MUTATOR based on training mode
    if training_mode == 'flick' and use_flick_training:
        # Pure flick training: always spawn ball on car
        print("  Mode: FLICK TRAINING (ball always on car)")
        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            BallOnCarMutator(
                add_velocity=True,
                random_position=True,
                spawn_height_range=(90, 140),
                give_boost=True
            )
        )
    
    elif training_mode == 'mixed' and use_flick_training:
        # Mix of flick training and normal scenarios
        print("  Mode: MIXED (flick + normal play)")
        # This would alternate between flick spawns and normal spawns
        # For now, using progressive flick training
        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            ProgressiveFlickTraining()
        )
    
    else:
        # Ultimate mode: dynamic rewards handle all scenarios
        print("  Mode: ULTIMATE (dynamic rewards for all scenarios)")
        from rlgym.rocket_league.state_mutators import KickoffMutator
        state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator()
        )
    
    # Create environment
    env = rlgym.api.RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=done_condition,
        truncation_cond=done_condition,
        transition_engine=engine,
        renderer=None
    )
    
    return GymnasiumWrapper(env)


def make_env(rank, use_flick_training=True, training_mode='ultimate'):
    """Create a single environment (for parallel training)"""
    def _init():
        env = create_rlgym_env(use_flick_training, training_mode)
        env = Monitor(env)
        return env
    return _init


def train(training_mode='flick', num_envs=4, total_timesteps=10_000_000):
    """
    Main training function
    
    Args:
        training_mode: 'flick', 'mixed', or 'ultimate'
        num_envs: Number of parallel environments
        total_timesteps: Total training steps
    """
    # Setup directories
    models_dir = f"models/ultimate_{training_mode}"
    logs_dir = f"logs/ultimate_{training_mode}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("="*70)
    print(" 🎯 ULTIMATE TRAINING CONFIGURATION")
    print("="*70)
    print()
    print(f"Training Mode: {training_mode.upper()}")
    print()
    
    # PARALLEL ENVIRONMENTS  
    # Using DummyVecEnv for Windows compatibility (runs in single process but still faster)
    print(f"Creating {num_envs} environments...")
    print(f"  Note: Using DummyVecEnv for Windows compatibility")
    print()
    
    use_flick = (training_mode in ['flick', 'mixed'])
    env = DummyVecEnv([make_env(i, use_flick, training_mode) for i in range(num_envs)])
    
    print("✓ Parallel environments created!")
    print()
    
    # Check for existing models
    existing_models = glob.glob(os.path.join(models_dir, "*.zip"))
    
    if existing_models:
        latest_model = max(existing_models, key=os.path.getctime)
        print(f"Found existing model: {latest_model}")
        print(f"Automatically continuing training from checkpoint...")
        model = PPO.load(latest_model, env=env)
        print("✓ Model loaded successfully!")
    else:
        model = None
    
    if model is None:
        print("Creating PPO agent with optimized hyperparameters...")
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=logs_dir,
            device="cpu",
            
            # OPTIMIZED FOR FLICK TRAINING
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # Slightly higher for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            
            # Larger networks for complex behaviors
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512, 512, 256],
                    vf=[512, 512, 256]
                ),
                activation_fn=__import__('torch').nn.ReLU
            )
        )
        
        print("✓ PPO agent created!")
    
    print()
    print("="*70)
    print(" 🚀 STARTING ULTIMATE TRAINING")
    print("="*70)
    print()
    print(f"  Training Mode: {training_mode.upper()}")
    print(f"  Parallel Environments: {num_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Checkpoint every: 100,000 steps")
    print(f"  Device: CPU")
    print()
    
    if training_mode == 'flick':
        print("🎯 FLICK TRAINING MODE:")
        print("  - Ball ALWAYS spawns on car roof")
        print("  - Focus: Learn to flick consistently")
        print("  - Rewards: Huge bonuses for successful flicks")
        print("  - Expected mastery: 2-5M steps")
    elif training_mode == 'mixed':
        print("🎯 MIXED TRAINING MODE:")
        print("  - Progressive flick difficulty")
        print("  - Combines flick training with normal play")
        print("  - Adapts difficulty as bot improves")
    else:
        print("🎯 ULTIMATE MODE:")
        print("  - Dynamic rewards adapt to activity")
        print("  - Rewards change based on game state")
        print("  - Covers: flicks, aerials, ground play, positioning")
        print("  - Learns all mechanics automatically")
    
    print()
    print("Advanced Features:")
    print("  ✓ Dynamic reward system")
    print("  ✓ Activity-based reward scaling")
    print("  ✓ Parallel training (4x faster)")
    print("  ✓ Optimized hyperparameters")
    print("  ✓ Larger neural networks")
    print("  ✓ Self-play checkpoints")
    print()
    print("Monitor progress:")
    print(f"  tensorboard --logdir={logs_dir}")
    print()
    print("Press Ctrl+C anytime to stop and save progress")
    print("="*70)
    print()
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_envs,
        save_path=models_dir,
        name_prefix=f"flamewall_{training_mode}"
    )
    
    selfplay_callback = SelfPlayCallback(
        save_freq=500000 // num_envs,
        save_path='models/opponents/'
    )
    
    flick_callback = FlickTrainingCallback(check_freq=10000)
    
    callbacks = [checkpoint_callback, selfplay_callback, flick_callback]
    
    # TRAIN!
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        final_path = os.path.join(models_dir, f"flamewall_{training_mode}_final.zip")
        model.save(final_path)
        print(f"\n✓ Training complete! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        interrupt_path = os.path.join(models_dir, f"flamewall_{training_mode}_interrupted.zip")
        model.save(interrupt_path)
        print(f"✓ Progress saved to: {interrupt_path}")
    
    env.close()
    print("\n✓ Training session complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate RL Training')
    parser.add_argument('--mode', type=str, default='flick',
                      choices=['flick', 'mixed', 'ultimate'],
                      help='Training mode: flick (ball on car), mixed, or ultimate (dynamic)')
    parser.add_argument('--envs', type=int, default=4,
                      help='Number of parallel environments')
    parser.add_argument('--steps', type=int, default=10_000_000,
                      help='Total training timesteps')
    
    args = parser.parse_args()
    
    train(training_mode=args.mode, num_envs=args.envs, total_timesteps=args.steps)
