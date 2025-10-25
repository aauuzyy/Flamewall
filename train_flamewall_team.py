"""
🔥🔥🔥 FLAMEWALL TEAM TRAINING 🔥🔥🔥
Train 3 Flamewalls to become an UNSTOPPABLE WALL!

Focus:
- 3v3 team play with 3 Flamewalls vs 3 opponents
- Passing and rotation
- Aerial goals
- Boost management and FEATHERING
- Team positioning (THE WALL!)
"""

import os
import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
import rlgym.rocket_league.math as rlm
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, BALL_RADIUS, BALL_MAX_SPEED

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


# ========================================
# 🎯 FLAMEWALL TEAM REWARD SYSTEM
# ========================================

class FlamewallTeamReward:
    """
    Reward system designed for 3v3 team play
    Focuses on: goals, passing, aerials, boost management, positioning
    """
    
    def __init__(self):
        self.last_ball_toucher = {}  # Track who touched ball last per episode
        self.last_touch_height = {}  # Track height of last touch
        self.last_boost_amounts = {}  # Track boost usage for feathering rewards
        
    def reset(self, agents, initial_state, shared_info):
        """Reset tracking for new episode"""
        episode_id = id(initial_state)
        self.last_ball_toucher[episode_id] = None
        self.last_touch_height[episode_id] = 0
        self.last_boost_amounts[episode_id] = {}
        
    def get_rewards(self, state, previous_state, shared_info):
        """Calculate rewards for all agents"""
        rewards = {}
        episode_id = id(state)
        
        # Initialize tracking if needed
        if episode_id not in self.last_ball_toucher:
            self.last_ball_toucher[episode_id] = None
            self.last_touch_height[episode_id] = 0
            self.last_boost_amounts[episode_id] = {}
        
        for agent_id, car in state.cars.items():
            reward = 0.0
            prev_car = previous_state.cars[agent_id]
            
            # --- GOAL REWARDS (HUGE!) ---
            if state.goal_scored:
                if state.last_goal_scored_by_team == car.team_num:
                    # Our team scored!
                    if agent_id == state.last_touch_by_agent_id:
                        reward += 50.0  # Goal scorer gets MASSIVE reward
                    else:
                        reward += 15.0  # Teammates get assist/support reward
                else:
                    # Enemy scored :(
                    reward -= 20.0
            
            # --- BALL TOUCH REWARDS ---
            ball_dist = np.linalg.norm(car.physics.position - state.ball.position)
            prev_ball_dist = np.linalg.norm(prev_car.physics.position - previous_state.ball.position)
            
            if ball_dist < 150:  # Close to ball (touching range)
                ball_height = state.ball.position[2]
                
                # Basic touch reward
                if prev_ball_dist >= 150:  # Just touched
                    reward += 2.0
                    
                    # 🚀 AERIAL TOUCH REWARDS (Progressive)
                    if ball_height > 200:  # Low aerial
                        reward += 3.0
                    if ball_height > 400:  # Medium aerial
                        reward += 5.0
                    if ball_height > 600:  # High aerial
                        reward += 10.0
                    if ball_height > 800:  # CEILING SHOT HEIGHT!
                        reward += 20.0
                    
                    # 🎯 PASSING REWARD (touching ball toward teammate)
                    if self.last_ball_toucher.get(episode_id) is not None:
                        if self.last_ball_toucher[episode_id] != agent_id:
                            # Different teammate touched - potential pass!
                            prev_toucher_team = previous_state.cars[self.last_ball_toucher[episode_id]].team_num
                            if prev_toucher_team == car.team_num:
                                reward += 8.0  # PASS REWARD!
                                
                                # MID-AIR PASS (both touches were aerial)
                                if self.last_touch_height.get(episode_id, 0) > 200 and ball_height > 200:
                                    reward += 15.0  # AERIAL PASS!
                    
                    # Update last toucher
                    self.last_ball_toucher[episode_id] = agent_id
                    self.last_touch_height[episode_id] = ball_height
            
            # --- BOOST MANAGEMENT REWARDS ---
            current_boost = car.boost_amount
            prev_boost = prev_car.boost_amount
            boost_delta = current_boost - prev_boost
            
            # Initialize boost tracking
            if agent_id not in self.last_boost_amounts[episode_id]:
                self.last_boost_amounts[episode_id][agent_id] = current_boost
            
            # 💨 BOOST FEATHERING REWARD (using boost efficiently)
            if prev_boost > 0:
                # Reward small boost usage (feathering) instead of holding
                if 0 < abs(boost_delta) < 5:  # Small boost usage
                    reward += 0.5  # Feathering reward!
                elif abs(boost_delta) > 20:  # Wasting boost
                    reward -= 0.3
            
            # Penalty for being out of boost at critical moments
            if current_boost < 10 and ball_dist < 500:
                reward -= 0.5
            
            # Reward picking up boost pads
            if boost_delta > 10 and ball_dist > 500:  # Picked up boost while rotating
                reward += 1.0
            
            # --- VELOCITY & POSITIONING REWARDS ---
            speed = np.linalg.norm(car.physics.linear_velocity)
            
            # Ball chase reward (move toward ball when attacking)
            if car.team_num == BLUE_TEAM:
                attacking = car.physics.position[1] < state.ball.position[1]
            else:
                attacking = car.physics.position[1] > state.ball.position[1]
            
            if attacking and ball_dist < 2000:
                # Reward speed toward ball
                ball_dir = state.ball.position - car.physics.position
                ball_dir_norm = ball_dir / (np.linalg.norm(ball_dir) + 1e-5)
                velocity_toward_ball = np.dot(car.physics.linear_velocity, ball_dir_norm)
                reward += velocity_toward_ball * 0.001
            
            # --- TEAM POSITIONING (THE WALL!) ---
            # Find teammates
            teammates = [c for aid, c in state.cars.items() if c.team_num == car.team_num and aid != agent_id]
            
            if len(teammates) >= 2:
                # Calculate spread (want good triangular formation)
                teammate_positions = [tm.physics.position[:2] for tm in teammates]  # X, Y only
                my_pos = car.physics.position[:2]
                
                # Reward maintaining distance from teammates (not clumping)
                for tm_pos in teammate_positions:
                    dist_to_teammate = np.linalg.norm(my_pos - tm_pos)
                    if 800 < dist_to_teammate < 2000:  # Good spacing
                        reward += 0.3
                    elif dist_to_teammate < 400:  # Too close (clumping)
                        reward -= 0.5
            
            # --- DEFENSIVE POSITIONING ---
            if not attacking:
                # Reward being between ball and own goal
                if car.team_num == BLUE_TEAM:
                    goal_y = -5120
                else:
                    goal_y = 5120
                
                ball_to_goal = np.array([0, goal_y, 0]) - state.ball.position
                car_to_goal = np.array([0, goal_y, 0]) - car.physics.position
                
                # Reward being on defensive side of ball
                if np.sign(ball_to_goal[1]) == np.sign(car_to_goal[1]):
                    if np.abs(car_to_goal[1]) < np.abs(ball_to_goal[1]):
                        reward += 0.5  # Good defensive position!
            
            # --- SPEED REWARDS ---
            # Reward high speed when far from ball (rotation speed)
            if ball_dist > 1500 and speed > 1000:
                reward += 0.2
            
            # Reward supersonic when appropriate
            if speed > 2200 and ball_dist > 1000:
                reward += 0.5
            
            rewards[agent_id] = float(reward)
        
        return rewards


# ========================================
# 🏋️ GYMNASIUM WRAPPER
# ========================================

class GymnasiumWrapper(gym.Env):
    """Wraps RLGym multi-agent environment for SB3 (single agent from team perspective)"""
    
    def __init__(self, env):
        super().__init__()
        self.env = env
        
        # Reset to get observation spaces
        obs_dict = env.reset()
        sample_obs = next(iter(obs_dict.values()))
        
        # Create spaces based on sample
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(90)  # LookupTableAction has 90 actions
        
        # Track our controlled agents (blue team)
        self.our_agents = None
        
    def reset(self, **kwargs):
        obs_dict = self.env.reset()  # RLGym returns only obs_dict
        
        # Identify blue team agents (our Flamewalls)
        state = self.env._env._game_state
        self.our_agents = [aid for aid, car in state.cars.items() if car.team_num == BLUE_TEAM]
        
        # Return first blue agent's observation
        if self.our_agents:
            return obs_dict[self.our_agents[0]], {}
        return obs_dict[0], {}
    
    def step(self, action):
        # Create action dict for all our agents (same action = coordinated learning)
        action_dict = {}
        if self.our_agents:
            for agent_id in self.our_agents:
                action_dict[agent_id] = action
            
            # Random actions for opponents
            state = self.env._env._game_state
            for aid, car in state.cars.items():
                if car.team_num == ORANGE_TEAM:
                    action_dict[aid] = self.action_space.sample()
        
        # Step environment
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        # Combine rewards from all our agents
        total_reward = sum(reward_dict.get(aid, 0) for aid in self.our_agents) if self.our_agents else 0
        
        # Use first agent's observation
        obs = obs_dict[self.our_agents[0]] if self.our_agents else obs_dict[0]
        done = done_dict.get(self.our_agents[0], False) if self.our_agents else False
        truncated = trunc_dict.get(self.our_agents[0], False) if self.our_agents else False
        info = info_dict.get(self.our_agents[0], {}) if self.our_agents else {}
        
        # Combine done and truncated
        done = bool(done) or bool(truncated)
        
        return obs, float(total_reward), done, False, dict(info) if isinstance(info, dict) else {}
    
    def render(self, mode='human'):
        return None
    
    def close(self):
        self.env.close()


# ========================================
# 🎮 ENVIRONMENT CREATION
# ========================================

def make_env():
    """Create 3v3 RLGym environment with Flamewall team rewards"""
    
    # Reward function
    reward_fn = FlamewallTeamReward()
    
    # Terminal conditions
    terminal_condition = AnyCondition(
        TimeoutCondition(timeout_seconds=30),  # 30 second games
        GoalCondition(),
        NoTouchTimeoutCondition(timeout_seconds=15),  # Reset if no one touches for 15s
    )
    
    # State mutators (3v3 setup with kickoffs)
    mutators = MutatorSequence(
        FixedTeamSizeMutator(blue_size=3, orange_size=3),  # 3v3!
        KickoffMutator(),
    )
    
    # Create environment
    rlgym_env = RLGym(
        state_mutator=mutators,
        obs_builder=DefaultObs(zero_padding=11),  # MUST match training! 492 dims
        action_parser=LookupTableAction(),
        reward_fn=reward_fn,
        termination_cond=terminal_condition,
        transition_engine=RocketSimEngine(),
    )
    
    # Wrap for SB3
    return GymnasiumWrapper(rlgym_env)


# ========================================
# 🚀 MAIN TRAINING FUNCTION
# ========================================

def train():
    print("=" * 70)
    print(" 🔥🔥🔥 FLAMEWALL TEAM TRAINING 🔥🔥🔥")
    print("=" * 70)
    print()
    print("Training 3 Flamewalls to become an UNSTOPPABLE WALL!")
    print()
    
    # Model directory
    model_dir = 'models/flamewall_team'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    print("Creating 3v3 training environment...")
    env = DummyVecEnv([make_env])
    print("✓ Environment ready!")
    print()
    
    # Check for existing model
    existing_models = [f for f in os.listdir(model_dir) if f.endswith('_steps.zip')]
    model = None
    
    if existing_models:
        # Find highest step count
        step_counts = []
        for fname in existing_models:
            try:
                steps = int(fname.split('_')[-2])
                step_counts.append((steps, fname))
            except:
                pass
        
        if step_counts:
            step_counts.sort(reverse=True)
            latest_steps, latest_file = step_counts[0]
            model_path = os.path.join(model_dir, latest_file)
            
            print(f"Found existing model at {latest_steps} steps!")
            print(f"Resuming training from: {latest_file}")
            model = PPO.load(model_path, env=env)
            print("✓ Model loaded!")
            print()
    
    if model is None:
        print("No existing model found. Starting fresh training!")
        print()
        
        # Create new model with optimized hyperparameters
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={
                'net_arch': [512, 512, 256]  # Larger network for team play
            },
            tensorboard_log='logs/flamewall_team',
            device='cpu'
        )
    
    # Checkpoint callback (save every 100K steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=model_dir,
        name_prefix='flamewall_team',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    print("=" * 70)
    print(" 🚀 TRAINING CONFIGURATION")
    print("=" * 70)
    print()
    print("  Mode: 3v3 (3 Flamewalls vs 3 Random)")
    print("  Total Steps: 20,000,000")
    print("  Checkpoint: Every 100,000 steps")
    print("  Device: CPU")
    print()
    print("Focus Areas:")
    print("  ✓ Goal scoring (HUGE rewards)")
    print("  ✓ Passing between teammates")
    print("  ✓ Aerial shots and passes")
    print("  ✓ Boost feathering")
    print("  ✓ Team positioning (THE WALL!)")
    print("  ✓ Rotation and spacing")
    print()
    print("Press Ctrl+C to stop and save progress")
    print("=" * 70)
    print()
    
    try:
        # Train!
        model.learn(
            total_timesteps=20_000_000,
            callback=checkpoint_callback,
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(model_dir, 'flamewall_team_final.zip')
        model.save(final_path)
        print(f"\n✓ Training complete! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        interrupted_path = os.path.join(model_dir, 'flamewall_team_interrupted.zip')
        model.save(interrupted_path)
        print(f"✓ Progress saved to: {interrupted_path}")
    
    finally:
        env.close()


if __name__ == "__main__":
    train()
