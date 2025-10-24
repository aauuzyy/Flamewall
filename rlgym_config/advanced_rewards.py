"""
Advanced Reward Functions for Faster Learning
Comprehensive reward shaping to guide the bot toward good play
"""

import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, BALL_RADIUS, CAR_MAX_SPEED
from rlgym.rocket_league.math import quat_to_euler


class VelocityPlayerToBallReward(RewardFunction):
    """Reward agent for moving toward the ball"""
    
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection
    
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        
        for i, agent in enumerate(agents):
            player = agent.data
            ball = shared_info['ball']
            
            # Vector from player to ball
            diff = ball.position - player.car_data.position
            norm_diff = diff / np.linalg.norm(diff)
            
            # Player velocity
            vel = player.car_data.linear_velocity
            
            if self.use_scalar_projection:
                # Scalar projection of velocity onto player->ball direction
                reward = np.dot(norm_diff, vel) / CAR_MAX_SPEED
            else:
                # Simple dot product
                reward = np.dot(norm_diff, vel)
                
            rewards[i] = reward
            
        return rewards


class VelocityBallToGoalReward(RewardFunction):
    """Reward agent for hitting ball toward opponent goal"""
    
    def __init__(self, own_goal_penalty=True):
        super().__init__()
        self.own_goal_penalty = own_goal_penalty
    
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        ball = shared_info['ball']
        
        for i, agent in enumerate(agents):
            player = agent.data
            
            # Determine target goal based on team
            if player.team_num == BLUE_TEAM:
                # Blue team attacks orange goal (positive x)
                objective = np.array([1, 0, 0])
            else:
                # Orange team attacks blue goal (negative x)
                objective = np.array([-1, 0, 0])
            
            # Reward ball velocity toward target goal
            vel = ball.linear_velocity
            reward = np.dot(objective, vel) / CAR_MAX_SPEED
            
            # Optional: penalize hitting toward own goal
            if self.own_goal_penalty:
                wrong_dir_reward = -np.dot(objective, vel) / CAR_MAX_SPEED
                if wrong_dir_reward > 0:
                    reward -= wrong_dir_reward * 0.5
            
            rewards[i] = reward
            
        return rewards


class FacingBallReward(RewardFunction):
    """Reward agent for facing the ball"""
    
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        
        for i, agent in enumerate(agents):
            player = agent.data
            ball = shared_info['ball']
            
            # Vector from player to ball
            diff = ball.position - player.car_data.position
            norm_diff = diff / np.linalg.norm(diff)
            
            # Player forward direction from quaternion
            quat = player.car_data.quaternion
            euler = quat_to_euler(quat)
            yaw = euler[2]
            forward = np.array([np.cos(yaw), np.sin(yaw), 0])
            
            # Dot product gives cosine of angle between forward and ball direction
            reward = np.dot(forward, norm_diff)
            
            rewards[i] = max(reward, 0)  # Only positive when facing ball
            
        return rewards


class TouchVelocityReward(RewardFunction):
    """Reward powerful touches"""
    
    def __init__(self):
        super().__init__()
        self.last_touch_time = {}
        self.last_ball_vel = None
    
    def reset(self, initial_state, shared_info=None):
        self.last_touch_time = {}
        self.last_ball_vel = None
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        ball = shared_info['ball']
        
        current_vel = np.linalg.norm(ball.linear_velocity)
        
        # Check if ball velocity increased (indicating a touch)
        if self.last_ball_vel is not None:
            vel_increase = current_vel - self.last_ball_vel
            
            if vel_increase > 100:  # Significant velocity change
                # Find closest player (likely touched)
                min_dist = float('inf')
                closest_idx = -1
                
                for i, agent in enumerate(agents):
                    player = agent.data
                    dist = np.linalg.norm(ball.position - player.car_data.position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                # Reward based on velocity increase
                if closest_idx >= 0 and min_dist < 300:  # Within reasonable touch distance
                    rewards[closest_idx] = vel_increase / 1000.0
        
        self.last_ball_vel = current_vel
        return rewards


class BoostDisciplineReward(RewardFunction):
    """Penalize boost waste, reward boost collection when low"""
    
    def __init__(self):
        super().__init__()
        self.last_boost = {}
    
    def reset(self, initial_state, shared_info=None):
        self.last_boost = {}
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        
        for i, agent in enumerate(agents):
            player = agent.data
            current_boost = player.boost_amount
            agent_id = id(agent)
            
            reward = 0
            
            if agent_id in self.last_boost:
                boost_change = current_boost - self.last_boost[agent_id]
                
                # Collected boost while low - good!
                if boost_change > 0 and self.last_boost[agent_id] < 0.3:
                    reward += 0.1
                
                # Used boost while full - wasteful!
                elif boost_change < 0 and self.last_boost[agent_id] > 0.8:
                    reward -= 0.05
            
            self.last_boost[agent_id] = current_boost
            rewards[i] = reward
            
        return rewards


class AerialReward(RewardFunction):
    """Reward aerial play"""
    
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        ball = shared_info['ball']
        
        for i, agent in enumerate(agents):
            player = agent.data
            
            # Ball must be in the air
            if ball.position[2] > 300:  # Ball above 300 units
                # Player close to ball
                dist = np.linalg.norm(ball.position - player.car_data.position)
                
                if dist < 400:
                    # Reward being near airborne ball
                    height_reward = min(ball.position[2] / 2000, 1.0)
                    proximity_reward = max(0, 1.0 - dist / 400)
                    rewards[i] = height_reward * proximity_reward * 0.5
        
        return rewards


class EventReward(RewardFunction):
    """Reward/penalize major events"""
    
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        
        # Check for goals, saves, demos, etc. from shared_info
        # This would need to track game state changes
        
        return rewards
