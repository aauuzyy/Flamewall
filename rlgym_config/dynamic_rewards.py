"""
Dynamic Reward System
Adjusts reward weights based on what the bot is currently doing
Focuses learning on the most relevant actions
"""

import numpy as np
from rlgym.api import RewardFunction


class DynamicRewardSystem(RewardFunction):
    """
    Adapts reward weights based on current game state and activity
    """
    
    def __init__(self):
        super().__init__()
        self.last_ball_pos = None
        self.last_car_pos = {}
        self.last_ball_vel = 0
        self.touch_timer = {}
        self.aerial_time = {}
        
    def reset(self, initial_state, shared_info=None):
        self.last_ball_pos = None
        self.last_car_pos = {}
        self.last_ball_vel = 0
        self.touch_timer = {}
        self.aerial_time = {}
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        ball = shared_info['ball']
        
        for i, agent in enumerate(agents):
            player = agent.data
            agent_id = id(agent)
            reward = 0
            
            # Get current state
            car_pos = player.car_data.position
            car_vel = player.car_data.linear_velocity
            ball_pos = ball.position
            ball_vel = ball.linear_velocity
            ball_speed = np.linalg.norm(ball_vel)
            car_speed = np.linalg.norm(car_vel)
            
            # Distance to ball
            dist_to_ball = np.linalg.norm(ball_pos - car_pos)
            
            # Check if ball is on car roof (for flick training)
            ball_on_car = self._is_ball_on_car(ball_pos, car_pos, dist_to_ball)
            
            # === DYNAMIC REWARD WEIGHTING ===
            
            # SCENARIO 1: BALL ON CAR (Learning Flicks)
            if ball_on_car:
                # High reward for maintaining ball control
                reward += 2.0  # Base reward for having ball on car
                
                # Reward for moving with ball
                if car_speed > 500:
                    reward += car_speed / 2300.0 * 1.5
                
                # Reward for jumping (flick attempt)
                if car_pos[2] > 50:  # Car is jumping
                    reward += 1.0
                    
                    # Extra reward if ball gains upward velocity during jump
                    if ball_vel[2] > 500:  # Ball going up fast
                        reward += ball_vel[2] / 1000.0 * 3.0  # Big reward!
                
                # Reward for ball velocity toward goal during jump
                if player.team_num == 0:
                    goal_dir = np.array([1, 0, 0])
                else:
                    goal_dir = np.array([-1, 0, 0])
                
                vel_toward_goal = np.dot(goal_dir, ball_vel)
                if vel_toward_goal > 0:
                    reward += vel_toward_goal / 2300.0 * 2.0
            
            # SCENARIO 2: BALL IN AIR (Aerial Play)
            elif ball_pos[2] > 300:
                # Reward being near airborne ball
                if dist_to_ball < 500:
                    proximity_reward = (500 - dist_to_ball) / 500
                    reward += proximity_reward * 1.5
                    
                    # Extra reward if car is also in air
                    if car_pos[2] > 100:
                        height_match = 1.0 - abs(car_pos[2] - ball_pos[2]) / 1000
                        reward += max(0, height_match) * 2.0
                        
                        # Track aerial time
                        if agent_id not in self.aerial_time:
                            self.aerial_time[agent_id] = 0
                        self.aerial_time[agent_id] += 1
                        
                        # Reward sustained aerial control
                        if self.aerial_time[agent_id] > 10:
                            reward += 0.5
                
                # Reward hitting airborne ball toward goal
                if dist_to_ball < 200:
                    if player.team_num == 0:
                        goal_dir = np.array([1, 0, 0])
                    else:
                        goal_dir = np.array([-1, 0, 0])
                    
                    ball_to_goal = np.dot(goal_dir, ball_vel)
                    if ball_to_goal > 0:
                        reward += ball_to_goal / 2300.0 * 1.5
            
            # SCENARIO 3: GROUND PLAY (Ball close, on ground)
            elif dist_to_ball < 500 and ball_pos[2] < 200:
                # Reward approaching ball
                if agent_id in self.last_car_pos:
                    old_dist = np.linalg.norm(self.last_ball_pos - self.last_car_pos[agent_id])
                    if old_dist > dist_to_ball:  # Getting closer
                        reward += 0.3
                
                # Reward touching/dribbling
                if dist_to_ball < 150:
                    reward += 0.5
                    
                    # Reward ball speed increase (powerful touch)
                    if ball_speed > self.last_ball_vel + 200:
                        reward += (ball_speed - self.last_ball_vel) / 1000.0
                
                # Reward moving ball toward goal
                if player.team_num == 0:
                    goal_dir = np.array([1, 0, 0])
                else:
                    goal_dir = np.array([-1, 0, 0])
                
                ball_to_goal_vel = np.dot(goal_dir, ball_vel)
                if ball_to_goal_vel > 0:
                    reward += ball_to_goal_vel / 2300.0 * 1.0
            
            # SCENARIO 4: FAR FROM BALL (Positioning/Rotation)
            else:
                # Reward moving toward ball
                if self.last_ball_pos is not None:
                    ball_dir = ball_pos - car_pos
                    if np.linalg.norm(ball_dir) > 0:
                        ball_dir_norm = ball_dir / np.linalg.norm(ball_dir)
                        vel_to_ball = np.dot(ball_dir_norm, car_vel)
                        if vel_to_ball > 0:
                            reward += vel_to_ball / 2300.0 * 0.5
                
                # Reward supersonic speed when far
                if car_speed > 2200:
                    reward += 0.2
                
                # Reward facing ball
                forward = self._get_forward_vec(player.car_data.quaternion)
                if dist_to_ball > 0:
                    to_ball = (ball_pos - car_pos) / dist_to_ball
                    facing = np.dot(forward, to_ball)
                    if facing > 0:
                        reward += facing * 0.3
            
            # === BOOST MANAGEMENT (Always Active) ===
            boost = player.boost_amount
            
            # Penalize low boost when far from ball
            if boost < 0.2 and dist_to_ball > 2000:
                reward -= 0.1
            
            # Reward collecting boost when low
            if agent_id in self.last_car_pos:
                if boost > 0.8 and dist_to_ball > 1500:
                    reward += 0.05  # Good boost management
            
            # === SPECIAL EVENTS (High Impact) ===
            
            # Goal scored (detect from shared info or ball position)
            if abs(ball_pos[0]) > 5100 and abs(ball_pos[1]) < 900:
                # Check if it's in opponent's goal
                if (player.team_num == 0 and ball_pos[0] > 0) or \
                   (player.team_num == 1 and ball_pos[0] < 0):
                    reward += 20.0  # HUGE reward for scoring!
            
            # Save attempt (ball near own goal, moving away)
            own_goal_x = -5100 if player.team_num == 0 else 5100
            dist_to_own_goal = abs(ball_pos[0] - own_goal_x)
            
            if dist_to_own_goal < 1500 and abs(ball_pos[1]) < 1500:
                # Ball near own goal, reward if we hit it away
                if dist_to_ball < 200:
                    ball_dir_from_goal = ball_pos[0] - own_goal_x
                    ball_vel_dir = ball_vel[0]
                    
                    # If ball moving away from own goal after touch
                    if (ball_dir_from_goal > 0 and ball_vel_dir > 0) or \
                       (ball_dir_from_goal < 0 and ball_vel_dir < 0):
                        reward += 5.0  # Big reward for defense!
            
            # Store state for next iteration
            self.last_car_pos[agent_id] = car_pos.copy()
            
            rewards[i] = reward
        
        # Update global tracking
        if self.last_ball_pos is None:
            self.last_ball_pos = ball_pos.copy()
        else:
            self.last_ball_pos = ball_pos.copy()
        
        self.last_ball_vel = np.linalg.norm(ball_vel)
        
        # Reset aerial timer if on ground
        for i, agent in enumerate(agents):
            agent_id = id(agent)
            if agent.data.car_data.position[2] < 50:
                if agent_id in self.aerial_time:
                    self.aerial_time[agent_id] = 0
        
        return rewards
    
    def _is_ball_on_car(self, ball_pos, car_pos, dist):
        """Check if ball is on top of car (for dribbling/flicks)"""
        # Ball must be close horizontally
        horizontal_dist = np.sqrt((ball_pos[0] - car_pos[0])**2 + 
                                  (ball_pos[1] - car_pos[1])**2)
        
        # Ball must be above car
        height_diff = ball_pos[2] - car_pos[2]
        
        # Typical ball-on-car: horizontal dist < 120, height diff 80-180
        return horizontal_dist < 120 and 80 < height_diff < 200
    
    def _get_forward_vec(self, quat):
        """Extract forward direction from quaternion"""
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([np.cos(yaw), np.sin(yaw), 0])


class ActivityBasedRewardScaler:
    """
    Tracks bot performance and scales rewards to focus on weaknesses
    """
    
    def __init__(self):
        self.flick_success_rate = 0.1
        self.aerial_success_rate = 0.1
        self.ground_success_rate = 0.5
        
        self.flick_attempts = 0
        self.flick_successes = 0
        self.aerial_attempts = 0
        self.aerial_successes = 0
        
    def update_stats(self, activity_type, success):
        """Update success rates for different activities"""
        if activity_type == 'flick':
            self.flick_attempts += 1
            if success:
                self.flick_successes += 1
            self.flick_success_rate = self.flick_successes / max(1, self.flick_attempts)
        
        elif activity_type == 'aerial':
            self.aerial_attempts += 1
            if success:
                self.aerial_successes += 1
            self.aerial_success_rate = self.aerial_successes / max(1, self.aerial_attempts)
    
    def get_reward_multiplier(self, activity_type):
        """Return multiplier based on current weakness"""
        if activity_type == 'flick':
            # More weight if we're bad at flicks
            return 1.0 + (1.0 - self.flick_success_rate) * 2.0
        
        elif activity_type == 'aerial':
            return 1.0 + (1.0 - self.aerial_success_rate) * 1.5
        
        return 1.0
