"""
Tsunami Surge 1v3 Training Rewards
Special reward function designed for training one bot against 3 opponents
"""

import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData


class Tsunami1v3RewardFunction(RewardFunction):
    """
    Reward function optimized for 1v3 scenarios
    Encourages aggressive solo play, boost management, and overwhelming opponents
    """
    
    def __init__(
        self,
        goal_w=10.0,              # Scoring is CRUCIAL when outnumbered
        shot_w=3.0,               # Shots on goal are valuable
        save_w=5.0,               # Saves are critical (you're the only defender!)
        demo_w=4.0,               # Demos are KEY to evening the odds
        boost_pickup_w=0.5,       # Boost management is vital
        boost_penalty_w=0.3,      # Penalty for wasting boost
        speed_reward_w=0.3,       # Stay fast and aggressive
        ball_touch_w=0.8,         # Stay on the ball
        survival_w=0.5,           # Reward for not getting demo'd
        pressure_w=2.0,           # Constant offensive pressure
        kickoff_w=1.0,            # Win kickoffs
    ):
        super().__init__()
        self.goal_w = goal_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.demo_w = demo_w
        self.boost_pickup_w = boost_pickup_w
        self.boost_penalty_w = boost_penalty_w
        self.speed_reward_w = speed_reward_w
        self.ball_touch_w = ball_touch_w
        self.survival_w = survival_w
        self.pressure_w = pressure_w
        self.kickoff_w = kickoff_w
        
        # Tracking variables
        self.last_boost = {}
        self.last_demo_count = {}
        self.last_ball_touch = {}
        self.last_speed = {}
        
    def reset(self, initial_state: GameState):
        """Reset tracking variables"""
        self.last_boost = {}
        self.last_demo_count = {}
        self.last_ball_touch = {}
        self.last_speed = {}
        
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Calculate reward for the current state"""
        
        reward = 0.0
        player_id = player.car_id
        
        # Initialize tracking for new players
        if player_id not in self.last_boost:
            self.last_boost[player_id] = player.boost_amount
            self.last_demo_count[player_id] = 0
            self.last_ball_touch[player_id] = False
            self.last_speed[player_id] = 0
        
        # ===== GOAL REWARD =====
        # Massive reward for scoring (you're 1v3!)
        if state.last_touch == player_id:
            ball = state.ball
            # Check if ball is in opponent goal
            if abs(ball.position[1]) > 5000:  # Near goal line
                if (player.team_num == 0 and ball.position[1] > 5100) or \
                   (player.team_num == 1 and ball.position[1] < -5100):
                    reward += self.goal_w
        
        # ===== SHOT REWARD =====
        # Reward for taking shots (offensive pressure)
        ball_vel_towards_goal = 0
        if player.team_num == 0:
            ball_vel_towards_goal = state.ball.linear_velocity[1]  # Positive = towards orange goal
        else:
            ball_vel_towards_goal = -state.ball.linear_velocity[1]  # Negative = towards blue goal
        
        if ball_vel_towards_goal > 500 and np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2]) < 500:
            reward += self.shot_w * (ball_vel_towards_goal / 2300)  # Normalize by max ball speed
        
        # ===== SAVE REWARD =====
        # Critical to defend your own goal when alone
        ball_vel_towards_own_goal = 0
        if player.team_num == 0:
            ball_vel_towards_own_goal = -state.ball.linear_velocity[1]  # Negative = towards blue goal
        else:
            ball_vel_towards_own_goal = state.ball.linear_velocity[1]  # Positive = towards orange goal
        
        if ball_vel_towards_own_goal > 500:
            # Ball heading towards our goal
            own_goal_y = -5120 if player.team_num == 0 else 5120
            ball_dist_to_goal = abs(state.ball.position[1] - own_goal_y)
            player_dist_to_ball = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
            
            if player_dist_to_ball < 1000 and ball_dist_to_goal < 3000:
                # Player is close to ball and ball is threatening - reward defensive play
                reward += self.save_w * (1.0 - ball_dist_to_goal / 3000)
        
        # ===== DEMO REWARD =====
        # Demos even the odds in 1v3!
        opponents_alive = sum(1 for p in state.players if p.team_num != player.team_num and not p.is_demoed)
        if opponents_alive < 3:  # Someone got demo'd
            if player_id not in self.last_demo_count:
                self.last_demo_count[player_id] = 0
            
            demos_now = 3 - opponents_alive
            if demos_now > self.last_demo_count[player_id]:
                reward += self.demo_w * (demos_now - self.last_demo_count[player_id])
            self.last_demo_count[player_id] = demos_now
        
        # ===== BOOST MANAGEMENT =====
        # Smart boost usage
        boost_diff = player.boost_amount - self.last_boost[player_id]
        if boost_diff > 0:
            # Picked up boost
            reward += self.boost_pickup_w
        elif player.boost_amount < 20:
            # Low boost penalty (vulnerable when outnumbered)
            reward -= self.boost_penalty_w
        
        self.last_boost[player_id] = player.boost_amount
        
        # ===== SPEED REWARD =====
        # Stay fast and aggressive
        speed = np.linalg.norm(player.car_data.linear_velocity)
        if speed > 1500:  # Supersonic or near it
            reward += self.speed_reward_w
        
        # ===== BALL TOUCH REWARD =====
        # Control the ball constantly
        ball_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        if ball_dist < 300:  # Very close to ball
            reward += self.ball_touch_w
        
        # ===== SURVIVAL REWARD =====
        # Don't get demo'd (you're alone!)
        if not player.is_demoed:
            reward += self.survival_w
        else:
            reward -= 2.0  # Big penalty for dying
        
        # ===== PRESSURE REWARD =====
        # Constant offensive pressure - stay in opponent half
        opponent_goal_y = 5120 if player.team_num == 0 else -5120
        if (player.team_num == 0 and player.car_data.position[1] > 0) or \
           (player.team_num == 1 and player.car_data.position[1] < 0):
            # In opponent half
            reward += self.pressure_w * 0.1
            
            # Extra reward if ball is also in opponent half
            if (player.team_num == 0 and state.ball.position[1] > 0) or \
               (player.team_num == 1 and state.ball.position[1] < 0):
                reward += self.pressure_w * 0.2
        
        # ===== KICKOFF REWARD =====
        # Win kickoffs to maintain pressure
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:  # Kickoff
            ball_dist = np.linalg.norm(player.car_data.position[:2] - state.ball.position[:2])
            if ball_dist < 500:
                reward += self.kickoff_w
        
        return reward
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """Final reward at end of episode"""
        # Big bonus if you won while outnumbered!
        if player.team_num == 0 and state.blue_score > state.orange_score:
            return 50.0  # MASSIVE reward for winning 1v3
        elif player.team_num == 1 and state.orange_score > state.blue_score:
            return 50.0
        elif state.blue_score == state.orange_score:
            return 5.0  # Decent reward for tying 1v3
        else:
            return -5.0  # Small penalty for losing
