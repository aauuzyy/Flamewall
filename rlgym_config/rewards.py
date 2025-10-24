"""
Custom reward function for RLGym training
Rewards the agent for good Rocket League behaviors
"""

import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.common_rewards import (
    VelocityPlayerToBallReward,
    VelocityBallToGoalReward,
    EventReward,
    SaveBoostReward,
    LiuDistancePlayerToBallReward,
    FaceBallReward,
    TouchBallReward,
    AlignBallGoal,
)
from rlgym.utils.reward_functions.combined_reward import CombinedReward


class CustomReward(RewardFunction):
    """
    Custom reward function that combines multiple reward types
    This encourages the agent to:
    - Move toward the ball
    - Hit the ball toward the opponent's goal
    - Face the ball
    - Touch the ball
    - Score goals
    - Save boost
    """
    
    def __init__(self):
        super().__init__()
        
        # Create a combined reward with weighted components
        self.reward_fn = CombinedReward(
            (
                VelocityPlayerToBallReward(),      # Reward moving toward ball
                VelocityBallToGoalReward(),         # Reward ball moving to goal
                EventReward(
                    team_goal=10.0,                 # Big reward for scoring
                    concede=-10.0,                  # Big penalty for being scored on
                    shot=2.0,                       # Reward for shots
                    save=3.0,                       # Reward for saves
                    demo=1.0,                       # Small reward for demos
                ),
                TouchBallReward(),                  # Reward for touching ball
                FaceBallReward(),                   # Reward for facing ball
                LiuDistancePlayerToBallReward(),    # Reward for being close to ball
                AlignBallGoal(),                    # Reward for aligning ball with goal
                SaveBoostReward(),                  # Reward for conserving boost
            ),
            (
                1.0,   # VelocityPlayerToBall weight
                1.5,   # VelocityBallToGoal weight (important!)
                1.0,   # EventReward weight
                0.5,   # TouchBall weight
                0.3,   # FaceBall weight
                0.5,   # LiuDistance weight
                1.0,   # AlignBallGoal weight
                0.1,   # SaveBoost weight
            )
        )
    
    def reset(self, initial_state: GameState):
        """
        Called each time the environment is reset
        """
        self.reward_fn.reset(initial_state)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate and return the reward for the current state
        """
        return self.reward_fn.get_reward(player, state, previous_action)
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate final reward at episode end
        """
        return self.reward_fn.get_final_reward(player, state, previous_action)
