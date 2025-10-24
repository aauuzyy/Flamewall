"""
Custom terminal conditions for RLGym
Defines when training episodes should end
"""

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
    NoTouchTimeoutCondition,
)


class CustomTerminalCondition(TerminalCondition):
    """
    Combines multiple terminal conditions:
    - End episode after a goal is scored
    - End episode after timeout (to prevent infinite episodes)
    - End episode if no one touches ball for too long
    """
    
    def __init__(self, max_steps=5000, no_touch_timeout_steps=500):
        """
        Args:
            max_steps: Maximum steps before ending episode
            no_touch_timeout_steps: Steps without ball touch before ending
        """
        super().__init__()
        
        # Combine multiple conditions
        self.timeout = TimeoutCondition(max_steps)
        self.goal_scored = GoalScoredCondition()
        self.no_touch = NoTouchTimeoutCondition(no_touch_timeout_steps)
        
    def reset(self, initial_state: GameState):
        """
        Reset all conditions
        """
        self.timeout.reset(initial_state)
        self.goal_scored.reset(initial_state)
        self.no_touch.reset(initial_state)
    
    def is_terminal(self, current_state: GameState) -> bool:
        """
        Check if episode should end
        """
        # End if ANY condition is met
        return (self.timeout.is_terminal(current_state) or
                self.goal_scored.is_terminal(current_state) or
                self.no_touch.is_terminal(current_state))
