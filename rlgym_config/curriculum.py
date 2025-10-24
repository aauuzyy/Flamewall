"""
Curriculum Learning System
Progressively increases difficulty as the bot improves
"""

import numpy as np
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
import rlgym.rocket_league.common_values as cv


class StateBallNearSetter:
    """Custom mutator: Place ball near the car for easy shots"""
    
    def __init__(self, distance_range=(500, 1500)):
        self.distance_range = distance_range
    
    def __call__(self, state):
        """Set ball near player"""
        # Get first car position
        if len(state.cars) > 0:
            car_pos = state.cars[0].physics.position
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(*self.distance_range)
            
            # Set ball position
            state.ball.position[0] = car_pos[0] + distance * np.cos(angle)
            state.ball.position[1] = car_pos[1] + distance * np.sin(angle)
            state.ball.position[2] = BALL_RADIUS + 50  # Slightly off ground
            
            # Low velocity
            state.ball.linear_velocity[:] = 0
            state.ball.angular_velocity[:] = 0
        
        return state


class StateMovingBallSetter:
    """Custom mutator: Ball with velocity for interception practice"""
    
    def __init__(self, speed_range=(500, 1500)):
        self.speed_range = speed_range
    
    def __call__(self, state):
        """Set ball with velocity"""
        # Random position
        state.ball.position[0] = np.random.uniform(-3000, 3000)
        state.ball.position[1] = np.random.uniform(-4000, 4000)
        state.ball.position[2] = np.random.uniform(BALL_RADIUS, 500)
        
        # Random velocity
        speed = np.random.uniform(*self.speed_range)
        angle = np.random.uniform(0, 2 * np.pi)
        
        state.ball.linear_velocity[0] = speed * np.cos(angle)
        state.ball.linear_velocity[1] = speed * np.sin(angle)
        state.ball.linear_velocity[2] = np.random.uniform(-200, 200)
        
        return state


class StateAerialSetter:
    """Custom mutator: Ball in air for aerial practice"""
    
    def __init__(self, height_range=(400, 1200)):
        self.height_range = height_range
    
    def __call__(self, state):
        """Set ball in air"""
        # Random aerial position
        state.ball.position[0] = np.random.uniform(-2000, 2000)
        state.ball.position[1] = np.random.uniform(-3000, 3000)
        state.ball.position[2] = np.random.uniform(*self.height_range)
        
        # Moderate velocity
        state.ball.linear_velocity[0] = np.random.uniform(-500, 500)
        state.ball.linear_velocity[1] = np.random.uniform(-500, 500)
        state.ball.linear_velocity[2] = np.random.uniform(-300, 300)
        
        # Set car somewhat near
        if len(state.cars) > 0:
            state.cars[0].physics.position[0] = state.ball.position[0] + np.random.uniform(-800, 800)
            state.cars[0].physics.position[1] = state.ball.position[1] + np.random.uniform(-800, 800)
            state.cars[0].physics.position[2] = 17  # On ground
        
        return state


class CurriculumSchedule:
    """
    Manages curriculum progression based on performance
    """
    
    def __init__(self):
        self.current_stage = 0
        self.performance_history = []
        self.stage_names = [
            "Stage 1: Stationary Ball Practice",
            "Stage 2: Moving Ball Interception", 
            "Stage 3: Kickoff Training",
            "Stage 4: Aerial Practice",
            "Stage 5: Full 1v1 Competitive"
        ]
        
        # Performance thresholds to advance (average reward)
        self.advancement_thresholds = [0.5, 1.0, 2.0, 3.0]
        
    def get_current_mutator(self):
        """Return the appropriate mutator for current stage"""
        if self.current_stage == 0:
            # Stage 1: Easy stationary ball shots
            return MutatorSequence(
                FixedTeamSizeMutator(blue_size=1, orange_size=0),  # No opponent
                StateBallNearSetter(distance_range=(300, 1000))
            )
        elif self.current_stage == 1:
            # Stage 2: Moving ball interception
            return MutatorSequence(
                FixedTeamSizeMutator(blue_size=1, orange_size=0),
                StateMovingBallSetter(speed_range=(300, 1200))
            )
        elif self.current_stage == 2:
            # Stage 3: Standard kickoffs
            return MutatorSequence(
                FixedTeamSizeMutator(blue_size=1, orange_size=1),  # Add opponent
                KickoffMutator()
            )
        elif self.current_stage == 3:
            # Stage 4: Aerial training
            return MutatorSequence(
                FixedTeamSizeMutator(blue_size=1, orange_size=1),
                StateAerialSetter(height_range=(400, 1500))
            )
        else:
            # Stage 5: Full competitive
            return MutatorSequence(
                FixedTeamSizeMutator(blue_size=1, orange_size=1),
                KickoffMutator()
            )
    
    def update_performance(self, avg_reward):
        """
        Update performance tracking and potentially advance curriculum
        """
        self.performance_history.append(avg_reward)
        
        # Check if we should advance (based on last 10 episodes)
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(self.performance_history[-10:])
            
            # Check if we've met the threshold for current stage
            if self.current_stage < len(self.advancement_thresholds):
                threshold = self.advancement_thresholds[self.current_stage]
                
                if recent_avg >= threshold:
                    self.advance_stage()
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        if self.current_stage < len(self.stage_names) - 1:
            self.current_stage += 1
            print(f"\n{'='*70}")
            print(f"🎓 CURRICULUM ADVANCEMENT!")
            print(f"   Now entering: {self.stage_names[self.current_stage]}")
            print(f"{'='*70}\n")
            self.performance_history = []  # Reset for new stage
    
    def get_stage_info(self):
        """Return current stage information"""
        return {
            'stage': self.current_stage,
            'name': self.stage_names[self.current_stage],
            'total_stages': len(self.stage_names)
        }


# Constants for custom mutators
BALL_RADIUS = 92.75
