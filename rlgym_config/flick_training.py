"""
Ball-on-Car Spawner for Flick Training
Forces the bot to learn flicks by always spawning ball on top of car
"""

import numpy as np


class BallOnCarMutator:
    """
    Spawns ball directly on top of the car's roof
    Perfect for learning flicks and dribbling
    """
    
    def __init__(self, 
                 add_velocity=False, 
                 random_position=False,
                 spawn_height_range=(100, 150),
                 give_boost=True):
        """
        Args:
            add_velocity: Give car initial forward velocity
            random_position: Random field position (otherwise center field)
            spawn_height_range: Height range to spawn ball above car
            give_boost: Give car 100 boost to practice flicks
        """
        self.add_velocity = add_velocity
        self.random_position = random_position
        self.spawn_height_range = spawn_height_range
        self.give_boost = give_boost
    
    def apply(self, state, shared_info=None):
        """Mutate state to place ball on car (RLGym API)"""
        return self.__call__(state)
    
    def __call__(self, state):
        """Mutate state to place ball on car"""
        # Get cars - state.cars is a dict in RLGym 2.0
        if not hasattr(state, 'cars') or len(state.cars) == 0:
            return state
        
        # Get first car (player) - cars is a dict with agent IDs as keys
        car_ids = list(state.cars.keys())
        first_car_id = car_ids[0]
        car = state.cars[first_car_id]
        
        # Set car position
        if self.random_position:
            # Random position on field
            car.physics.position[0] = np.random.uniform(-2000, 2000)
            car.physics.position[1] = np.random.uniform(-2000, 2000)
        else:
            # Center field
            car.physics.position[0] = 0
            car.physics.position[1] = 0
        
        car.physics.position[2] = 17  # Ground level
        
        # Set car velocity
        if self.add_velocity:
            # Random forward velocity (500-1200)
            speed = np.random.uniform(500, 1200)
            angle = np.random.uniform(-np.pi/4, np.pi/4)  # Slightly random direction
            
            car.physics.linear_velocity[0] = speed * np.cos(angle)
            car.physics.linear_velocity[1] = speed * np.sin(angle)
            car.physics.linear_velocity[2] = 0
        else:
            # Start stationary
            car.physics.linear_velocity[:] = 0
        
        # Zero angular velocity
        car.physics.angular_velocity[:] = 0
        
        # Set car rotation (facing forward, upright)
        # Quaternion for neutral orientation
        car.physics.quaternion[0] = 1  # w
        car.physics.quaternion[1] = 0  # x
        car.physics.quaternion[2] = 0  # y
        car.physics.quaternion[3] = 0  # z
        
        # Give full boost
        if self.give_boost:
            car.boost_amount = 1.0
        
        # === PLACE BALL ON TOP OF CAR ===
        
        # Ball position: directly above car
        state.ball.position[0] = car.physics.position[0]
        state.ball.position[1] = car.physics.position[1]
        
        # Random height above car (for variety in training)
        ball_height = np.random.uniform(*self.spawn_height_range)
        state.ball.position[2] = car.physics.position[2] + ball_height
        
        # Ball velocity: match car velocity (so it stays on car initially)
        state.ball.linear_velocity[0] = car.physics.linear_velocity[0]
        state.ball.linear_velocity[1] = car.physics.linear_velocity[1]
        state.ball.linear_velocity[2] = 0  # No vertical velocity
        
        # Zero ball angular velocity
        state.ball.angular_velocity[:] = 0
        
        # Set opponent car far away (if exists)
        if len(state.cars) > 1:
            opponent_id = car_ids[1] if len(car_ids) > 1 else None
            if opponent_id:
                opponent = state.cars[opponent_id]
                # Place opponent in their goal
                if opponent.team_num == 0:  # Blue
                    opponent.physics.position[0] = -4500
                else:  # Orange
                    opponent.physics.position[0] = 4500
                
                opponent.physics.position[1] = 0
                opponent.physics.position[2] = 17
                opponent.physics.linear_velocity[:] = 0
                opponent.physics.angular_velocity[:] = 0
        
        return state


class ProgressiveFlickTraining:
    """
    Progressive difficulty for flick training
    Starts easy, gets harder as bot improves
    """
    
    def __init__(self):
        self.stage = 0
        self.success_count = 0
        self.attempt_count = 0
        
        # Different stages of flick training
        self.stages = [
            # Stage 0: Stationary, ball perfectly centered
            {'velocity': False, 'height': (110, 120), 'random_pos': False},
            
            # Stage 1: Stationary, varied height
            {'velocity': False, 'height': (90, 140), 'random_pos': False},
            
            # Stage 2: Moving slowly, perfect height
            {'velocity': True, 'height': (110, 120), 'random_pos': False},
            
            # Stage 3: Moving, varied height
            {'velocity': True, 'height': (80, 150), 'random_pos': False},
            
            # Stage 4: Moving, varied height, random positions
            {'velocity': True, 'height': (80, 150), 'random_pos': True},
        ]
    
    def apply(self, state, shared_info=None):
        """Apply appropriate mutator based on current stage (RLGym API)"""
        return self.__call__(state)
    
    def __call__(self, state):
        """Apply appropriate mutator based on current stage"""
        stage_config = self.stages[min(self.stage, len(self.stages) - 1)]
        
        mutator = BallOnCarMutator(
            add_velocity=stage_config['velocity'],
            random_position=stage_config['random_pos'],
            spawn_height_range=stage_config['height'],
            give_boost=True
        )
        
        return mutator(state)
    
    def update_progress(self, success):
        """Update training progress, advance stage if ready"""
        self.attempt_count += 1
        if success:
            self.success_count += 1
        
        # Advance stage after 100 attempts with >60% success rate
        if self.attempt_count >= 100:
            success_rate = self.success_count / self.attempt_count
            
            if success_rate > 0.6 and self.stage < len(self.stages) - 1:
                self.stage += 1
                print(f"\n🎓 FLICK TRAINING ADVANCEMENT!")
                print(f"   Success rate: {success_rate*100:.1f}%")
                print(f"   Moving to Stage {self.stage + 1}/{len(self.stages)}")
                print(f"   {self._get_stage_description()}\n")
                
                # Reset counters
                self.success_count = 0
                self.attempt_count = 0
    
    def _get_stage_description(self):
        """Get description of current stage"""
        descriptions = [
            "Stationary car, perfect ball placement",
            "Stationary car, varied ball height",
            "Moving car, perfect ball placement",
            "Moving car, varied ball height",
            "Full randomization - Master level!"
        ]
        return descriptions[min(self.stage, len(descriptions) - 1)]


class FlickSuccessDetector:
    """
    Detects when a successful flick has been performed
    """
    
    def __init__(self):
        self.last_ball_on_car = False
        self.last_ball_vel_z = 0
        self.last_ball_vel = 0
    
    def check_flick(self, ball_pos, car_pos, ball_vel, ball_on_car):
        """
        Returns True if a flick was just performed successfully
        
        Flick criteria:
        - Ball was on car recently
        - Ball now has significant upward velocity
        - Ball speed increased significantly
        """
        success = False
        
        # Ball just left car with high upward velocity
        if self.last_ball_on_car and not ball_on_car:
            ball_speed_z = ball_vel[2]
            ball_speed = np.linalg.norm(ball_vel)
            
            # Successful flick: upward velocity > 600, total speed increase > 500
            if ball_speed_z > 600 and ball_speed > self.last_ball_vel + 500:
                success = True
        
        # Update tracking
        self.last_ball_on_car = ball_on_car
        self.last_ball_vel_z = ball_vel[2]
        self.last_ball_vel = np.linalg.norm(ball_vel)
        
        return success
