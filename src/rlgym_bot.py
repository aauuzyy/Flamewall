"""
RLGym-trained bot for RLBot
This bot loads a trained model and uses it to play
"""

import os
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# Try to import stable-baselines3 and handle if not installed
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False
    print("Warning: stable-baselines3 not installed. Bot will not work.")


class RLGymBot(BaseAgent):
    """
    Bot that uses a trained RLGym model to make decisions
    """
    
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        self.vec_normalize = None
        self.controls = SimpleControllerState()
        
        # Action mapping for discrete actions
        # This maps the discrete action outputs to controller inputs
        self.action_map = self._create_action_map()
        
    def initialize_agent(self):
        """
        Load the trained model
        """
        if not HAVE_SB3:
            print("Cannot load model: stable-baselines3 not installed")
            return
            
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "PPO", "flamewall_final_model.zip")
        vec_normalize_path = os.path.join(os.path.dirname(__file__), "..", "models", "PPO", "vec_normalize.pkl")
        
        try:
            if os.path.exists(model_path):
                print(f"Loading trained model from {model_path}")
                self.model = PPO.load(model_path)
                
                # Load normalization if it exists
                if os.path.exists(vec_normalize_path):
                    print(f"Loading VecNormalize from {vec_normalize_path}")
                    # Note: We'll normalize observations manually since we're not using the full environment
                else:
                    print("No VecNormalize found, using raw observations")
            else:
                print(f"Model not found at {model_path}")
                print("Train a model first using train_rlgym.py")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Use the trained model to generate controls
        """
        if self.model is None:
            # No model loaded, use simple default behavior
            return self._default_behavior(packet)
        
        try:
            # Build observation from packet
            obs = self._build_observation(packet)
            
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Convert action to controls
            controls = self._action_to_controls(action)
            
            return controls
            
        except Exception as e:
            print(f"Error in get_output: {e}")
            return self._default_behavior(packet)
    
    def _build_observation(self, packet: GameTickPacket) -> np.ndarray:
        """
        Build observation vector from game packet
        This should match the observation builder used in training
        """
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball.physics
        
        inverted = self.team == 1
        
        # Get positions
        if inverted:
            player_pos = np.array([-my_car.physics.location.x,
                                   -my_car.physics.location.y,
                                   my_car.physics.location.z])
            player_vel = np.array([-my_car.physics.velocity.x,
                                   -my_car.physics.velocity.y,
                                   my_car.physics.velocity.z])
            ball_pos = np.array([-ball.location.x, -ball.location.y, ball.location.z])
            ball_vel = np.array([-ball.velocity.x, -ball.velocity.y, ball.velocity.z])
        else:
            player_pos = np.array([my_car.physics.location.x,
                                   my_car.physics.location.y,
                                   my_car.physics.location.z])
            player_vel = np.array([my_car.physics.velocity.x,
                                   my_car.physics.velocity.y,
                                   my_car.physics.velocity.z])
            ball_pos = np.array([ball.location.x, ball.location.y, ball.location.z])
            ball_vel = np.array([ball.velocity.x, ball.velocity.y, ball.velocity.z])
        
        # Normalize
        player_pos_norm = player_pos / np.array([4096, 5120, 2044])
        ball_pos_norm = ball_pos / np.array([4096, 5120, 2044])
        player_vel_norm = player_vel / 2300
        ball_vel_norm = ball_vel / 6000
        
        relative_pos = (ball_pos - player_pos) / np.array([4096, 5120, 2044])
        relative_vel = (ball_vel - player_vel) / 2300
        
        # Rotation
        rotation = my_car.physics.rotation
        forward = self._rotation_to_forward(rotation)
        up = self._rotation_to_up(rotation)
        
        if inverted:
            forward = np.array([-forward[0], -forward[1], forward[2]])
            up = np.array([-up[0], -up[1], up[2]])
        
        # Angular velocity
        ang_vel = np.array([my_car.physics.angular_velocity.x,
                           my_car.physics.angular_velocity.y,
                           my_car.physics.angular_velocity.z]) / 5.5
        if inverted:
            ang_vel = np.array([-ang_vel[0], -ang_vel[1], ang_vel[2]])
        
        # Build observation array
        obs = np.concatenate([
            player_pos_norm,
            player_vel_norm,
            ball_pos_norm,
            ball_vel_norm,
            relative_pos,
            relative_vel,
            forward,
            up,
            ang_vel,
            [my_car.boost / 100.0],
            [float(my_car.has_wheel_contact)],
            [float(my_car.jumped)],  # Approximation for has_flip
            [float(my_car.is_demolished)],
        ])
        
        return obs.astype(np.float32)
    
    def _rotation_to_forward(self, rotation):
        """Convert rotation to forward vector"""
        pitch = rotation.pitch
        yaw = rotation.yaw
        
        forward = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])
        return forward
    
    def _rotation_to_up(self, rotation):
        """Convert rotation to up vector"""
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll
        
        up = np.array([
            -np.cos(yaw) * np.sin(roll) * np.sin(pitch) - np.sin(yaw) * np.cos(roll),
            -np.sin(yaw) * np.sin(roll) * np.sin(pitch) + np.cos(yaw) * np.cos(roll),
            np.cos(pitch) * np.sin(roll)
        ])
        return up
    
    def _action_to_controls(self, action: np.ndarray) -> SimpleControllerState:
        """
        Convert discrete action to controller state
        """
        if isinstance(action, np.ndarray):
            action = int(action[0])
        
        action_values = self.action_map.get(action, [0, 0, 0, 0, 0, 0, 0, 0])
        
        controls = SimpleControllerState()
        controls.throttle = action_values[0]
        controls.steer = action_values[1]
        controls.pitch = action_values[2]
        controls.yaw = action_values[3]
        controls.roll = action_values[4]
        controls.jump = action_values[5] == 1
        controls.boost = action_values[6] == 1
        controls.handbrake = action_values[7] == 1
        
        return controls
    
    def _create_action_map(self):
        """
        Create mapping from discrete actions to controller inputs
        This is a simplified action space for easier learning
        """
        # Format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        actions = {
            0: [0, 0, 0, 0, 0, 0, 0, 0],      # No-op
            1: [1, 0, 0, 0, 0, 0, 0, 0],      # Forward
            2: [-1, 0, 0, 0, 0, 0, 0, 0],     # Backward
            3: [1, -1, 0, 0, 0, 0, 0, 0],     # Forward + Left
            4: [1, 1, 0, 0, 0, 0, 0, 0],      # Forward + Right
            5: [1, 0, 0, 0, 0, 0, 1, 0],      # Forward + Boost
            6: [1, -1, 0, 0, 0, 0, 1, 0],     # Forward + Left + Boost
            7: [1, 1, 0, 0, 0, 0, 1, 0],      # Forward + Right + Boost
            8: [1, 0, 0, 0, 0, 1, 0, 0],      # Forward + Jump
            9: [0, 0, -1, 0, 0, 0, 0, 0],     # Pitch down (aerial)
            10: [0, 0, 1, 0, 0, 0, 0, 0],     # Pitch up (aerial)
            11: [0, 0, 0, -1, 0, 0, 0, 0],    # Yaw left
            12: [0, 0, 0, 1, 0, 0, 0, 0],     # Yaw right
            13: [1, 0, 0, 0, 0, 0, 0, 1],     # Forward + Powerslide
            14: [0, -1, 0, 0, 0, 0, 0, 1],    # Left + Powerslide
            15: [0, 1, 0, 0, 0, 0, 0, 1],     # Right + Powerslide
        }
        return actions
    
    def _default_behavior(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Simple default behavior when no model is loaded
        """
        controls = SimpleControllerState()
        
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball.physics
        
        # Simple chase ball behavior
        car_to_ball = [
            ball.location.x - my_car.physics.location.x,
            ball.location.y - my_car.physics.location.y
        ]
        
        # Angle to ball
        import math
        angle_to_ball = math.atan2(car_to_ball[1], car_to_ball[0])
        car_angle = my_car.physics.rotation.yaw
        angle_diff = angle_to_ball - car_angle
        
        # Normalize angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        controls.throttle = 1.0
        controls.steer = angle_diff * 2
        controls.boost = abs(angle_diff) < 0.3
        
        return controls
