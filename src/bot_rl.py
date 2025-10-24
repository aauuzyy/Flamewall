"""
Flamewall RL Bot - Uses trained reinforcement learning model
"""

import os
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class FlamewallRL(BaseAgent):
    """
    RL Bot that uses the trained Stable-Baselines3 PPO model
    """
    
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        self.model_loaded = False
        
    def initialize_agent(self):
        """Load the trained model"""
        try:
            # Import here to avoid loading if not needed
            from stable_baselines3 import PPO
            
            # Path to trained model (use the best available)
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rlgym_real')
            
            # Try to load the final model, or fall back to latest checkpoint
            model_files = [
                'flamewall_final.zip',
                'flamewall_1000000_steps.zip',
                'flamewall_950000_steps.zip',
            ]
            
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                if os.path.exists(model_path):
                    print(f"Loading trained model: {model_path}")
                    self.model = PPO.load(model_path)
                    self.model_loaded = True
                    print("✓ Model loaded successfully!")
                    break
            
            if not self.model_loaded:
                print("⚠️  No trained model found! Train a model first with: py -3.10 train_real.py")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Get controls from the trained RL model
        """
        controls = SimpleControllerState()
        
        if not self.model_loaded or self.model is None:
            # Fallback: just sit still if model isn't loaded
            return controls
        
        try:
            # Convert game state to observation (simplified version)
            obs = self._packet_to_observation(packet)
            
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Convert action to controls
            controls = self._action_to_controls(action)
            
        except Exception as e:
            print(f"Error getting model output: {e}")
            # Return default controls on error
            pass
        
        return controls
    
    def _packet_to_observation(self, packet: GameTickPacket):
        """
        Convert RLBot packet to RLGym observation format
        This is a simplified version - the actual observation builder is more complex
        """
        my_car = packet.game_cars[self.index]
        ball = packet.game_ball.physics
        
        # Basic observation: car position, velocity, ball position, velocity
        # This is simplified - RLGym's DefaultObs creates a 492-dim vector
        obs = []
        
        # Car info
        obs.extend([
            my_car.physics.location.x / 4096,  # Normalize positions
            my_car.physics.location.y / 4096,
            my_car.physics.location.z / 2044,
            my_car.physics.velocity.x / 2300,  # Normalize velocities
            my_car.physics.velocity.y / 2300,
            my_car.physics.velocity.z / 2300,
            my_car.physics.rotation.pitch,
            my_car.physics.rotation.yaw,
            my_car.physics.rotation.roll,
            my_car.physics.angular_velocity.x / 5.5,
            my_car.physics.angular_velocity.y / 5.5,
            my_car.physics.angular_velocity.z / 5.5,
        ])
        
        # Ball info
        obs.extend([
            ball.location.x / 4096,
            ball.location.y / 4096,
            ball.location.z / 2044,
            ball.velocity.x / 2300,
            ball.velocity.y / 2300,
            ball.velocity.z / 2300,
        ])
        
        # Boost amount
        obs.append(my_car.boost / 100)
        
        # Pad to expected size (492 dimensions for DefaultObs)
        # Fill remaining with zeros
        while len(obs) < 492:
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _action_to_controls(self, action):
        """
        Convert RLGym action (0-89) to RLBot controls
        RLGym uses a lookup table for 90 discrete actions
        """
        controls = SimpleControllerState()
        
        # RLGym LookupTableAction has 90 actions
        # This is a simplified mapping - the actual lookup table is more complex
        # For now, just map to basic controls
        
        # Simple action mapping (0-89)
        # Actions represent different combinations of throttle, steer, pitch, yaw, roll, jump, boost, handbrake
        
        # Extract action components (this is approximate - real mapping is in LookupTableAction)
        throttle_idx = action % 3  # 0=back, 1=none, 2=forward
        steer_idx = (action // 3) % 3  # 0=left, 1=straight, 2=right
        jump_idx = (action // 9) % 2  # 0=no jump, 1=jump
        boost_idx = (action // 18) % 2  # 0=no boost, 1=boost
        handbrake_idx = (action // 36) % 2  # 0=no handbrake, 1=handbrake
        
        # Throttle
        if throttle_idx == 0:
            controls.throttle = -1.0
        elif throttle_idx == 1:
            controls.throttle = 0.0
        else:
            controls.throttle = 1.0
        
        # Steer
        if steer_idx == 0:
            controls.steer = -1.0
        elif steer_idx == 1:
            controls.steer = 0.0
        else:
            controls.steer = 1.0
        
        # Jump
        controls.jump = (jump_idx == 1)
        
        # Boost
        controls.boost = (boost_idx == 1)
        
        # Handbrake
        controls.handbrake = (handbrake_idx == 1)
        
        # Pitch/Yaw/Roll (simplified)
        pitch_idx = (action // 72) % 3
        if pitch_idx == 0:
            controls.pitch = -1.0
        elif pitch_idx == 1:
            controls.pitch = 0.0
        else:
            controls.pitch = 1.0
        
        return controls
