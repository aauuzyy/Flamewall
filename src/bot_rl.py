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
        self.obs_builder = None
        self.action_parser = None
        
    def initialize_agent(self):
        """Load the trained model"""
        try:
            # Import here to avoid loading if not needed
            from stable_baselines3 import PPO
            from rlgym.rocket_league.obs_builders import DefaultObs
            from rlgym.rocket_league.action_parsers import LookupTableAction
            import glob
            
            # Check both training directories and use the most recent
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_dirs = [
                os.path.join(base_dir, 'models', 'flamewall_team'),  # Main training directory
                os.path.join(base_dir, 'models', 'rlgym_advanced'),
                os.path.join(base_dir, 'models', 'rlgym_real'),
            ]
            
            best_model = None
            best_steps = 0
            
            for model_dir in model_dirs:
                if not os.path.exists(model_dir):
                    continue
                    
                # Find all checkpoint files
                checkpoints = glob.glob(os.path.join(model_dir, '*_steps.zip'))
                for checkpoint in checkpoints:
                    # Extract step count from filename
                    try:
                        steps = int(checkpoint.split('_')[-2])
                        if steps > best_steps:
                            best_steps = steps
                            best_model = checkpoint
                    except:
                        pass
                
                # Also check for final/interrupted models
                for special_file in ['flamewall_final.zip', 'flamewall_advanced_final.zip', 'flamewall_interrupted.zip']:
                    special_path = os.path.join(model_dir, special_file)
                    if os.path.exists(special_path):
                        # Consider these as "latest" if we haven't found a better one
                        if not best_model:
                            best_model = special_path
            
            if best_model:
                print(f"Loading trained model: {best_model}")
                print(f"  Training steps: {best_steps if best_steps > 0 else 'final'}")
                self.model = PPO.load(best_model)
                self.obs_builder = DefaultObs(zero_padding=11)  # MUST match training!
                self.action_parser = LookupTableAction()
                self.model_loaded = True
                print("✓ Model loaded successfully!")
            else:
                print("⚠️  No trained model found! Train a model first.")
                
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
            # Convert packet to RLGym state
            rlgym_state = self._packet_to_rlgym_state(packet)
            
            # Build observation using RLGym's obs builder
            # RLGym 2.0 API: build_obs(state, shared_info) returns {player_id: obs_array}
            shared_info = {}  # Empty dict for shared info
            obs_dict = self.obs_builder.build_obs(rlgym_state, shared_info)
            obs = obs_dict[self.index]
            
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Convert action to controls using RLGym's action parser
            controls = self._rlgym_action_to_controls(action, rlgym_state)
            
        except Exception as e:
            print(f"Error getting model output: {e}")
            # Return default controls on error
            pass
        
        return controls
    
    def _packet_to_rlgym_state(self, packet: GameTickPacket):
        """Convert RLBot packet to RLGym state format"""
        from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
        
        class SimpleState:
            def __init__(self):
                self.cars = {}
                self.ball = type('obj', (object,), {
                    'position': np.array([
                        packet.game_ball.physics.location.x,
                        packet.game_ball.physics.location.y,
                        packet.game_ball.physics.location.z
                    ]),
                    'linear_velocity': np.array([
                        packet.game_ball.physics.velocity.x,
                        packet.game_ball.physics.velocity.y,
                        packet.game_ball.physics.velocity.z
                    ]),
                    'angular_velocity': np.array([
                        packet.game_ball.physics.angular_velocity.x,
                        packet.game_ball.physics.angular_velocity.y,
                        packet.game_ball.physics.angular_velocity.z
                    ])
                })()
                
                for i in range(packet.num_cars):
                    car = packet.game_cars[i]
                    # Convert rotation to quaternion (simplified)
                    from math import cos, sin
                    pitch, yaw, roll = car.physics.rotation.pitch, car.physics.rotation.yaw, car.physics.rotation.roll
                    
                    self.cars[i] = type('obj', (object,), {
                        'team_num': BLUE_TEAM if car.team == 0 else ORANGE_TEAM,
                        'position': np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z]),
                        'linear_velocity': np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z]),
                        'angular_velocity': np.array([car.physics.angular_velocity.x, car.physics.angular_velocity.y, car.physics.angular_velocity.z]),
                        'quaternion': np.array([
                            cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2),
                            sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2),
                            cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2),
                            cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2)
                        ]),
                        'boost_amount': car.boost / 100.0,
                        'on_ground': car.has_wheel_contact,
                        'ball_touches': 0,
                        'is_demoed': car.is_demolished,
                        'has_jump': not car.jumped,
                        'has_flip': not car.double_jumped,
                    })()
                    
        return SimpleState()
    
    def _rlgym_action_to_controls(self, action, state):
        """Convert RLGym action to RLBot controls"""
        # Parse action using RLGym's action parser
        actions_dict = self.action_parser.parse_actions(np.array([action]), state)
        rlgym_controls = actions_dict[self.index]
        
        # Convert to RLBot format
        controls = SimpleControllerState()
        controls.throttle = rlgym_controls.throttle
        controls.steer = rlgym_controls.steer
        controls.pitch = rlgym_controls.pitch
        controls.yaw = rlgym_controls.yaw
        controls.roll = rlgym_controls.roll
        controls.jump = rlgym_controls.jump
        controls.boost = rlgym_controls.boost
        controls.handbrake = rlgym_controls.handbrake
        
        return controls
