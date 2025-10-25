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
            # RLGym 2.0 API: build_obs(agents, state, shared_info) returns {player_id: obs_array}
            agents = list(rlgym_state.cars.keys())  # Get list of agent IDs
            shared_info = {}  # Empty dict for shared info
            obs_dict = self.obs_builder.build_obs(agents, rlgym_state, shared_info)
            obs = obs_dict[self.index]
            
            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Convert action to controls using RLGym's action parser
            controls = self._rlgym_action_to_controls(action, rlgym_state)
            
        except Exception as e:
            import traceback
            # Only print error once (not every frame)
            if not hasattr(self, '_error_logged'):
                print(f"Error getting model output: {e}")
                print("Full traceback:")
                traceback.print_exc()
                self._error_logged = True
            # Return default controls on error
            pass
        
        return controls
    
    def _packet_to_rlgym_state(self, packet: GameTickPacket):
        """Convert RLBot packet to RLGym state format"""
        from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
        
        class FlexibleObject:
            """Object that returns default values for missing attributes"""
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            
            def __getattr__(self, name):
                # Return sensible defaults for missing attributes
                defaults = {
                    'forward': np.array([1, 0, 0]),
                    'up': np.array([0, 0, 1]),
                    'right': np.array([0, 1, 0]),
                    'position': np.zeros(3),
                    'linear_velocity': np.zeros(3),
                    'angular_velocity': np.zeros(3),
                    'quaternion': np.array([1, 0, 0, 0]),
                }
                for key, val in defaults.items():
                    if key in name.lower():
                        return val
                # Physics object should be a FlexibleObject too
                if name == 'physics':
                    return FlexibleObject()
                # Boolean attributes default to False (but return int-convertible)
                if name.startswith('is_') or name.startswith('has_') or name.startswith('can_') or name == 'on_ground':
                    return 0  # Return 0 instead of False for int() compatibility
                # Numeric attributes default to 0
                if 'time' in name or 'amount' in name or 'handbrake' in name or 'team_num' in name or 'ball_touches' in name:
                    return 0.0
                return 0
        
        class SimpleState:
            def __init__(self):
                self.cars = {}
                ball_pos = np.array([
                    packet.game_ball.physics.location.x,
                    packet.game_ball.physics.location.y,
                    packet.game_ball.physics.location.z
                ])
                ball_vel = np.array([
                    packet.game_ball.physics.velocity.x,
                    packet.game_ball.physics.velocity.y,
                    packet.game_ball.physics.velocity.z
                ])
                ball_ang_vel = np.array([
                    packet.game_ball.physics.angular_velocity.x,
                    packet.game_ball.physics.angular_velocity.y,
                    packet.game_ball.physics.angular_velocity.z
                ])
                
                self.ball = FlexibleObject(
                    position=ball_pos,
                    linear_velocity=ball_vel,
                    angular_velocity=ball_ang_vel
                )
                
                # Inverted ball (for orange team perspective)
                self.inverted_ball = FlexibleObject(
                    position=ball_pos * np.array([-1, -1, 1]),
                    linear_velocity=ball_vel * np.array([-1, -1, 1]),
                    angular_velocity=ball_ang_vel * np.array([-1, -1, 1])
                )
                
                # Boost pad timers (same for both teams)
                self.boost_pad_timers = np.zeros(34)
                self.inverted_boost_pad_timers = np.zeros(34)
                
                # Inverted cars dictionary (will be populated in the loop)
                self.inverted_cars = {}
                
                for i in range(packet.num_cars):
                    car = packet.game_cars[i]
                    # Convert rotation to quaternion
                    from math import cos, sin
                    pitch, yaw, roll = car.physics.rotation.pitch, car.physics.rotation.yaw, car.physics.rotation.roll
                    
                    quat = np.array([
                        cos(roll/2)*cos(pitch/2)*cos(yaw/2) + sin(roll/2)*sin(pitch/2)*sin(yaw/2),
                        sin(roll/2)*cos(pitch/2)*cos(yaw/2) - cos(roll/2)*sin(pitch/2)*sin(yaw/2),
                        cos(roll/2)*sin(pitch/2)*cos(yaw/2) + sin(roll/2)*cos(pitch/2)*sin(yaw/2),
                        cos(roll/2)*cos(pitch/2)*sin(yaw/2) - sin(roll/2)*sin(pitch/2)*cos(yaw/2)
                    ])
                    
                    # Use FlexibleObject - it will handle ANY missing attribute!
                    # Convert all boolean-like values to plain int (0 or 1) for compatibility
                    self.cars[i] = FlexibleObject(
                        team_num=BLUE_TEAM if car.team == 0 else ORANGE_TEAM,
                        position=np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z]),
                        linear_velocity=np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z]),
                        angular_velocity=np.array([car.physics.angular_velocity.x, car.physics.angular_velocity.y, car.physics.angular_velocity.z]),
                        quaternion=quat,
                        physics=FlexibleObject(
                            position=np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z]),
                            linear_velocity=np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z]),
                            angular_velocity=np.array([car.physics.angular_velocity.x, car.physics.angular_velocity.y, car.physics.angular_velocity.z]),
                            quaternion=quat,
                            forward=np.array([cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]),
                            up=np.array([-cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll),
                                       -sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll),
                                       cos(pitch)*sin(roll)]),
                            right=np.array([sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll),
                                          -cos(yaw)*sin(pitch)*cos(roll) - sin(yaw)*sin(roll),
                                          cos(pitch)*cos(roll)]),
                        ),
                        boost_amount=float(car.boost / 100.0),
                        on_ground=int(bool(car.has_wheel_contact)),
                        ball_touches=0,
                        is_demoed=int(bool(car.is_demolished)),
                        is_supersonic=int(bool(car.is_super_sonic)),
                        has_jump=int(not bool(car.jumped)),
                        has_jumped=int(bool(car.jumped)),
                        has_flip=int(not bool(car.double_jumped)),
                        has_flipped=int(bool(car.double_jumped)),
                        has_double_jumped=int(bool(car.double_jumped)),
                        can_flip=int(not bool(car.double_jumped)),
                        is_jumping=int(bool(car.jumped)),
                        is_flipping=int(bool(car.double_jumped)),
                    )
                    
                    # Create inverted car (for orange team perspective)
                    inv_pos = np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z]) * np.array([-1, -1, 1])
                    inv_vel = np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z]) * np.array([-1, -1, 1])
                    inv_ang_vel = np.array([car.physics.angular_velocity.x, car.physics.angular_velocity.y, car.physics.angular_velocity.z]) * np.array([-1, -1, 1])
                    
                    # Inverted rotation for orange team
                    inv_pitch, inv_yaw, inv_roll = -pitch, yaw + np.pi, -roll
                    inv_quat = np.array([
                        cos(inv_roll/2)*cos(inv_pitch/2)*cos(inv_yaw/2) + sin(inv_roll/2)*sin(inv_pitch/2)*sin(inv_yaw/2),
                        sin(inv_roll/2)*cos(inv_pitch/2)*cos(inv_yaw/2) - cos(inv_roll/2)*sin(inv_pitch/2)*sin(inv_yaw/2),
                        cos(inv_roll/2)*sin(inv_pitch/2)*cos(inv_yaw/2) + sin(inv_roll/2)*cos(inv_pitch/2)*sin(inv_yaw/2),
                        cos(inv_roll/2)*cos(inv_pitch/2)*sin(inv_yaw/2) - sin(inv_roll/2)*sin(inv_pitch/2)*cos(inv_yaw/2)
                    ])
                    
                    self.inverted_cars[i] = FlexibleObject(
                        team_num=BLUE_TEAM if car.team == 0 else ORANGE_TEAM,
                        position=inv_pos,
                        linear_velocity=inv_vel,
                        angular_velocity=inv_ang_vel,
                        quaternion=inv_quat,
                        physics=FlexibleObject(
                            position=inv_pos,
                            linear_velocity=inv_vel,
                            angular_velocity=inv_ang_vel,
                            quaternion=inv_quat,
                            forward=np.array([cos(inv_yaw)*cos(inv_pitch), sin(inv_yaw)*cos(inv_pitch), sin(inv_pitch)]),
                            up=np.array([-cos(inv_yaw)*sin(inv_pitch)*sin(inv_roll) - sin(inv_yaw)*cos(inv_roll),
                                       -sin(inv_yaw)*sin(inv_pitch)*sin(inv_roll) + cos(inv_yaw)*cos(inv_roll),
                                       cos(inv_pitch)*sin(inv_roll)]),
                            right=np.array([sin(inv_yaw)*sin(inv_pitch)*cos(inv_roll) - cos(inv_yaw)*sin(inv_roll),
                                          -cos(inv_yaw)*sin(inv_pitch)*cos(inv_roll) - sin(inv_yaw)*sin(inv_roll),
                                          cos(inv_pitch)*cos(inv_roll)]),
                        ),
                        boost_amount=float(car.boost / 100.0),
                        on_ground=int(bool(car.has_wheel_contact)),
                        ball_touches=0,
                        is_demoed=int(bool(car.is_demolished)),
                        is_supersonic=int(bool(car.is_super_sonic)),
                        has_jump=int(not bool(car.jumped)),
                        has_jumped=int(bool(car.jumped)),
                        has_flip=int(not bool(car.double_jumped)),
                        has_flipped=int(bool(car.double_jumped)),
                        has_double_jumped=int(bool(car.double_jumped)),
                        can_flip=int(not bool(car.double_jumped)),
                        is_jumping=int(bool(car.jumped)),
                        is_flipping=int(bool(car.double_jumped)),
                    )
                    
        return SimpleState()

    
    def _rlgym_action_to_controls(self, action, state):
        """Convert RLGym action to RLBot controls"""
        try:
            # Parse action using RLGym's action parser
            # Create actions dict for all agents (just use 0 for others)
            actions = {i: np.array([0]) for i in state.cars.keys()}
            actions[self.index] = np.array([action])
            
            rlgym_controls_dict = self.action_parser.parse_actions(actions, state, {})
            rlgym_controls = rlgym_controls_dict[self.index]
            
            # Convert to RLBot format
            controls = SimpleControllerState()
            controls.throttle = float(np.array(rlgym_controls.throttle).item())
            controls.steer = float(np.array(rlgym_controls.steer).item())
            controls.pitch = float(np.array(rlgym_controls.pitch).item())
            controls.yaw = float(np.array(rlgym_controls.yaw).item())
            controls.roll = float(np.array(rlgym_controls.roll).item())
            controls.jump = bool(rlgym_controls.jump)
            controls.boost = bool(rlgym_controls.boost)
            controls.handbrake = bool(rlgym_controls.handbrake)
            
            return controls
        except Exception as e:
            # Return default controls on error
            return SimpleControllerState()
