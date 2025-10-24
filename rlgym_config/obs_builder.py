"""
Custom observation builder for RLGym
Defines what information the agent receives about the game state
"""

import numpy as np
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState


class CustomObsBuilder(ObsBuilder):
    """
    Custom observation builder that provides the agent with:
    - Player position, velocity, rotation
    - Ball position, velocity
    - Relative positions and velocities
    - Boost amount
    - Other useful state information
    """
    
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        """
        Called each time the environment is reset
        """
        pass
    
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build observation vector for the agent
        
        Returns a numpy array containing normalized game state information
        """
        obs = []
        
        # Invert if player is on orange team (makes learning symmetric)
        inverted = player.team_num == 1
        
        if inverted:
            # Flip coordinates for orange team
            player_pos = np.array([-player.car_data.position[0], 
                                   -player.car_data.position[1], 
                                   player.car_data.position[2]])
            player_vel = np.array([-player.car_data.linear_velocity[0],
                                   -player.car_data.linear_velocity[1],
                                   player.car_data.linear_velocity[2]])
            ball_pos = np.array([-state.ball.position[0],
                                -state.ball.position[1],
                                state.ball.position[2]])
            ball_vel = np.array([-state.ball.linear_velocity[0],
                                -state.ball.linear_velocity[1],
                                state.ball.linear_velocity[2]])
        else:
            player_pos = np.array(player.car_data.position)
            player_vel = np.array(player.car_data.linear_velocity)
            ball_pos = np.array(state.ball.position)
            ball_vel = np.array(state.ball.linear_velocity)
        
        # Normalize positions (field is ~10000 units wide/long, ~2000 high)
        player_pos_norm = player_pos / np.array([4096, 5120, 2044])
        ball_pos_norm = ball_pos / np.array([4096, 5120, 2044])
        
        # Normalize velocities (max speed ~2300)
        player_vel_norm = player_vel / 2300
        ball_vel_norm = ball_vel / 6000  # Ball can move faster
        
        # Player to ball relative position and velocity
        relative_pos = (ball_pos - player_pos) / np.array([4096, 5120, 2044])
        relative_vel = (ball_vel - player_vel) / 2300
        
        # Add to observation
        obs.extend(player_pos_norm)      # 3 values
        obs.extend(player_vel_norm)      # 3 values
        obs.extend(ball_pos_norm)        # 3 values
        obs.extend(ball_vel_norm)        # 3 values
        obs.extend(relative_pos)         # 3 values
        obs.extend(relative_vel)         # 3 values
        
        # Player rotation (forward, up vectors)
        forward = np.array(player.car_data.forward())
        up = np.array(player.car_data.up())
        if inverted:
            forward = np.array([-forward[0], -forward[1], forward[2]])
            up = np.array([-up[0], -up[1], up[2]])
        obs.extend(forward)              # 3 values
        obs.extend(up)                   # 3 values
        
        # Angular velocity
        ang_vel = np.array(player.car_data.angular_velocity) / 5.5
        if inverted:
            ang_vel = np.array([-ang_vel[0], -ang_vel[1], ang_vel[2]])
        obs.extend(ang_vel)              # 3 values
        
        # Boost amount (normalized)
        obs.append(player.boost_amount)  # 1 value
        
        # On ground boolean
        obs.append(float(player.on_ground))  # 1 value
        
        # Has flip
        obs.append(float(player.has_flip))   # 1 value
        
        # Is demoed
        obs.append(float(player.is_demoed))  # 1 value
        
        # Total: 33 values
        
        return np.array(obs, dtype=np.float32)
