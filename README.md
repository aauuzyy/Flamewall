# Flamewall 🔥

## What is this?

Flamewall is a **hivemind AI bot** with coordinated team play capabilities for Rocket League.
Trained using Deep Reinforcement Learning with RLGym, Flamewall adds intelligent team coordination where multiple bots work together as a unified strategy.

**Key Features:**
- 🧠 **Hivemind Coordination**: Bots share state and dynamically assign roles
- ⚡ **Efficient Performance**: Optimized observation system for smooth gameplay
- 🎯 **Dynamic Role Assignment**: Automatic attacker/support role switching
- 🤝 **Team Strategy**: Bots avoid ball chasing and maintain strategic positioning
- 🏆 **Competitive Level**: Trained with PPO using attention mechanisms

## How does it work?

Flamewall uses **Deep Reinforcement Learning**, specifically Proximal Policy Optimization (PPO).
The bot learns by playing thousands of games at super speeds, with rewards for good plays like:
- Scoring goals
- Maintaining spacing (no ball chasing)
- Strategic positioning
- Team coordination

An **attention mechanism** (commonly used in AI for understanding context) allows the bot to handle any number of players on the field and coordinate with teammates.

## Can I play against it? 

Yes! Use the RLBot GUI to launch matches against Flamewall.
**Important**: Make sure FPS is set to 120 and VSync is turned off for best performance.

## Training

Flamewall is trained using a distributed learning system with RLGym 2.0.
Check `TRAINING_GUIDE.md` for information on customization and continued training.

## Technical Details

Flamewall uses an **attention-based neural network** architecture that processes game state efficiently:
- Dynamic observation system that scales with player count
- 90 discrete actions for precise control
- PyTorch JIT model for optimized inference
- Tick skip optimization for real-time performance

## Advanced Features

**Hivemind Coordination System:**
- Shared state tracking across all bot instances
- Dynamic role assignment based on field position and boost levels
- Anti-ball-chasing rewards during training
- Strategic spacing and coverage optimization

## Development & Training

Flamewall is built using RLGym 2.0 for efficient training:
- Distributed learning system for rapid iteration
- Custom reward functions emphasizing team play
- Proximal Policy Optimization (PPO) algorithm
- Attention mechanisms for context-aware decision making

Check `training/` directory for reward functions and training configuration.

## Requirements

- Python 3.7+
- PyTorch
- RLBot framework
- 120 FPS with VSync disabled (recommended)

## Contributing

This is an active development project. Feel free to experiment with:
- Custom reward functions in `training/reward.py`
- Observation builder modifications in `src_flamewall/flamewall_obs.py`
- Hyperparameter tuning in `src_flamewall/bot.py`



