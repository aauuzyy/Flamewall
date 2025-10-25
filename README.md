# Flamewall 🔥

## Two Bots, One Package!

This repository contains **TWO specialized Flamewall bots:**

1. **Flamewall** - 3v3 team play specialist with hivemind coordination
2. **フレイムウォール** (Fureimu Wōru) - 1v1 duel specialist with aggressive tuning

See `HOW_TO_USE_BOTH_BOTS.md` for setup instructions!

## What is this?

**Flamewall (3v3 Version)** is a hivemind AI bot with coordinated team play capabilities for Rocket League.
Trained using Deep Reinforcement Learning with RLGym, it features intelligent team coordination where multiple bots work together as a unified strategy.

**フレイムウォール (1v1 Version)** is an aggressive duel specialist optimized for 1v1 matches with faster reactions and pure offensive play.

**Key Features (3v3 Bot):**
- 🧠 **Hivemind Coordination**: Bots share state and dynamically assign roles
- ⚡ **Efficient Performance**: Optimized observation system for smooth gameplay
- 🎯 **Dynamic Role Assignment**: Automatic attacker/support role switching
- 🤝 **Team Strategy**: Bots avoid ball chasing and maintain strategic positioning
- 🏆 **Competitive Level**: Trained with PPO using attention mechanisms

**Key Features (1v1 Bot):**
- ⚔️ **Pure Aggression**: Beta=1.5 for aggressive decision making
- ⚡ **Lightning Fast**: tick_skip=4 for ultra-fast reactions (30 actions/sec!)
- 🎯 **No Hivemind Overhead**: Optimized for solo play
- � **Demo Detection System**: Detects opponents camping in goal (Nexto's weakness!)
- 🐌 **Slow Dribble Tactic**: Beta=0.8 for controlled approach to goal
- 💣 **ULTRA DEMO MODE**: Beta=2.0 when going for the demolition
- 🏆 **Duel Optimized**: Exploits defensive bots with tactical demos

See `1v1_DEMO_STRATEGY.md` for the full tactical breakdown! 💥

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

## Watch Flamewall Learn! 🎥

Want to see the training process live? Check out these channels:

- **[Twitch: aauuzy](https://www.twitch.tv/aauuzy)** - Watch Flamewall training live!
- **[YouTube: aauuzy](https://www.youtube.com/@aauuzy)** - Videos and highlights

Follow along as Flamewall improves through reinforcement learning!

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



