# 🔥 FLAMEWALL HIVEMIND BOT - Setup Complete

## What Is This?

**Flamewall** is an advanced Rocket League AI bot with hivemind coordination capabilities!

### New Structure:
```
src_flamewall/          ← Main bot directory
├── bot.py              ← Core bot logic with hivemind
├── agent.py            ← Model loading and action selection
├── flamewall_obs.py    ← Efficient observation builder
├── flamewall-model.pt  ← Pre-trained neural network model
├── bot.cfg             ← Bot configuration
└── requirements.txt    ← Python dependencies
```

## Key Features:

### 1. **Hivemind Coordination**
```python
# Shared state across all bot instances
_shared_state = {
    'ball_chaser': None,     # Which bot attacks
    'positions': {},         # Bot locations
    'boost_status': {},      # Boost amounts
    'assignments': {}        # Roles (attacker/support)
}
```

**Coordination Logic:**
- Bots share position and boost status every tick
- Closest bot with sufficient boost becomes ball chaser
- Others automatically take support positions
- Prevents ball chasing - only 1 bot attacks at a time
- Dynamic role switching based on distance (500 unit threshold)

### 2. **Configuration**
- `rlbot.cfg`: Points to `src_flamewall/bot.cfg` 
- `bot.cfg`: Name = "Flamewall", custom appearance
- Bot runs at 120 tick rate for optimal performance

## How It Works:

### Hivemind Behavior:
1. **Every tick**, each bot updates shared state with its position/boost
2. **Distance calculation** to ball for all bots
3. **Role assignment**:
   - Closest bot with 30+ boost → **Attacker** (ball chaser)
   - Others → **Support** (defensive/midfield positioning)
4. **Takeover threshold**: Must be 500+ units closer to steal attacker role

### Why This Works:
✅ **Trained with PPO**: Proximal Policy Optimization reinforcement learning
✅ **No performance issues**: Efficient observation building  
✅ **Team coordination**: Bots don't all chase ball  
✅ **Dynamic strategy**: Roles switch based on game state  
✅ **Easy to extend**: Can add more coordination logic  

## Next Steps:

### Test It:
```bash
# Run from RLBot GUI with 3 Flamewall bots
# Watch for "🔥 FLAMEWALL Ready - Hivemind ENABLED" messages
```

### Train It More:
Check `TRAINING_GUIDE.md` for details on:
- Fine-tuning the existing model
- Training from scratch
- Adding custom rewards
- Enhancing hivemind behavior

### Enhance Hivemind:
Ideas to add (in `get_output()` method):
- **Boost management**: Low-boost bot retreats to grab pads
- **Passing plays**: Attacker checks if teammate is in scoring position  
- **Defensive rotation**: Support bots position based on ball trajectory
- **Shot blocking**: Defender tracks opponent shots toward goal
- **Demolition avoidance**: Bots warn teammates of incoming demos

## Model Information:

**Current Model**: `flamewall-model.pt`
- **Architecture**: Attention-based neural network (PyTorch)
- **Training**: PPO (Proximal Policy Optimization)
- **Action Space**: 90 discrete actions (lookup table)
- **Observation**: Efficient state representation

## Credits:
- **Developer**: Gavin (aauuzyy)
- **Framework**: RLBot, RLGym, PyTorch
- **Architecture Inspiration**: Attention mechanisms from transformer models

---

**Ready to dominate with coordinated team play!** 🔥
