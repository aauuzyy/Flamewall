# Flamewall Bot Comparison 🔥

## Side-by-Side Comparison

| Feature | **Flamewall** (3v3) | **フレイムウォール** (1v1) |
|---------|---------------------|--------------------------|
| **Name** | Flamewall | フレイムウォール (Fureimu Wōru) |
| **Location** | `src_flamewall/` | `src_flamewall_1v1/` |
| **Optimized For** | 3v3 Team Play | 1v1 Duels + Demo Strat |
| **Beta Value** | 1.0 (Balanced) | 1.5 base / 0.8 dribble / 2.0 DEMO |
| **Tick Skip** | 8 (Standard) | 4 (ULTRA FAST - 30 actions/sec) |
| **Kickoffs** | Stochastic (Random) | Consistent (Reliable) |
| **Hivemind** | ✅ Active | ❌ Disabled |
| **Team Coordination** | ✅ Dynamic Roles | ❌ N/A |
| **Playstyle** | Strategic Support | ULTRA AGGRESSIVE + Demo Exploit |
| **Special Feature** | Role assignment | Nexto weakness exploit |

## When to Use Each Bot

### Use **Flamewall** (3v3) for:
- 3v3 Standard matches
- 2v2 matches where coordination helps
- Tournament play with multiple bots on same team
- Testing hivemind coordination features
- Balanced, team-oriented gameplay

### Use **フレイムウォール** (1v1) for:
- 1v1 Duel matches
- **Exploiting Nexto's weakness** (camping in goal)
- Demo-heavy aggressive playstyle
- Tournament 1v1 brackets
- Maximum individual skill + tactical demos
- Destroying defensive opponents

## Technical Differences

### Flamewall (3v3)
```python
beta = 1.0              # Balanced decision making
tick_skip = 8           # Standard reaction time
stochastic_kickoffs = True  # Varied kickoff approaches
hivemind = True         # Team coordination active
```

### フレイムウォール (1v1)
```python
beta = 1.5              # High baseline aggression
# Dynamic beta switching:
# - 0.8 when slow dribbling (careful control)
# - 2.0 when going for demo (ULTRA AGGRESSIVE)
tick_skip = 4           # 50% faster than 3v3 (30 actions/sec!)
stochastic_kickoffs = False # Consistent kickoffs
hivemind = False        # Solo optimization

# Demo detection system:
# - Tracks opponent position relative to goal
# - Enters "slow dribble mode" when opponent camps
# - Switches to "DEMO MODE" when in range
# - Exploits Nexto's static goalie behavior
```

## Model & Training

Both bots use the **same trained model** (`flamewall-model.pt`) but with different:
- Inference parameters (beta, tick_skip)
- Code logic (hivemind on/off)
- Kickoff strategies
- Optimization targets

This means they have the same fundamental skills, but express them differently based on game mode!

## Performance Notes

- Both require 120 FPS with VSync off for optimal play
- 1v1 bot has slightly higher CPU usage (faster tick rate)
- 3v3 bot has minimal overhead from hivemind coordination
- Both use the same observation builder and action space

---

**TL;DR:** Same brain, different strategy! Use Flamewall for teams, フレイムウォール for duels. 🔥
