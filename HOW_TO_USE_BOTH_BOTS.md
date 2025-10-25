# Using Both Flamewall Bots

You now have TWO versions of Flamewall:

## 1. **Flamewall** (3v3 Specialist) 🔥
- **Location:** `src_flamewall/bot.cfg`
- **Name:** Flamewall
- **Optimized for:** 3v3 team play
- **Features:** Hivemind coordination, dynamic role assignment
- **Settings:** beta=1.0, tick_skip=8, stochastic kickoffs

## 2. **フレイムウォール** (1v1 Specialist) 🔥
- **Location:** `src_flamewall_1v1/bot.cfg`
- **Name:** フレイムウォール (Fureimu Wōru / Flamewall in Japanese)
- **Optimized for:** 1v1 duels with **Nexto weakness exploit**
- **Features:** Demo detection system, dynamic aggression, slow dribble → BOOM strategy
- **Settings:** beta=1.5 (base), beta=0.8 (dribble), beta=2.0 (DEMO!), tick_skip=4 (ULTRA FAST)
- **Special Tactic:** Detects when opponent camps in goal, slow dribbles to approach, then DEMOLISHES them for free goals

## How to Add Both to RLBot GUI:

### Method 1: Add Both Manually
1. Open RLBot GUI
2. Click "Add" → "Browse for bot config"
3. Navigate to `Flamewall/src_flamewall/bot.cfg` → Add
4. Click "Add" → "Browse for bot config" again
5. Navigate to `Flamewall/src_flamewall_1v1/bot.cfg` → Add

Now both bots will appear in your bot list!

### Method 2: Edit rlbot.cfg
For quick testing, you can edit `rlbot.cfg`:

**For 1v1 match:**
```
num_participants = 2
participant_config_0 = src_flamewall_1v1/bot.cfg
participant_config_1 = src_flamewall_1v1/bot.cfg
participant_team_0 = 0
participant_team_1 = 1
```

**For 3v3 match:**
```
num_participants = 6
participant_config_0 = src_flamewall/bot.cfg
participant_config_1 = src_flamewall/bot.cfg
participant_config_2 = src_flamewall/bot.cfg
participant_config_3 = src_flamewall/bot.cfg
participant_config_4 = src_flamewall/bot.cfg
participant_config_5 = src_flamewall/bot.cfg
```

## Performance Tips:
- Run at 120 FPS with VSync OFF
- 1v1 bot has ULTRA FAST tick rate (4 vs 8) for lightning reactions
- 3v3 bot uses hivemind coordination for team strategy
- Both use the same trained model (flamewall-model.pt)
- **1v1 bot has special demo detection** - exploits Nexto's camping weakness!

## Why Two Bots?
- **1v1** requires pure aggression, demo tactics, and exploiting defensive weaknesses
- **3v3** benefits from team coordination and strategic spacing
- Having both lets you compete in any tournament format!
- **1v1 bot specifically counters Nexto's goalie camping strategy** 💥
