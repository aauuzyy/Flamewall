# Flamewall Bot - SIMPLE GUIDE

## ✅ What Works RIGHT NOW

You have a trained 100K step model that works!

###  To RUN the bot in matches:

1. Open RLBot GUI
2. Add Flamewall bot
3. Start match
4. **Bot will auto-load the 100K model and play!**

The bot is currently at Bronze 2 level (basic ball chasing, will miss a lot).

### 📊 To TRAIN more:

```bash
py -3.10 train_flamewall_team.py
```

This trains a 3v3 bot. Models save every 100K steps to `./models/flamewall_team/`

**Training progress:**
- 100K steps = Bronze (you are here!)
- 500K steps = Silver  
- 1M steps = Gold
- 2M+ steps = Platinum+

Just let it train overnight!

### 🎮 Bot Files

- **`src/bot.py`** - The bot that runs in matches (auto-loads latest model)
- **`train_flamewall_team.py`** - Training script for 3v3
- **`models/flamewall_team/`** - Your trained models

### 🚨 Important

- The bot needs packages installed for Python 3.11 (RLBot's Python):
  ```bash
  C:\Users\gavin\AppData\Local\RLBotGUIX\Python311\python.exe -m pip install rlgym-api rlgym-rocket-league stable-baselines3
  ```
  *(Already done!)*

- Training uses Python 3.10 (your Python)
- These are DIFFERENT Pythons - that's normal!

## That's It!

You're all set. The bot works, training works. Just be patient - 100K steps is very early in training!
