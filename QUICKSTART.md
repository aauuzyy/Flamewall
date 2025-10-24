# Flamewall RLGym Setup - Quick Start

## 🚀 Quick Start Guide

I've added RLGym (reinforcement learning) capabilities to your Flamewall bot! Here's how to use it:

## Step 1: Install Dependencies

Run this command to install all required packages:

```powershell
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes as it installs PyTorch, RLGym, and Stable-Baselines3.

## Step 2: Start Training

Once installed, start training your bot:

```powershell
python train_rlgym.py
```

The bot will start learning to play Rocket League! Training runs at 100x speed and saves progress automatically.

## Step 3: Monitor Progress (Optional)

Watch training in real-time:

```powershell
tensorboard --logdir=logs
```

Then open: http://localhost:6006

## Step 4: Use Your Trained Bot

After training (or loading a checkpoint), edit `src/bot.cfg`:

Change:
```ini
python_file = bot.py
```

To:
```ini
python_file = rlgym_bot.py
```

Now your bot will use the trained AI!

---

## 📁 What Was Added

### New Files:
- **`train_rlgym.py`** - Main training script
- **`rlgym_config/`** - Training configuration
  - `rewards.py` - Defines what behaviors get rewarded
  - `obs_builder.py` - Defines what the AI can see
  - `terminal_conditions.py` - When episodes end
- **`src/rlgym_bot.py`** - AI-powered bot for RLBot
- **`RLGYM_GUIDE.md`** - Detailed documentation

### Modified Files:
- **`requirements.txt`** - Added RL dependencies

---

## ⚙️ Key Training Parameters

Edit `train_rlgym.py` to customize:

```python
game_speed=100          # Training speed (100x faster)
team_size=1            # 1v1 matches (simplest)
total_timesteps=10_000_000  # Training duration
```

---

## 🎯 What the Bot Learns

The reward function teaches your bot to:
- ✅ Chase and hit the ball
- ✅ Hit ball toward opponent's goal
- ✅ Score goals (+10 reward!)
- ✅ Make saves
- ✅ Face the ball
- ✅ Manage boost efficiently

---

## 💡 Tips

1. **Training takes time**: Expect hours/days for good results
2. **Stop anytime**: Press Ctrl+C to save and stop training
3. **Use checkpoints**: Models save every 100k steps in `models/PPO/`
4. **Test regularly**: Load checkpoints and try them in matches
5. **Adjust rewards**: Edit `rlgym_config/rewards.py` if bot isn't learning what you want

---

## 🐛 Troubleshooting

**Installation fails?**
- Make sure Python 3.7-3.10 is installed (RLGym doesn't support 3.11+ yet)
- Try: `pip install --upgrade pip` first

**Training crashes?**
- Check Rocket League is closed (RLGym will launch it)
- Ensure you have enough disk space for models/logs

**Bot doesn't use trained model?**
- Make sure `models/PPO/flamewall_final_model.zip` exists
- Check `src/bot.cfg` points to `rlgym_bot.py`

---

## 📚 More Info

See **`RLGYM_GUIDE.md`** for detailed documentation!

**Resources:**
- RLGym: https://rlgym.org/
- RLBot Discord: https://discord.gg/rlbot
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

---

## 🎮 Have Fun!

You're now ready to train an AI to play Rocket League! The bot will learn from scratch, making mistakes at first but gradually improving. Watch it learn in TensorBoard and test checkpoints to see progress.

Good luck with your training! 🚀⚽
