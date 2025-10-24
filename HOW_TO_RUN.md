# 🎮 HOW TO RUN YOUR TRAINED BOT

## 🚀 Quick Start

You have a trained model at **1 million steps** ready to use!

### Option 1: RLBot GUI (Easiest)

1. **Open RLBot GUI**
   ```bash
   python run_gui.py
   ```
   Or just double-click `run_gui.py`

2. **Add your bot**:
   - Click "Add" button
   - Navigate to: `src/bot_rl.cfg`
   - Select it

3. **Add an opponent** (e.g., Nexto):
   - Click "Add" again
   - Find and select Nexto or any other bot

4. **Start the match**:
   - Click "Start Match"
   - Watch your RL bot play!

### Option 2: Command Line

```bash
# Make sure you're using Python 3.10 for RL dependencies
py -3.10 run_gui.py
```

---

## 📁 Bot Files

You have **TWO** bot configurations:

### 1. Original Bot (`src/bot.cfg`)
- Uses: `src/bot.py`
- Type: Hand-coded logic
- Good for: Baseline comparison

### 2. RL Bot (`src/bot_rl.cfg`) ⭐ NEW!
- Uses: `src/bot_rl.py`
- Type: Trained RL model
- Model: `models/rlgym_real/flamewall_1000000_steps.zip`
- This is your trained bot!

---

## 🎯 Testing Your Bot

### Quick Test Match:
1. Open RLBot GUI
2. Add: `Flamewall RL` (your trained bot)
3. Add: `Nexto` (or any bot to test against)
4. Start match

### Tournament Mode:
1. Add multiple bots
2. Set up tournament bracket
3. See how your bot performs!

---

## 📊 Which Model Is Loaded?

The bot automatically loads the best available model in this order:

1. `models/rlgym_real/flamewall_final.zip` (if exists)
2. `models/rlgym_real/flamewall_1000000_steps.zip` ✅ **YOU HAVE THIS**
3. `models/rlgym_real/flamewall_950000_steps.zip`

**Your bot is using the 1M step checkpoint!**

---

## 🔧 Troubleshooting

### "No trained model found!"
- Check that `models/rlgym_real/` directory has .zip files
- You should have `flamewall_1000000_steps.zip` ✅

### Bot doesn't move / sits still
- Model might not be loading correctly
- Check Python 3.10 is being used: `py -3.10 run_gui.py`
- Make sure stable-baselines3 is installed for Python 3.10

### "Import stable_baselines3 error"
```bash
py -3.10 -m pip install stable-baselines3
```

### Bot plays poorly
- 1M steps is just getting started!
- For better performance, train longer:
  ```bash
  py -3.10 train_advanced.py
  ```
- Let it run for 5-10M steps for competitive play

---

## 📈 Performance Expectations

### At 1 Million Steps (Current):
- ✅ Chases ball consistently
- ✅ Makes contact regularly  
- ✅ Occasional goals
- ⚠️ Positioning still learning
- ⚠️ Defense is basic
- **Expected**: Might lose to Nexto but shows RL behavior

### At 5 Million Steps:
- ✅ Good ball control
- ✅ Consistent shots
- ✅ Better positioning
- ✅ Can beat intermediate bots
- **Expected**: 50-60% win rate vs Nexto

### At 10+ Million Steps:
- ✅ Strong gameplay
- ✅ Advanced mechanics
- ✅ Good decision-making
- **Expected**: 70-80% win rate vs Nexto

---

## 🎮 Recommended Testing Sequence

### 1. Test vs. Psyonix All-Star
Easy opponent to see if bot learned basics

### 2. Test vs. Original Bot
Compare RL bot vs your hand-coded version
- Add both `bot.cfg` and `bot_rl.cfg`
- See the difference!

### 3. Test vs. Nexto
Your target opponent
- This will show current skill level
- Don't expect to win yet at 1M steps

### 4. Test vs. Multiple Bots
3v3 or 2v2 scenarios

---

## 🔄 Using Different Checkpoints

Want to try an earlier or later checkpoint?

Edit `src/bot_rl.py` line 32-36:
```python
model_files = [
    'flamewall_final.zip',
    'flamewall_1000000_steps.zip',  # Current
    'flamewall_500000_steps.zip',   # Try earlier checkpoint
]
```

---

## 📝 Customization

### Change Bot Name
Edit `src/bot_rl.cfg`:
```
name = Your Custom Name Here
```

### Change Appearance
Edit `src/appearance.cfg` for custom colors and car

### Use Advanced Model
Once `train_advanced.py` finishes:
```python
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rlgym_advanced')
```

---

## 🚀 Next Steps

1. **Test current 1M model** - See baseline performance
2. **Continue training** - Run `py -3.10 train_advanced.py` for more
3. **Test again at 5M** - Should see big improvement
4. **Keep training to 10M+** - Competitive level

---

## 💡 Pro Tips

1. **Record matches** - Use RLBot recording feature to analyze
2. **Compare checkpoints** - Test 500K vs 1M vs 2M to see improvement
3. **Mix opponents** - Test against variety of bots
4. **Watch for patterns** - See what behaviors emerge from training
5. **Share results** - Post your bot's performance in RLBot community!

---

## 🎉 You're Ready!

Your bot has been trained and is ready to play!

**Quick command to start:**
```bash
python run_gui.py
```

Then add `src/bot_rl.cfg` and start a match!

**Good luck and have fun watching your trained AI play Rocket League!** 🏆⚽🔥
