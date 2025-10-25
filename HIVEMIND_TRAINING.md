# 🔥 Flamewall Hivemind Training Guide

## ✅ TRAINING SETUP COMPLETE!

Your training system is now configured to teach Flamewall bots **team coordination**!

---

## 🎯 **What Changed:**

### New Hivemind Rewards in `training/reward.py`:

1. **`spacing_w=0.75`** - Rewards being 800-2500 units apart from teammates
2. **`coverage_w=0.5`** - Rewards covering different field zones  
3. **`no_stack_w=1.0`** - **PUNISHES** all 3 bots being within 1000 units of ball

### Reward Behaviors:

✅ **Good Spacing (800-2500 units)**: +0.375 reward per teammate pair  
❌ **Too Close (<500 units)**: -0.3 penalty (ball chasing!)  
❌ **All 3 Stacked on Ball**: -1.0 BIG PUNISHMENT  

---

## 🚀 **How to Train:**

### Simple Local Training (Slow but works):
```bash
python training/learner.py
```

### Fast Multi-Worker Training (Recommended):
```bash
# Terminal 1: Learner
python training/learner.py

# Terminal 2-4: Workers (run 3-4 of these)
python training/worker.py --id 0
python training/worker.py --id 1  
python training/worker.py --id 2
```

---

## ⚙️ **Training Configuration:**

Current settings in `learner.py`:
- **Batch Size**: 100,000 samples
- **Mini-batch**: 10,000  
- **Epochs**: 30 per iteration
- **Learning Rate**: 1e-4 (both actor and critic)
- **Gamma**: 0.995 (discount factor)

---

## 📊 **What to Expect:**

### First 100K Steps (2-4 hours):
- Bots learn basic **don't stack on ball**
- Start spreading out more
- Still chaotic but improving

### 500K Steps (12-24 hours):
- Clear spacing behavior
- Better rotation
- Noticeable coordination

### 1M+ Steps (2-3 days):
- Strong hivemind behavior
- Natural role assignment
- Competitive team play

---

## 🎮 **Monitor Training:**

The training uses **WandB** (Weights & Biases) for logging.

Check these metrics:
- **Spacing Reward**: Should increase over time
- **Stack Penalty**: Should decrease (fewer punishments)
- **Win Rate**: Should improve vs baseline

---

## 💾 **Checkpoints:**

Models save to: `models/flamewall/` (check learner.py for exact path)

Every 10 iterations, you get a new checkpoint. Test them by:
1. Copy checkpoint to `src_flamewall/flamewall-model.pt`
2. Run match in RLBot GUI
3. Compare vs old model

---

## 🔧 **Tuning Rewards:**

Edit `training/reward.py` to adjust:

**Make spacing MORE important:**
```python
spacing_w=1.5,        # Was 0.75
no_stack_w=2.0,       # Was 1.0
```

**Make spacing LESS important:**
```python
spacing_w=0.3,        # Was 0.75
no_stack_w=0.5,       # Was 1.0
```

---

## ⚠️ **Important Notes:**

1. **Start from current model**: The training will load `flamewall-model.pt` and improve it
2. **Don't overtrain**: Test checkpoints every 100K steps
3. **Compare vs Nexto**: Keep testing against base Nexto to see improvement
4. **Patience**: Coordination takes time to learn (not as fast as mechanical skills)

---

## 🎯 **Expected Results:**

After 500K steps of hivemind training, Flamewalls should:
- ✅ Never have all 3 bots chasing
- ✅ Maintain 1000+ unit spacing most of the time
- ✅ Show clear attacker/support roles
- ✅ **Beat regular Nexto bots** through better coordination

---

## 🔥 **READY TO TRAIN!**

Your reward function is set up to teach **real hivemind behavior**!

**When you're ready:**
```bash
cd C:\Users\gavin\AppData\Local\RLBotGUIX\MyBots\Flamewall
python training/learner.py
```

**Good luck making the smartest coordinated bot in Rocket League!** 🔥🧠
