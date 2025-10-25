# 🌊 Training Tsunami Surge for 1v3 Domination 💥

## The Challenge

Train Tsunami Surge to compete **1v3 against Flamewall teams**. This is an EXTREME challenge that will require:
- Exceptional boost management
- Aggressive demo plays to even the odds
- Perfect positioning and prediction
- Relentless offensive pressure
- Incredible defensive skills (you're the only defender!)

## Training Strategy

### Phase 1: 1v1 Mastery (Current)
- Train against single Flamewall bot
- Master basic mechanics, dribbling, demos
- Build foundation of aggressive play

### Phase 2: 1v2 Intermediate
- Add second opponent
- Learn to manage multiple threats
- Practice quick rotations between offense/defense
- Focus on demo plays to reduce numbers

### Phase 3: 1v3 ULTIMATE
- Face three Flamewall bots
- Learn to exploit hivemind coordination gaps
- Master overwhelming pressure tactics
- Become the ultimate solo carry

## Training Rewards

The custom reward function (`tsunami_1v3_reward.py`) emphasizes:

### Critical Rewards (High Weight)
- **Goals (10.0)** - Scoring is everything when outnumbered
- **Saves (5.0)** - You're the only defender
- **Demos (4.0)** - Even the odds by removing opponents
- **Shots (3.0)** - Constant offensive pressure
- **Pressure (2.0)** - Stay in opponent half

### Supporting Rewards (Lower Weight)
- **Ball Touch (0.8)** - Control the ball
- **Survival (0.5)** - Don't get demo'd
- **Boost Pickup (0.5)** - Smart boost management
- **Speed (0.3)** - Stay fast and aggressive
- **Kickoff (1.0)** - Win kickoffs for momentum

### Penalties
- **Low Boost (-0.3)** - Vulnerable when outnumbered
- **Getting Demo'd (-2.0)** - Can't afford to be out of play
- **Losing (-5.0)** - Motivation to improve

### Victory Bonus
- **Winning 1v3 (+50.0)** - MASSIVE reward for the ultimate achievement
- **Tying 1v3 (+5.0)** - Respectable result

## Training Configuration

### Recommended Setup:
```python
# In your training script
from training.tsunami_1v3_reward import Tsunami1v3RewardFunction

# Create reward function
reward_fn = Tsunami1v3RewardFunction(
    goal_w=10.0,
    shot_w=3.0,
    save_w=5.0,
    demo_w=4.0,
    # ... other weights
)

# Training opponents
# Phase 1: 1v1
opponents = [FlamewallBot]

# Phase 2: 1v2
opponents = [FlamewallBot, FlamewallBot]

# Phase 3: 1v3
opponents = [FlamewallBot, FlamewallBot, FlamewallBot]
```

## Key Training Tips

### 1. Demos Are Your Friend
- Reducing 1v3 to 1v2 or 1v1 is HUGE
- Train the bot to recognize demo opportunities
- Reward aggressive demo plays heavily

### 2. Boost Starvation
- Deny boost pads to opponents
- Control big boost pads
- Learn efficient boost usage

### 3. Counter the Hivemind
- Flamewall uses hivemind coordination
- Exploit: all bots might commit to ball at once
- Learn to bait and punish poor rotations

### 4. Stay Aggressive
- Playing defensive 1v3 = guaranteed loss
- Constant pressure keeps opponents reactive
- Make THEM defend, not you

### 5. Speed Is Everything
- Fast rotations between offense/defense
- Quick recoveries from challenges
- Maintain supersonic as much as possible

## Expected Training Time

- **Phase 1 (1v1)**: 2-4 hours of training
- **Phase 2 (1v2)**: 5-10 hours of training  
- **Phase 3 (1v3)**: 20-50 hours of training (INTENSE!)

Total: **30-60+ hours** of dedicated training

## Success Metrics

### Phase 1 (1v1)
- Win rate: 60%+
- Average goal differential: +0.5
- Demo rate: 1+ per game

### Phase 2 (1v2)
- Win rate: 40%+
- Average goal differential: -0.5 to 0
- Demo rate: 2+ per game

### Phase 3 (1v3) - THE DREAM
- Win rate: 25%+ (INCREDIBLE if achieved!)
- Average goal differential: -1.0 to 0
- Demo rate: 3+ per game
- **ANY win is legendary**

## Streaming the Training

Perfect content for your channels:
- **Twitch**: https://www.twitch.tv/aauuzy (Live training sessions)
- **YouTube**: https://www.youtube.com/@aauuzy (Training highlights, breakthroughs)

This is EPIC content - one bot learning to defeat THREE coordinated AI opponents!

## Hardware Requirements

1v3 training is INTENSIVE:
- Recommended: RTX 3060+ or equivalent
- 16GB+ RAM
- Fast CPU (3.5GHz+ multi-core)
- Consider using cloud GPUs for faster training

## The Ultimate Goal

If Tsunami Surge can win even **ONE match 1v3 against Flamewall**, it will be:
- One of the most impressive RL bot achievements
- Perfect tournament demonstration
- Incredible stream content
- Proof that smart training > raw numbers

## Let's Make History! 🌊💥

This is going to be LEGENDARY. Document everything - every breakthrough, every strategy discovered, every hard-fought victory. This is the kind of challenge that could put Tsunami Surge on the map!

---

**Remember**: Even if it doesn't win consistently 1v3, getting CLOSE is still incredible. The training process itself will make Tsunami Surge extremely strong in normal 1v1 and even 2v2/3v3 scenarios!
