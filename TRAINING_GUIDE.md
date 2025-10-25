# 🔥 Flamewall Training & Customization Guide

## Current Status

Your Flamewall bot is using a **pre-trained model** (`flamewall-model.pt`) trained with Deep Reinforcement Learning.

You have 3 options:

---

## Option 1: Use As-Is (Recommended for Tournaments)

**Pros:**
- ✅ Already trained and functional
- ✅ Proven attention-based architecture
- ✅ No training time needed
- ✅ Hivemind coordination adds strategic edge

**Just run it and compete!**

---

## Option 2: Train Your Own Model

### Setup Training Environment

Your existing training files use RLGym 2.0!

**Files to use:**
```
training/
├── learner.py    ← Main training loop
├── worker.py     ← RLGym environment worker
├── agent.py      ← Model architecture
├── obs.py        ← Observation builder
├── reward.py     ← Reward function
└── terminal.py   ← Episode termination conditions
```

### Key Training Parameters:

```python
# Recommended training config
n_steps = 4096              # Steps per rollout
batch_size = 256            # Mini-batch size  
gamma = 0.99                # Discount factor
learning_rate = 5e-5        # PPO learning rate
ent_coef = 0.01             # Entropy coefficient
clip_range = 0.2            # PPO clip range
```

### To Continue Training:

1. **Start training:**
```bash
# Load current model and continue training
python training/learner.py --load src_flamewall/flamewall-model.pt
```

2. **Add hivemind-specific rewards:**
```python
# In training/reward.py
class HivemindReward:
    def get_reward(self, player, state, previous_action):
        reward = 0
        
        # Existing rewards (touch ball, score, etc.)
        reward += standard_rewards(player, state)
        
        # NEW: Reward spacing (don't stack)
        teammates = [p for p in state.players if p.team == player.team]
        if len(teammates) > 1:
            distances = [np.linalg.norm(player.pos - t.pos) 
                        for t in teammates if t != player]
            # Reward being spread out (500-2000 units apart)
            avg_dist = np.mean(distances)
            if 500 < avg_dist < 2000:
                reward += 0.1  # Good spacing
            elif avg_dist < 300:
                reward -= 0.2  # Too clustered (ball chasing)
        
        # NEW: Reward defensive positioning
        if player.role == 'support':
            # Check if positioned between ball and goal
            ball_to_goal = goal_pos - state.ball.position
            player_to_goal = goal_pos - player.position
            # Reward being in defensive position
            alignment = np.dot(ball_to_goal, player_to_goal)
            if alignment > 0:
                reward += 0.05
        
        return reward
```

3. **Train with multiple workers:**
```bash
# Run 4 training instances (faster)
python training/worker.py --id 0 &
python training/worker.py --id 1 &
python training/worker.py --id 2 &
python training/worker.py --id 3 &
python training/learner.py
```

### Expected Training Time:
- **50K steps**: 1-2 hours (basic behaviors)
- **500K steps**: 8-12 hours (Competitive level)
- **5M steps**: 2-3 days (Advanced play)
- **50M+ steps**: 2-3 weeks (Elite level)

---

## Option 3: Enhance Hivemind Logic (No Training)

You can improve coordination **without retraining** by modifying `src_flamewall/bot.py`:

### Example Enhancements:

#### 1. Boost Management
```python
# In get_output() after line 150
if player.boost_amount < 20:
    # Low boost - retreat to nearest pad
    my_role = 'boost_getter'
    # Another bot takes over as chaser
```

#### 2. Passing Detection
```python
# Check if teammate is in better scoring position
if Flamewall._shared_state.get('ball_chaser') == self.index:
    for teammate_idx, teammate_pos in Flamewall._shared_state['positions'].items():
        if teammate_idx != self.index:
            # Check if teammate is closer to goal
            teammate_to_goal = np.linalg.norm(teammate_pos[:2] - enemy_goal[:2])
            my_to_goal = np.linalg.norm(my_pos[:2] - enemy_goal[:2])
            
            if teammate_to_goal < my_to_goal - 1000:
                # Teammate is in better position - pass!
                # Modify action to pass instead of shoot
                pass
```

#### 3. Defensive Rotation
```python
# Support bots maintain defensive shape
if my_role == 'support':
    # Position based on ball trajectory
    ball_vel = self.game_state.ball.linear_velocity
    predicted_ball_pos = ball_pos + ball_vel * 2.0  # 2 seconds ahead
    
    # Get position between predicted ball and own goal
    defense_pos = (predicted_ball_pos + own_goal) / 2
    # Bias action toward defense_pos
```

---

## Recommended Path for Tournaments:

### Immediate (Next Match):
1. ✅ **Use current setup** (pre-trained model + hivemind)
2. ✅ Test coordination behavior
3. ✅ Tune thresholds (500 unit spacing, 30 boost minimum)

### Short-term (1-2 days):
1. Add boost management logic
2. Tune role-switching threshold
3. Add defensive positioning
4. Test against various opponents

### Long-term (1-2 weeks):
1. Fine-tune model with hivemind rewards
2. Train for 500K-1M more steps
3. Add advanced passing/rotation
4. Test in scrimmages

---

## Quick Config Tweaks

### Make More/Less Aggressive:

**In `src_flamewall/bot.py` line ~155:**
```python
# More aggressive (take ball sooner)
if ball_distance < chaser_distance - 300:  # Was 500
    Flamewall._shared_state['ball_chaser'] = self.index

# Less aggressive (wait for bigger opening)
if ball_distance < chaser_distance - 800:  # Was 500
    Flamewall._shared_state['ball_chaser'] = self.index
```

### Require More Boost to Attack:
```python
# Line ~150
if ball_distance < chaser_distance - 500 and player.boost_amount > 50:  # Was 30
```

### Add Third Role (Midfielder):
```python
# After line 165
teammates = [p for p in self.game_state.players if p.team_num == self.team]
if len(teammates) >= 3:
    # Sort by ball distance
    distances = [(i, np.linalg.norm(pos[:2] - ball_pos[:2])) 
                 for i, pos in Flamewall._shared_state['positions'].items()]
    sorted_bots = sorted(distances, key=lambda x: x[1])
    
    if self.index == sorted_bots[0][0]:
        my_role = 'attacker'
    elif self.index == sorted_bots[1][0]:
        my_role = 'midfielder'  # New role!
    else:
        my_role = 'defender'
```

---

## Need Help?

- **RLGym Docs**: https://rlgym.org
- **RLBot Discord**: https://discord.gg/5cNbXgG

**Your bot is tournament-ready with trained skill + hivemind coordination!** 🔥
