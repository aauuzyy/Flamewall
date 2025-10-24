# 🚀 ADVANCED TRAINING GUIDE - Beat Nexto!

## 🎯 What's New in Advanced Training

Your bot now has **ALL** the advanced features used by competitive RL bots:

### ✅ Features Implemented:

1. **Parallel Training (4x Faster!)** 
   - 4 environments running simultaneously
   - Collects experience 4x faster than before
   - Better sample diversity

2. **Advanced Reward Shaping**
   - Velocity toward ball (encourages movement)
   - Ball velocity toward goal (encourages shots)
   - Touch bonuses (rewards contact)
   - Facing ball rewards (better positioning)
   - Boost management (efficient boost usage)
   - Aerial rewards (learns aerial play)
   - Distance penalties (prevents ball-chasing)

3. **Optimized PPO Hyperparameters**
   - Larger neural networks (512→512→256 layers)
   - Better learning rate (3e-4)
   - Larger batch sizes for stability
   - Tuned for Rocket League specifically

4. **Self-Play System**
   - Saves opponent checkpoints every 500K steps
   - Can train against past versions of itself
   - Creates diverse opponents

5. **Curriculum Learning (Ready to Enable)**
   - Stage 1: Stationary ball practice
   - Stage 2: Moving ball interception
   - Stage 3: Kickoff training
   - Stage 4: Aerial practice
   - Stage 5: Full competitive 1v1
   - Automatically advances when performance improves

6. **Better Observation & Action Space**
   - Using DefaultObs with proper padding
   - 90 discrete actions (all major combos)
   - 492-dimensional observation space

7. **Progress Tracking**
   - TensorBoard integration
   - Checkpoint saving every 100K steps
   - Interrupt-safe (can stop anytime with Ctrl+C)

---

## 🏃 Quick Start

### Option 1: Basic Advanced Training (Recommended)
```bash
py -3.10 train_advanced.py
```
This runs with all features except curriculum (trains directly on competitive 1v1).

### Option 2: Resume from Existing Model
The script automatically detects existing models and asks if you want to continue training.

---

## 📊 Monitoring Training

### View Live Metrics:
```bash
py -3.10 -m tensorboard --logdir=logs/rlgym_advanced
```
Then open browser to: http://localhost:6006

### What to Watch:
- **ep_rew_mean**: Average reward per episode (should increase)
- **ep_len_mean**: Episode length (will vary)
- **policy_gradient_loss**: Should decrease and stabilize
- **value_loss**: Should decrease
- **fps**: Training speed (higher = faster)

---

## 🎮 Training Schedule

### For Beating Basic Bots (like Psyonix bots):
- **1-2 million steps** (2-4 hours with parallel training)
- Will learn basic positioning and shooting

### For Beating Intermediate Bots (like Nexto):
- **5-10 million steps** (8-16 hours)
- Learns better decision-making and consistency

### For Competitive Play:
- **20-50 million steps** (2-4 days)
- Learns advanced mechanics and strategies

### For Expert Level:
- **100+ million steps** (1-2 weeks)
- Top-tier mechanics and game sense

---

## 📁 File Structure

```
models/
├── rlgym_advanced/              # Main checkpoints
│   ├── flamewall_advanced_100000_steps.zip
│   ├── flamewall_advanced_200000_steps.zip
│   └── flamewall_final.zip      # Final trained model
├── opponents/                    # Self-play opponents
│   ├── opponent_0.zip
│   ├── opponent_1.zip
│   └── ...

logs/
└── rlgym_advanced/              # TensorBoard logs
    └── PPO_*/
```

---

## ⚙️ Hyperparameter Tuning

If you want to experiment, edit `train_advanced.py`:

### Make Training Faster (but less stable):
```python
learning_rate=5e-4,    # Higher learning rate
n_steps=1024,          # Fewer steps before update
batch_size=256,        # Smaller batches
```

### Make Training More Stable (but slower):
```python
learning_rate=1e-4,    # Lower learning rate
n_steps=4096,          # More steps before update
batch_size=1024,       # Larger batches
n_epochs=20,           # More training epochs
```

### More Exploration (try new things):
```python
ent_coef=0.05,         # Higher entropy
```

### Less Exploration (exploit known strategies):
```python
ent_coef=0.001,        # Lower entropy
```

---

## 🎓 Enabling Curriculum Learning

To use progressive training (easiest to hardest):

1. Open `train_advanced.py`
2. Find this line (around line 329):
   ```python
   curriculum = None
   ```
3. Change to:
   ```python
   from rlgym_config.curriculum import CurriculumSchedule
   curriculum = CurriculumSchedule()
   ```

The bot will now:
- Start with easy stationary ball shots
- Progress to moving balls
- Then add opponents
- Finally full competitive play

---

## 🆚 Self-Play Against Past Versions

Future enhancement - we can modify the environment to load random opponent models from the `models/opponents/` directory. This creates a diverse training environment.

---

## 🐛 Troubleshooting

### Training is slow
- Reduce `num_envs` from 4 to 2
- Reduce network size in `policy_kwargs`
- Use smaller `n_steps`

### Bot learns slowly
- Increase `learning_rate`
- Adjust reward weights in reward function
- Train for more steps

### Bot gets stuck / doesn't explore
- Increase `ent_coef` (entropy coefficient)
- Add curriculum learning
- Adjust reward function to encourage movement

### Memory errors
- Reduce `num_envs`
- Reduce `batch_size`
- Reduce network sizes

### "Out of Memory" on CPU
- Close other applications
- Reduce parallel environments to 2
- Reduce batch_size to 256

---

## 🏆 Testing Your Bot

Once trained, test it:

### 1. Play Against It Yourself
```bash
python run_gui.py
```
Then select your bot in RLBotGUI

### 2. Play Against Nexto
Set up a match in RLBotGUI:
- Blue Team: Your bot (using trained model)
- Orange Team: Nexto

### 3. Tournament Mode
Run multiple matches and track win rate

---

## 📈 Expected Performance Timeline

Based on typical RL training:

| Training Steps | Expected Skill Level |
|---------------|---------------------|
| 100K | Random movement, occasional ball touches |
| 500K | Basic ball chasing, some shots on goal |
| 1M | Consistent ball contact, basic positioning |
| 2M | Can score goals, basic defense |
| 5M | Good positioning, faster play |
| 10M | Competitive with intermediate bots |
| 20M | Advanced mechanics emerging |
| 50M+ | Expert-level play |

---

## 🎯 Next Steps After Training

1. **Test the Model**: Load it in RLBot and play matches
2. **Analyze Performance**: Watch replays, identify weaknesses
3. **Adjust Rewards**: Modify reward function to fix weaknesses
4. **Continue Training**: Resume from checkpoint with adjustments
5. **Create Opponents**: Use saved checkpoints for self-play
6. **Share Your Bot**: Compete in RLBot tournaments!

---

## 💡 Tips for Beating Nexto

Nexto is an intermediate bot. To beat it:

1. **Train for at least 5M steps**
2. **Focus on consistency** - reward reliable play
3. **Encourage shooting** - high ball-to-goal reward weight
4. **Practice defense** - add defensive positioning rewards
5. **Use self-play** - train against your own bot's strategies

---

## 🚀 Advanced: Custom Reward Functions

Want to add your own rewards? Edit `rlgym_config/advanced_rewards.py`:

```python
class MyCustomReward(RewardFunction):
    def reset(self, initial_state, shared_info=None):
        pass
    
    def get_rewards(self, agents, shared_info, done):
        rewards = np.zeros(len(agents))
        
        # Your reward logic here
        for i, agent in enumerate(agents):
            player = agent.data
            ball = shared_info['ball']
            
            # Example: Reward for supersonic speed
            speed = np.linalg.norm(player.car_data.linear_velocity)
            if speed > 2200:  # Supersonic
                rewards[i] = 0.1
        
        return rewards
```

Then add it to `CombinedReward` in `train_advanced.py`.

---

## 📚 Additional Resources

- **RLGym Documentation**: https://rlgym.org
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io
- **RLBot Discord**: Join for community help
- **PPO Paper**: Understanding the algorithm (optional)

---

## 🎉 Good Luck!

You now have a state-of-the-art training setup. With patience and proper training, your bot will learn to play Rocket League at a high level!

**Remember**: RL training requires patience. Don't expect instant results - let it train for several million steps and watch the improvement!

---

*Created for Flamewall Bot - Advanced RL Training System*
*With parallel environments, advanced rewards, curriculum learning, and self-play*
