# Flamewall RLGym Training Guide

This bot now includes RLGym integration for reinforcement learning training!

## Setup

### 1. Install Dependencies

First, install all required packages:

```powershell
pip install -r requirements.txt
```

This will install:
- RLGym (reinforcement learning environment)
- Stable-Baselines3 (RL algorithms)
- PyTorch (deep learning framework)
- TensorBoard (training visualization)
- And all RLBot dependencies

### 2. Training Your Bot

To start training your bot with reinforcement learning:

```powershell
python train_rlgym.py
```

**Training Options:**
- The bot will train for 10 million steps by default (this can take hours/days)
- Models are saved every 100,000 steps to `models/PPO/`
- You can stop training anytime with Ctrl+C and the model will be saved
- Training will run at 100x game speed for faster learning

**Monitor Training:**
To watch training progress in real-time:
```powershell
tensorboard --logdir=logs
```
Then open http://localhost:6006 in your browser

### 3. Using Your Trained Bot

After training, you can use your trained bot in RLBot:

1. Open `src/bot.cfg` and change the python_file to use the RLGym bot:
   ```
   python_file = rlgym_bot.py
   ```

2. Run your bot through RLBot GUI or command line as normal

The bot will automatically load the trained model from `models/PPO/flamewall_final_model.zip`

## File Structure

### Training Files
- `train_rlgym.py` - Main training script
- `rlgym_config/` - Custom RLGym configurations
  - `rewards.py` - Reward function (what behaviors to encourage)
  - `obs_builder.py` - Observation builder (what the agent sees)
  - `terminal_conditions.py` - Episode ending conditions

### Bot Files
- `src/rlgym_bot.py` - Bot that uses trained model
- `src/bot.py` - Original hand-coded bot (keep as backup!)

### Generated During Training
- `models/` - Saved models and checkpoints
- `logs/` - TensorBoard training logs

## Customizing Training

### Adjust Reward Function
Edit `rlgym_config/rewards.py` to change what behaviors are rewarded:
- Increase goal scoring reward
- Add rewards for specific mechanics (aerials, dribbling, etc.)
- Penalize bad behaviors

### Change Training Parameters
Edit `train_rlgym.py` to adjust:
- `learning_rate` - How fast the agent learns
- `n_steps` - Steps per training update
- `total_timesteps` - Total training duration
- `game_speed` - Training speed multiplier

### Modify Observation Space
Edit `rlgym_config/obs_builder.py` to change what the agent can see:
- Add information about opponents
- Include boost pad locations
- Add historical information

## Tips for Better Training

1. **Start Simple**: Train on 1v1 first, then expand to larger teams
2. **Use Checkpoints**: Training can take days - use the checkpointed models
3. **Adjust Rewards**: If the bot isn't learning desired behaviors, adjust reward weights
4. **Monitor Training**: Watch TensorBoard to ensure rewards are increasing
5. **Test Regularly**: Load checkpoints and test in actual matches to see progress
6. **Be Patient**: Good RL agents can take millions of steps to train

## Troubleshooting

**"Module not found" errors**: 
- Run `pip install -r requirements.txt` again

**Training is slow**:
- Make sure `game_speed=100` in train_rlgym.py
- Close other programs
- Training is CPU/GPU intensive

**Bot doesn't load model**:
- Make sure you've trained and saved a model first
- Check that `models/PPO/flamewall_final_model.zip` exists

**Poor performance**:
- Train longer (millions of steps)
- Adjust reward function
- Try different hyperparameters

## Advanced: Continue Training

To continue training from a checkpoint:

```python
# In train_rlgym.py, replace model creation with:
model = PPO.load("models/PPO/flamewall_rl_model_1000000_steps", env=env)
```

Then call `model.learn()` as normal.

## Resources

- [RLGym Documentation](https://rlgym.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [RLBot Discord](https://discord.gg/rlbot)
- [RLGym Discord](https://discord.gg/rlgym)
