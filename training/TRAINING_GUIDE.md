# 🌊 Tsunami Surge 1v1 Training Setup

## Training Configuration
- **Mode**: 1v1 Self-Play (Tsunami vs Tsunami)
- **Total Steps**: 1,000,000 (1M)
- **Save Frequency**: Every 50,000 steps (20 checkpoints total)
- **Workers**: 10 parallel instances
- **Game Speed**: 100x simulation
- **Match Length**: 5 minutes
- **Boost**: Limited (Competitive settings)

## Prerequisites

### 1. Install Redis
Download Redis for Windows:
- Option A: https://github.com/microsoftarchive/redis/releases (download Redis-x64-3.0.504.msi)
- Option B: Use Docker: `docker run -d -p 6379:6379 redis`

Start Redis server (default port 6379)

### 2. Get Weights & Biases API Key
1. Sign up at https://wandb.ai/
2. Go to https://wandb.ai/authorize
3. Copy your API key
4. Set environment variable:
   ```powershell
   $env:WANDB_KEY="your_api_key_here"
   ```

### 3. Set Redis Password
```powershell
$env:REDIS_PASSWORD="your_password_or_leave_empty"
```

## How to Start Training

### Quick Start (10 Workers)
1. Open PowerShell in `training/` folder
2. Set environment variables (see above)
3. Start learner:
   ```powershell
   ..\training_env\Scripts\python.exe learner.py localhost
   ```
4. In another terminal, run:
   ```
   start_10_workers.bat
   ```

### Manual Start (Single Worker)
```powershell
# Terminal 1 - Learner
cd training
..\training_env\Scripts\python.exe learner.py localhost

# Terminal 2 - Worker
cd training
..\training_env\Scripts\python.exe worker.py localhost
```

## Training Progress

### Checkpoints
Saved every 50k steps to `ppos/tsunami_<timestamp>/tsunami_<iteration>/checkpoint.pt`

Example checkpoint locations:
- 50k steps: `ppos/tsunami_1729814400/tsunami_50/checkpoint.pt`
- 100k steps: `ppos/tsunami_1729814400/tsunami_100/checkpoint.pt`
- ...
- 1M steps: `ppos/tsunami_1729814400/tsunami_1000/checkpoint.pt`

### Model Exports (.zip)
Created every 50k steps - these can be loaded into the bot

### Live Metrics
View real-time training on Weights & Biases:
- Visit https://wandb.ai/your_username/tsunami-surge
- Metrics tracked:
  - Speed, Demos, Touches
  - Episode Length, Timeouts
  - Boost Usage
  - Ball positioning (Behind Ball)
  - Touch Height, Distance to Ball

## What's Being Learned

The bot trains by playing against itself (self-play):
- **Every match**: Tsunami Surge (v1) plays Tsunami Surge (v2)
- Both bots update from the same experiences
- Creates a competitive feedback loop
- Bot discovers strategies by playing against an equally skilled opponent

## Expected Training Time

With 10 workers at 100x speed:
- **50k steps**: ~30-60 minutes
- **500k steps**: ~5-8 hours
- **1M steps**: ~10-16 hours

## Stopping Training

Press `Ctrl+C` in the learner terminal. Training will automatically save a checkpoint. You can resume anytime - the script auto-detects the latest checkpoint!

## Using the Trained Model

After training completes:
1. Find the final checkpoint: `ppos/tsunami_<timestamp>/tsunami_1000/checkpoint.pt`
2. Export it to the bot model (you'll need to convert it from checkpoint to .pt format)
3. Copy to `src_flamewall_1v1/tsunami-model.pt`
4. Bot will use the trained model!

---

🌊 **Let the tsunami of training begin!** 🌊
