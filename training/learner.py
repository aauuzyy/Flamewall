import os
import sys
from pathlib import Path
import glob

# Add parent directory to Python path so training modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from redis import Redis
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator

from training.agent import get_agent
from training.obs import NectoObsBuilder
from training.parser import NectoAction
from training.reward import FlamewallRewardFunction

from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall


def find_latest_checkpoint(save_dir="ppos"):
    """
    Automatically find the most recent checkpoint in the save directory.
    Returns the path to the latest checkpoint or None if no checkpoints exist.
    """
    checkpoint_pattern = os.path.join(save_dir, "**", "checkpoint.pt")
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoints:
        print("No existing checkpoints found. Starting fresh training.")
        return None
    
    # Sort by modification time, most recent first
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest}")
    return latest


config = dict(
    seed=123,
    actor_lr=1e-4,
    critic_lr=1e-4,
    n_steps=1_000_000,  # 1M total steps
    batch_size=100_000,
    minibatch_size=10_000,
    epochs=30,
    gamma=0.995,
    iterations_per_save=50,  # Save every 50k steps (50 iterations * 1000 steps)
    iterations_per_model=50,  # Export model every 50k steps
    ent_coef=0.01,
)

if __name__ == "__main__":
    from rocket_learn.ppo import PPO

    run_id = None  # Set to None for fresh training, auto-checkpoint handles resume

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    wandb_key = os.environ.get("WANDB_KEY", "d9c330f3bf997ee4711e9e25752b1933fcea2cfc")
    redis_password = os.environ.get("REDIS_PASSWORD", None)
    _, ip = sys.argv
    wandb.login(key=wandb_key, timeout=30)
    
    # Tsunami Surge 1v1 Self-Play Training
    logger = wandb.init(
        name="tsunami-surge-1v1",
        project="tsunami-surge",
        id=run_id,
        config=config,
        tags=["1v1", "self-play", "tsunami"],
        settings=wandb.Settings(init_timeout=30)
    )
    print("=" * 60)
    print("🌊 TSUNAMI SURGE - 1v1 Self-Play Training")
    print("=" * 60)
    print(f"Training Configuration:")
    print(f"  • Mode: 1v1 Self-Play")
    print(f"  • Total Steps: {config['n_steps']:,}")
    print(f"  • Save Every: {config['iterations_per_save']}k steps")
    print(f"  • Game Speed: 100x")
    print(f"  • Match Length: 5 minutes")
    print(f"  • Boost: Limited (Competitive)")
    print("=" * 60)
    
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=redis_password)

    stat_trackers = [
        Speed(), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(), DistToBall()
    ]
    
    # Use tsunami name for rollout generation
    rollout_gen = RedisRolloutGenerator("tsunami",
                                        redis,
                                        lambda: NectoObsBuilder(None, 6),
                                        lambda: FlamewallRewardFunction(),
                                        NectoAction,
                                        save_every=logger.config.iterations_per_save,
                                        model_every=logger.config.iterations_per_model,
                                        logger=logger,
                                        clear=run_id is None,
                                        max_age=1,
                                        stat_trackers=stat_trackers
                                        )
    
    print("\n📊 Stat Trackers Active:")
    print("  • Speed, Demos, Touch, Episode Length")
    print("  • Boost Usage, Ball Positioning")
    print("  • Touch Height, Distance to Ball\n")

    agent = get_agent(actor_lr=logger.config.actor_lr, critic_lr=logger.config.critic_lr)

    # Force CPU usage (no CUDA available)
    device = torch.device("cpu")
    print(f"🖥️  Using device: {device}\n")

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        ent_coef=logger.config.ent_coef,
        logger=logger,
        device=device,
    )

    # Auto-detect and load the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(save_dir="ppos")
    if latest_checkpoint:
        print(f"\n♻️  Resuming training from: {latest_checkpoint}")
        alg.load(latest_checkpoint)
        alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr
        print("✅ Checkpoint loaded successfully! Continuing training...\n")
    else:
        print("\n� Starting fresh Tsunami Surge training from scratch!\n")

    print("🚀 Starting training loop...")
    print("💾 Checkpoints will be saved every 50k steps to ppos/")
    print("📦 Model exports (.zip) will be created every 50k steps")
    print("📈 Live metrics: Check Weights & Biases dashboard")
    print("=" * 60 + "\n")

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
