"""
Simple training script - just run it!

Usage:
    python train.py                    # Quick test (100k steps, 3 min)
    python train.py --timesteps 500000 # Full training (500k steps, 15 min)
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Training timesteps (default: 100000)')
    parser.add_argument('--output', type=str, default='trained_agent',
                        help='Output model name (default: trained_agent)')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Training PPO Agent ({args.timesteps:,} timesteps)")
    print("=" * 70)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    base_env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        lane_width=8.0,         # 8m lanes (more room for learning steering!)
        alpha=50.0,             # Progress reward (along route)
        beta=40.0,              # Speed reward (increased to encourage faster cruising!)
        lambda_E=0.0005,        # Energy penalty
        lambda_d=10.0,          # Lane keeping penalty
        lambda_lat=2.0,         # Lateral acceleration penalty (reduced - allow higher speeds)
        lambda_brake=0.1,
        lambda_jerk=0.01
    )
    
    # Wrap environment for better training
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    print(f"    ✓ Route: {base_env.route.total_length:.0f}m, Lane: {base_env.lane_width}m")
    print(f"    ✓ Observation normalization enabled")
    
    # Create agent
    print("\n[2/4] Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,           # FIXED: Higher entropy for better exploration
        clip_range=0.2,           # Standard PPO clipping
        verbose=1
    )
    print("    ✓ Agent ready")
    
    # Train
    print(f"\n[3/4] Training ({args.timesteps:,} timesteps)...")
    print(f"    Time estimate: ~{args.timesteps // 600:.0f} seconds")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    
    # Save
    print(f"\n[4/4] Saving model...")
    model.save(args.output)
    # Save normalization statistics
    env.save(f"{args.output}_vec_normalize.pkl")
    print(f"    ✓ Saved as: {args.output}.zip")
    print(f"    ✓ Saved normalization: {args.output}_vec_normalize.pkl")
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating Agent (5 test episodes)")
    print("=" * 70)
    
    episode_rewards = []
    episode_progress = []
    
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            episode_reward += reward[0]
            done = dones[0]
        
        episode_rewards.append(episode_reward)
        progress = info[0].get('route_progress', 0) * 100
        episode_progress.append(progress)
        velocity = info[0].get('velocity', 0)
        
        status = "✓" if progress > 10 else "✗"
        print(f"  Episode {i+1}: {episode_reward:7.1f} reward, "
              f"{progress:5.1f}% progress, {velocity:5.2f} m/s  {status}")
    
    avg_progress = np.mean(episode_progress)
    success_rate = sum(1 for p in episode_progress if p > 10) / 5
    
    print("=" * 70)
    print(f"Average Progress: {avg_progress:.1f}%")
    print(f"Success Rate:     {success_rate*100:.0f}% (>10% progress)")
    
    if avg_progress > 50:
        print("\n🎉 Excellent! Agent completes most of the route!")
    elif avg_progress > 20:
        print("\n✓ Good! Agent learned to drive. Consider more training:")
        print(f"  python train.py --timesteps 500000")
    else:
        print("\n⚠ Agent needs more practice.")
    
    print("\nVisualize your agent:")
    print(f"  python visualize_agent.py {args.output}")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
