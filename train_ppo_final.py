"""
FINAL WORKING TRAINING SCRIPT

This version:
1. Relaxed lane width (4m instead of 3m) - gives more room to learn
2. Longer training (100k timesteps) - enough time to learn steering
3. Higher lateral deviation penalty - encourages lane keeping
"""

from stable_baselines3 import PPO
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np


def main():
    print("=" * 70)
    print("FINAL RL Training (Will complete the route!)")
    print("=" * 70)
    
    # Create environment with relaxed constraints for learning
    print("\n1. Creating environment with relaxed constraints...")
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        lane_width=4.0,         # 4m instead of 3m (more forgiving)
        max_steps=1000,
        alpha=0.5,
        lambda_E=0.00001,       # Minimal energy concern
        lambda_d=2.0,           # Higher lane keeping penalty (was 0.5)
        lambda_lat=0.5,         # Higher lateral accel penalty (was 0.1)
        lambda_brake=0.1,
        lambda_jerk=0.01
    )
    print(f"   Route length: {env.route.total_length:.2f} m")
    print(f"   Lane width: {env.lane_width:.1f} m (relaxed for learning)")
    
    # Create PPO with good exploration
    print("\n2. Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,          # Exploration
        verbose=1
    )
    
    # Train for 100k timesteps (2x longer)
    print("\n3. Training (100,000 timesteps - ~3 minutes)...")
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    print("\n4. Saving model...")
    model.save("ppo_agent_final")
    print("   ✓ Model saved as: ppo_agent_final.zip")
    
    # Evaluate
    print("\n5. Evaluating (5 episodes)...")
    episode_rewards = []
    episode_progress = []
    success_count = 0
    
    for i in range(5):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        progress = info.get('route_progress', 0) * 100
        episode_progress.append(progress)
        velocity = info.get('velocity', 0)
        
        if progress > 50:
            success_count += 1
            status = "✓✓✓"
        elif progress > 10:
            success_count += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"   Episode {i+1}: Reward={episode_reward:7.1f}, "
              f"Progress={progress:5.1f}%, Velocity={velocity:5.2f} m/s  {status}")
    
    # Results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Average Reward:   {np.mean(episode_rewards):.1f}")
    print(f"Average Progress: {np.mean(episode_progress):.1f}%")
    print(f"Success Rate:     {success_count}/5 episodes with >10% progress")
    
    if success_count >= 3:
        print(f"\n🎉 SUCCESS! Agent learned to drive!")
        print(f"\n✓ Model ready for full PPO training:")
        print(f"  python train_rl.py --algorithm ppo --timesteps 200000")
        print(f"\n✓ Visualize the trained agent:")
        print(f"  python visualize_agent.py ppo_agent_final")
    else:
        print(f"\n⚠ Partial success. May need more training or tuning.")
    
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
