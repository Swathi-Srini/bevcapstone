"""
Training script with flexible route selection.

Usage:
    python train_custom_route.py --route diagonal       # Default (0,0) to (5,5)
    python train_custom_route.py --route horizontal     # (0,0) to (0,5) - easier
    python train_custom_route.py --route vertical       # (0,0) to (5,0) - easier  
    python train_custom_route.py --route complex        # (0,0) to (5,3)
    python train_custom_route.py --route lshape         # (0,0) to (3,5)
    python train_custom_route.py --route custom --start 1 2 --goal 4 3
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np

ROUTES = {
    'diagonal': ((0, 0), (5, 5), "Corner to corner"),
    'horizontal': ((0, 0), (0, 5), "Across top row"),
    'vertical': ((0, 0), (5, 0), "Down left column"),
    'short': ((0, 0), (3, 3), "Halfway diagonal"),
    'complex': ((0, 0), (5, 3), "Mixed route"),
    'lshape': ((0, 0), (3, 5), "L-shaped path"),
}

def main():
    parser = argparse.ArgumentParser(description='Train agent on custom route')
    parser.add_argument('--route', type=str, default='horizontal',
                        choices=list(ROUTES.keys()) + ['custom'],
                        help='Predefined route name')
    parser.add_argument('--start', type=int, nargs=2, default=None,
                        help='Custom start node (row col)')
    parser.add_argument('--goal', type=int, nargs=2, default=None,
                        help='Custom goal node (row col)')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Training timesteps')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model name')
    args = parser.parse_args()
    
    # Determine route
    if args.route == 'custom':
        if args.start is None or args.goal is None:
            print("Error: --start and --goal required for custom route")
            return
        start_node = tuple(args.start)
        goal_node = tuple(args.goal)
        route_desc = f"{start_node} to {goal_node}"
    else:
        start_node, goal_node, route_desc = ROUTES[args.route]
    
    # Auto-generate output name if not provided
    if args.output is None:
        args.output = f"agent_{args.route}_{args.timesteps//1000}k"
    
    print("=" * 70)
    print(f"Training PPO Agent - Custom Route")
    print("=" * 70)
    print(f"Route: {route_desc}")
    print(f"Start: {start_node} → Goal: {goal_node}")
    print(f"Timesteps: {args.timesteps:,}")
    print("=" * 70)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    base_env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        lane_width=8.0,
        start_node=start_node,
        goal_node=goal_node,
        max_steps=1000,
        alpha=50.0,
        beta=10.0,
        lambda_E=0.0001,
        lambda_d=5.0,
        lambda_lat=0.1,
        lambda_brake=0.05,
        lambda_jerk=0.005
    )
    
    # Wrap environment
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
        ent_coef=0.01,
        clip_range=0.2,
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
        
        status = "✓" if progress > 80 else "~" if progress > 30 else "✗"
        print(f"  Episode {i+1}: {episode_reward:7.1f} reward, "
              f"{progress:5.1f}% progress, {velocity:5.2f} m/s  {status}")
    
    avg_progress = np.mean(episode_progress)
    success_rate = sum(1 for p in episode_progress if p > 80) / 5
    
    print("=" * 70)
    print(f"Average Progress: {avg_progress:.1f}%")
    print(f"Success Rate:     {success_rate*100:.0f}% (>80% progress)")
    
    if avg_progress > 80:
        print("\n🎉 Excellent! Agent completes the route!")
    elif avg_progress > 50:
        print("\n✓ Good progress! Consider more training:")
        print(f"  python train_custom_route.py --route {args.route} --timesteps 100000")
    else:
        print("\n⚠ Agent needs more practice.")
    
    print(f"\nVisualize your agent:")
    print(f"  python visualize_agent.py {args.output}")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
