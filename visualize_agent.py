"""
Visualize a trained RL agent.

This script loads a trained model and visualizes its behavior.
"""

import argparse
import numpy as np
import os
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from minimal_grid_town_env import MinimalGridTownEnv


def visualize_agent(model_path, algorithm='ppo', n_episodes=3, start_node=None, goal_node=None):
    """
    Load and visualize a trained agent.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        algorithm: Algorithm type ('ppo', 'sac', 'td3', 'a2c')
        n_episodes: Number of episodes to visualize
        start_node: Optional custom start node (row, col)
        goal_node: Optional custom goal node (row, col)
    """
    algorithms = {
        'ppo': PPO,
        'sac': SAC,
        'td3': TD3,
        'a2c': A2C
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print("=" * 70)
    print(f"Visualizing {algorithm.upper()} Agent")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}.zip")
    model = algorithms[algorithm].load(model_path)
    
    # Create environment with rendering (MUST match training configuration!)
    env_kwargs = {
        'grid_size': (6, 6),
        'block_spacing': 30.0,
        'lane_width': 8.0,
        'alpha': 50.0,
        'beta': 40.0,
        'lambda_E': 0.0005,
        'lambda_d': 10.0,
        'lambda_lat': 2.0,
        'lambda_brake': 0.1,
        'lambda_jerk': 0.01,
        'render_mode': "human"
    }
    
    # Add custom route if specified
    if start_node is not None:
        env_kwargs['start_node'] = start_node
        print(f"Custom start: {start_node}")
    if goal_node is not None:
        env_kwargs['goal_node'] = goal_node
        print(f"Custom goal: {goal_node}")
    
    base_env = MinimalGridTownEnv(**env_kwargs)
    
    # Check if VecNormalize stats exist (for models trained with normalization)
    vec_normalize_path = f"{model_path}_vec_normalize.pkl"
    if os.path.exists(vec_normalize_path):
        print(f"Loading normalization stats from: {vec_normalize_path}")
        env = DummyVecEnv([lambda: base_env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during visualization
        env.norm_reward = False  # Don't normalize rewards during evaluation
        use_vec_env = True
    else:
        print("No normalization stats found - using raw environment")
        env = base_env
        use_vec_env = False
    
    print(f"\nRoute length: {base_env.route.total_length:.0f}m")
    print(f"Running {n_episodes} episodes...")
    print("Close the window after each episode to continue.\n")
    
    for episode in range(n_episodes):
        if use_vec_env:
            obs = env.reset()
        else:
            obs, info = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"Episode {episode + 1} starting...")
        
        while not done:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            if use_vec_env:
                obs, reward, dones, infos = env.step(action)
                episode_reward += reward[0]
                done = dones[0]
                info = infos[0]
                # Render the underlying environment
                env.venv.envs[0].render()
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                # Render
                env.render()
            
            step += 1
        
        # Print episode results
        progress = info.get('route_progress', 0) * 100
        velocity = info.get('velocity', 0)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Route Progress: {progress:.1f}%")
        print(f"  Final Velocity: {velocity:.2f} m/s")
        
        if progress > 99:
            print("  Status: SUCCESS ✓")
        else:
            print("  Status: INCOMPLETE")
        print()
    
    env.close()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Visualize trained RL agent")
    parser.add_argument('model', type=str, nargs='?', default='trained_agent',
                       help='Path to model file (without .zip)')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3', 'a2c'],
                       help='RL algorithm used')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to visualize')
    parser.add_argument('--start', type=int, nargs=2, default=None,
                       help='Custom start node (row col), e.g., --start 0 0')
    parser.add_argument('--goal', type=int, nargs=2, default=None,
                       help='Custom goal node (row col), e.g., --goal 5 5')
    
    args = parser.parse_args()
    
    # Convert start/goal to tuples if provided
    start_node = tuple(args.start) if args.start else None
    goal_node = tuple(args.goal) if args.goal else None
    
    visualize_agent(args.model, args.algorithm, args.episodes, start_node, goal_node)


if __name__ == "__main__":
    main()
