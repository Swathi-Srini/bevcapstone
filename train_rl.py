"""
Train RL agents on MinimalGridTownEnv using Stable-Baselines3.

Supports multiple algorithms:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- A2C (Advantage Actor-Critic)
"""

import os
import argparse
from datetime import datetime
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from minimal_grid_town_env import MinimalGridTownEnv


class TrainingMonitorCallback:
    """Custom callback to monitor training progress."""
    
    def __init__(self, check_freq=1000):
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def __call__(self, locals_dict, globals_dict):
        # Get episode info
        if locals_dict.get('dones')[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episodes: {len(self.episode_rewards)}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.0f}")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        self.current_episode_reward += locals_dict.get('rewards')[0]
        self.current_episode_length += 1
        
        return True


def make_env(config=None, render_mode=None):
    """
    Create and wrap the environment.
    
    Args:
        config: Environment configuration dict
        render_mode: Rendering mode
        
    Returns:
        Wrapped environment
    """
    if config is None:
        config = {}
    
    config['render_mode'] = render_mode
    env = MinimalGridTownEnv(**config)
    env = Monitor(env)  # Record episode statistics
    
    return env


def get_default_config():
    """Get default environment configuration for training."""
    return {
        'grid_size': (6, 6),
        'block_spacing': 30.0,
        'lane_width': 4.0,        # 4m lanes (relaxed for learning)
        'dt': 0.1,
        'max_steps': 1000,
        # FIXED reward weights (from working train_ppo_final.py)
        'alpha': 0.7,             # Speed reward
        'lambda_E': 0.0005,       # Energy penalty
        'lambda_d': 5.0,          # Lane keeping penalty (balanced)
        'lambda_lat': 0.5,        # (Not used anymore - heading reward instead)
        'lambda_brake': 0.1,      # Low brake penalty
        'lambda_jerk': 0.01       # Very low jerk penalty
    }


def train_ppo(
    total_timesteps=100000,
    env_config=None,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    save_dir="models/ppo",
    log_dir="logs/ppo",
    eval_freq=5000,
    save_freq=10000
):
    """
    Train PPO agent.
    
    Args:
        total_timesteps: Total training timesteps
        env_config: Environment configuration
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation
        save_freq: Frequency of saving checkpoints
        
    Returns:
        Trained model
    """
    print("=" * 70)
    print("Training PPO (Proximal Policy Optimization)")
    print("=" * 70)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    if env_config is None:
        env_config = get_default_config()
    
    env = make_env(env_config)
    eval_env = make_env(env_config)
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="ppo_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Tensorboard logs: {log_dir}")
    print(f"To monitor: tensorboard --logdir {log_dir}")
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "ppo_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")
    
    return model


def train_sac(
    total_timesteps=100000,
    env_config=None,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    save_dir="models/sac",
    log_dir="logs/sac",
    eval_freq=5000,
    save_freq=10000
):
    """
    Train SAC agent (for continuous action spaces).
    
    SAC is generally more sample-efficient than PPO but requires more memory.
    """
    print("=" * 70)
    print("Training SAC (Soft Actor-Critic)")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if env_config is None:
        env_config = get_default_config()
    
    env = make_env(env_config)
    eval_env = make_env(env_config)
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="sac_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Tensorboard logs: {log_dir}")
    print(f"To monitor: tensorboard --logdir {log_dir}")
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    final_path = os.path.join(save_dir, "sac_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")
    
    return model


def train_td3(
    total_timesteps=100000,
    env_config=None,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    save_dir="models/td3",
    log_dir="logs/td3",
    eval_freq=5000,
    save_freq=10000
):
    """
    Train TD3 agent (Twin Delayed DDPG).
    
    TD3 is good for continuous control with less hyperparameter sensitivity than SAC.
    """
    print("=" * 70)
    print("Training TD3 (Twin Delayed DDPG)")
    print("=" * 70)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if env_config is None:
        env_config = get_default_config()
    
    env = make_env(env_config)
    eval_env = make_env(env_config)
    
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="td3_checkpoint"
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Tensorboard logs: {log_dir}")
    print(f"To monitor: tensorboard --logdir {log_dir}")
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )
    
    final_path = os.path.join(save_dir, "td3_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")
    
    return model


def evaluate_model(model, env_config=None, n_episodes=10, render=False):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        env_config: Environment configuration
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        
    Returns:
        Dictionary with evaluation statistics
    """
    if env_config is None:
        env_config = get_default_config()
    
    render_mode = "human" if render else None
    env = make_env(env_config, render_mode=render_mode)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check if reached goal
        if info.get('route_progress', 0) > 0.99:
            success_count += 1
        
        print(f"Episode {episode + 1}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Progress={info.get('route_progress', 0)*100:.1f}%")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes
    }
    
    print("\n" + "=" * 70)
    print("Evaluation Results:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
    print("=" * 70)
    
    env.close()
    return stats


def load_and_evaluate(model_path, algorithm='ppo', env_config=None, n_episodes=10, render=False):
    """
    Load a saved model and evaluate it.
    
    Args:
        model_path: Path to saved model
        algorithm: Algorithm type ('ppo', 'sac', 'td3', 'a2c')
        env_config: Environment configuration
        n_episodes: Number of episodes to evaluate
        render: Whether to render
        
    Returns:
        Evaluation statistics
    """
    algorithms = {
        'ppo': PPO,
        'sac': SAC,
        'td3': TD3,
        'a2c': A2C
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithms.keys())}")
    
    print(f"Loading {algorithm.upper()} model from: {model_path}")
    model = algorithms[algorithm].load(model_path)
    
    return evaluate_model(model, env_config, n_episodes, render)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train RL agents on MinimalGridTownEnv")
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3', 'a2c'],
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes after training')
    parser.add_argument('--render-eval', action='store_true',
                       help='Render during evaluation')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load pre-trained model (for evaluation only)')
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = get_default_config()
    
    if args.load:
        # Load and evaluate only
        print(f"Loading model from: {args.load}")
        load_and_evaluate(
            args.load,
            algorithm=args.algorithm,
            env_config=env_config,
            n_episodes=args.eval_episodes,
            render=args.render_eval
        )
    else:
        # Train
        if args.algorithm == 'ppo':
            model = train_ppo(total_timesteps=args.timesteps, env_config=env_config)
        elif args.algorithm == 'sac':
            model = train_sac(total_timesteps=args.timesteps, env_config=env_config)
        elif args.algorithm == 'td3':
            model = train_td3(total_timesteps=args.timesteps, env_config=env_config)
        else:
            raise ValueError(f"Algorithm {args.algorithm} not implemented")
        
        # Evaluate
        print("\n" + "=" * 70)
        print("Evaluating trained model...")
        print("=" * 70)
        evaluate_model(model, env_config, n_episodes=args.eval_episodes, render=args.render_eval)


if __name__ == "__main__":
    main()
