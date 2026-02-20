"""
Debug script to see what's happening with the trained agent.
"""

import numpy as np
from stable_baselines3 import PPO
from minimal_grid_town_env import MinimalGridTownEnv


def debug_agent():
    """Debug the trained agent step-by-step."""
    print("="*70)
    print("Debugging Trained Agent")
    print("="*70)
    
    # Load model
    print("\n1. Loading trained agent...")
    try:
        model = PPO.load("trained_agent")
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Create environment
    print("\n2. Creating environment...")
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        max_steps=1000,
        alpha=1.0,
        lambda_E=0.0001,
        lambda_d=1.0,
        lambda_lat=0.3,
        lambda_brake=0.5,
        lambda_jerk=0.05
    )
    print(f"   Route length: {env.route.total_length:.2f} m")
    print(f"   Lane width: {env.lane_width:.2f} m")
    print(f"   Lane boundary: ±{env.lane_width/2:.2f} m")
    
    # Reset
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"   Initial observation: {obs}")
    print(f"   Initial position: ({info['x']:.2f}, {info['y']:.2f})")
    print(f"   Initial velocity: {info['velocity']:.2f} m/s")
    
    # Run a few steps
    print("\n4. Running first 20 steps...")
    for step in range(20):
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        steering, throttle, brake = action
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print details
        v, d, heading_error, curv, dist_to_goal = obs
        print(f"   Step {step+1:2d}: "
              f"action=[{steering:6.3f}, {throttle:5.3f}, {brake:5.3f}], "
              f"v={v:5.2f} m/s, "
              f"d={d:6.3f} m, "
              f"reward={reward:8.2f}, "
              f"progress={info['route_progress']*100:5.1f}%")
        
        if terminated:
            print(f"   >>> TERMINATED at step {step+1}")
            if abs(d) > env.lane_width / 2:
                print(f"       Reason: Lane departure (|d|={abs(d):.3f} > {env.lane_width/2:.3f})")
            elif info['route_progress'] >= 0.99:
                print(f"       Reason: Goal reached!")
            break
        
        if truncated:
            print(f"   >>> TRUNCATED at step {step+1}")
            break
    
    print("\n" + "="*70)
    print("Debug Complete")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    debug_agent()
