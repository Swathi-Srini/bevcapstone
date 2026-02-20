"""Test if there's a directional bias in routes"""
from minimal_grid_town_env import MinimalGridTownEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

def test_route_direction(model_path, start, goal, route_name):
    """Test agent on a specific route direction"""
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    base_env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        lane_width=8.0,
        start_node=start,
        goal_node=goal,
        max_steps=1000,
        alpha=50.0,
        beta=10.0,
        lambda_E=0.0001,
        lambda_d=5.0,
    )
    
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(f"{model_path}_vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    
    # Test 5 episodes
    successes = 0
    progress_list = []
    
    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
        
        progress = info[0].get('route_progress', 0) * 100
        progress_list.append(progress)
        if progress > 80:
            successes += 1
    
    avg_progress = np.mean(progress_list)
    
    print(f"{route_name:30s}: {avg_progress:5.1f}% avg, {successes}/5 success")
    return avg_progress

print("=" * 70)
print("DIRECTIONAL BIAS TEST")
print("=" * 70)
print("Testing if agent has directional preference...\n")

# Test forward and reverse pairs
print("Forward vs Reverse Routes:")
print("-" * 70)

test_route_direction("trained_agent_fixed", (0, 0), (5, 5), "Forward: (0,0) → (5,5)")
test_route_direction("trained_agent_fixed", (5, 5), (0, 0), "Reverse: (5,5) → (0,0)")
print()

test_route_direction("trained_agent_fixed", (0, 0), (0, 5), "Forward: (0,0) → (0,5)")
test_route_direction("trained_agent_fixed", (0, 5), (0, 0), "Reverse: (0,5) → (0,0)")
print()

test_route_direction("trained_agent_fixed", (0, 0), (5, 0), "Forward: (0,0) → (5,0)")
test_route_direction("trained_agent_fixed", (5, 0), (0, 0), "Reverse: (5,0) → (0,0)")
print()

test_route_direction("trained_agent_fixed", (0, 0), (3, 5), "Forward: (0,0) → (3,5)")
test_route_direction("trained_agent_fixed", (3, 5), (0, 0), "Reverse: (3,5) → (0,0)")
print()

test_route_direction("trained_agent_fixed", (4, 4), (1, 2), "Your example: (4,4) → (1,2)")
test_route_direction("trained_agent_fixed", (1, 2), (4, 4), "Reverse: (1,2) → (4,4)")

print("\n" + "=" * 70)
print("If reverse routes have lower success, there's a directional bias!")
print("=" * 70)
