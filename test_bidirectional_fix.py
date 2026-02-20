"""Quick bidirectional test"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np

def test_route_direction(model_path, start, goal, name):
    model = PPO.load(model_path)
    base_env = MinimalGridTownEnv(
        grid_size=(6, 6), block_spacing=30.0, lane_width=8.0,
        start_node=start, goal_node=goal, max_steps=1000,
        alpha=50.0, beta=10.0, lambda_E=0.0001, lambda_d=5.0,
    )
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(f"{model_path}_vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    
    progress_list = []
    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, info = env.step(action)
            done = dones[0]
        progress_list.append(info[0].get('route_progress', 0) * 100)
    
    avg = np.mean(progress_list)
    success = sum(1 for p in progress_list if p > 80)
    print(f"{name:30s}: {avg:5.1f}% avg, {success}/5 success")
    return avg

print("=" * 70)
print("BIDIRECTIONAL TEST - Agent WITHOUT curvature_ahead")
print("=" * 70)

test_route_direction("agent_no_curvature", (0, 0), (5, 5), "Forward: (0,0) → (5,5)")
test_route_direction("agent_no_curvature", (5, 5), (0, 0), "Reverse: (5,5) → (0,0)")
print()
test_route_direction("agent_no_curvature", (0, 0), (3, 5), "Forward: (0,0) → (3,5)")
test_route_direction("agent_no_curvature", (3, 5), (0, 0), "Reverse: (3,5) → (0,0)")
print()
test_route_direction("agent_no_curvature", (4, 4), (1, 2), "Your test: (4,4) → (1,2)")
test_route_direction("agent_no_curvature", (1, 2), (4, 4), "Reverse: (1,2) → (4,4)")

print("\n✅ If both directions work equally well, the fix is successful!")
