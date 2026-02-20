"""Test different routes to see which ones are interesting"""
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np

print("=" * 70)
print("Available Routes in 6x6 Grid")
print("=" * 70)

routes = [
    {"name": "Diagonal (Current)", "start": (0, 0), "goal": (5, 5), "description": "Corner to corner"},
    {"name": "Long Horizontal", "start": (0, 0), "goal": (0, 5), "description": "Across top row"},
    {"name": "Long Vertical", "start": (0, 0), "goal": (5, 0), "description": "Down left column"},
    {"name": "Short Diagonal", "start": (0, 0), "goal": (3, 3), "description": "Halfway diagonal"},
    {"name": "Complex Path", "start": (0, 0), "goal": (5, 3), "description": "Mixed route"},
    {"name": "L-Shape", "start": (0, 0), "goal": (3, 5), "description": "Up then across"},
]

print("\nAnalyzing all possible routes:\n")

for i, route_config in enumerate(routes, 1):
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        lane_width=8.0,
        start_node=route_config["start"],
        goal_node=route_config["goal"],
        alpha=50.0,
        beta=10.0,
        lambda_E=0.0001,
        lambda_d=5.0,
    )
    
    route_length = env.route.total_length
    num_points = len(env.route.route_points)
    
    print(f"{i}. {route_config['name']}")
    print(f"   Start: {route_config['start']} → Goal: {route_config['goal']}")
    print(f"   Description: {route_config['description']}")
    print(f"   Route length: {route_length:.1f} m")
    print(f"   Waypoints: {num_points}")
    print(f"   Difficulty: {'Easy' if route_length < 200 else 'Medium' if route_length < 350 else 'Hard'}")
    print()

print("=" * 70)
print("Recommendations:")
print("=" * 70)
print("• Short routes (< 200m): Good for quick testing")
print("• Medium routes (200-350m): Current training difficulty")
print("• Long routes (> 350m): Challenge for well-trained agents")
print()
print("Try training on route 2 or 6 for variety!")
