"""Check if forward and reverse routes are geometrically identical"""
from minimal_grid_town_env import MinimalGridTownEnv
import numpy as np

def analyze_route(start, goal,name):
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30.0,
        start_node=start,
        goal_node=goal
    )
    
    # Get route info
    route_length = env.route.total_length
    num_points = len(env.route.route_points)
    curvatures = env.route.route_curvature
    max_curv = np.max(np.abs(curvatures))
    avg_curv = np.mean(np.abs(curvatures))
    
    # Get first few curvature samples
    curv_profile = curvatures[::100][:5]  # Sample every 100 points, take first 5
    
    print(f"\n{name}")
    print(f"  Length: {route_length:.1f}m")
    print(f"  Points: {num_points}")
    print(f"  Max curvature: {max_curv:.4f}")
    print(f"  Avg curvature: {avg_curv:.4f}")
    print(f"  Curvature profile (first 5 samples): {[f'{c:.3f}' for c in curv_profile]}")
    
    return env.route

print("=" * 70)
print("ROUTE GEOMETRY ANALYSIS")
print("=" * 70)
print("\nAre forward and reverse routes identical (just flipped)?")

route1 = analyze_route((0, 0), (5, 5), "Route (0,0) → (5,5)")
route2 = analyze_route((5, 5), (0, 0), "Route (5,5) → (0,0)")

# Check if routes are mirror images
print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)

if abs(route1.total_length - route2.total_length) < 0.1:
    print("✓ Same length")
else:
    print("✗ Different lengths!")

if len(route1.route_points) == len(route2.route_points):
    print("✓ Same number of points")
else:
    print("✗ Different number of points!")

# Check if they're spatial mirror images (reverse order, same positions)
route1_reversed = route1.route_points[::-1]
route2_points = route2.route_points

# Find max distance between corresponding points
max_dist = 0
for i in range(0, len(route1_reversed), 100):
    if i < len(route2_points):
        dist = np.linalg.norm(route1_reversed[i] - route2_points[i])
        max_dist = max(max_dist, dist)

print(f"\nMax spatial difference: {max_dist:.2f}m")
if max_dist < 1.0:
    print(" → Routes are geometrically identical (just reversed)")
    print("\n💡 The problem is NOT route geometry!")
    print("   The issue must be in how observations are computed relative to direction.")
else:
    print(" → Routes are DIFFERENT geometries!")
    print("\n💡 Dijkstra chose different paths for forward vs reverse!")

print("\n" + "=" * 70)
print("INSIGHT:")
print("=" * 70)
print("The agent sees:")
print("  • curvature_ahead - this changes completely when reversed")
print("  • distance_to_goal - this decreases differently")
print("  • heading_error - relative to tangent (direction-dependent)")
print("\nThese observations are inherently directional!")
print("Solution: Train with augmented data (both directions) or")
print("          use only ultra-local features (d, v) without looking ahead.")
