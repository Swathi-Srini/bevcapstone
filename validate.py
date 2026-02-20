"""
Validation script to check that all components work correctly.
Run this before using the environment to ensure everything is set up properly.
"""

import sys
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from graph_pathfinding import GridGraph
        from route_geometry import RouteGeometry
        from vehicle_dynamics import VehicleDynamics
        from energy_model import EnergyModel
        from minimal_grid_town_env import MinimalGridTownEnv
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_graph():
    """Test graph construction and pathfinding."""
    print("\nTesting graph construction and pathfinding...")
    try:
        from graph_pathfinding import GridGraph
        
        graph = GridGraph((4, 4), 30.0)
        path = graph.dijkstra((0, 0), (3, 3))
        
        assert len(path) > 0, "Path should not be empty"
        assert path[0] == (0, 0), "Path should start at start node"
        assert path[-1] == (3, 3), "Path should end at goal node"
        
        print(f"✓ Graph test passed (path length: {len(path)} nodes)")
        return True
    except Exception as e:
        print(f"✗ Graph test failed: {e}")
        return False

def test_route_geometry():
    """Test route geometry construction."""
    print("\nTesting route geometry...")
    try:
        from route_geometry import RouteGeometry
        
        # Create simple path
        nodes = [
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            np.array([10.0, 10.0])
        ]
        
        route = RouteGeometry(nodes, interpolation_step=0.5)
        
        assert route.route_points is not None, "Route points should exist"
        assert len(route.route_points) > len(nodes), "Route should be interpolated"
        assert route.total_length > 0, "Route length should be positive"
        
        print(f"✓ Route geometry test passed (total length: {route.total_length:.2f} m)")
        return True
    except Exception as e:
        print(f"✗ Route geometry test failed: {e}")
        return False

def test_vehicle_dynamics():
    """Test vehicle dynamics."""
    print("\nTesting vehicle dynamics...")
    try:
        from vehicle_dynamics import VehicleDynamics
        
        vehicle = VehicleDynamics(dt=0.1)
        vehicle.reset(0.0, 0.0, 0.0, 0.0)
        
        # Test forward motion
        for _ in range(10):
            vehicle.step(0.0, 0.5, 0.0)  # Straight, half throttle
        
        assert vehicle.v > 0, "Vehicle should have positive velocity"
        assert vehicle.x > 0, "Vehicle should have moved forward"
        
        print(f"✓ Vehicle dynamics test passed (final v: {vehicle.v:.2f} m/s)")
        return True
    except Exception as e:
        print(f"✗ Vehicle dynamics test failed: {e}")
        return False

def test_energy_model():
    """Test energy model."""
    print("\nTesting energy model...")
    try:
        from energy_model import EnergyModel
        
        energy = EnergyModel()
        
        # Test at different speeds
        power_0 = energy.compute_power(0.0, 1.0)
        power_10 = energy.compute_power(10.0, 0.0)
        
        assert power_0 >= 0, "Power should be non-negative"
        assert power_10 >= 0, "Power should be non-negative"
        
        print(f"✓ Energy model test passed (P @ 10 m/s: {power_10/1000:.2f} kW)")
        return True
    except Exception as e:
        print(f"✗ Energy model test failed: {e}")
        return False

def test_environment():
    """Test full environment."""
    print("\nTesting full environment...")
    try:
        from minimal_grid_town_env import MinimalGridTownEnv
        
        env = MinimalGridTownEnv(
            grid_size=(4, 4),
            block_spacing=20.0,
            max_steps=50
        )
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (5,), "Observation should have 5 dimensions"
        
        # Test step
        action = np.array([0.0, 0.5, 0.0])  # Straight, half throttle
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (5,), "Observation should have 5 dimensions"
        assert isinstance(reward, (int, float)), "Reward should be a number"
        
        # Test multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        env.close()
        
        print(f"✓ Environment test passed (progress: {info['route_progress']*100:.1f}%)")
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 70)
    print("MinimalGridTownEnv Validation")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_graph,
        test_route_geometry,
        test_vehicle_dynamics,
        test_energy_model,
        test_environment
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    if all(results):
        print("✓ All validation tests passed!")
        print("=" * 70)
        print("\nThe environment is ready to use. You can now run:")
        print("  python demo.py")
        return 0
    else:
        print("✗ Some validation tests failed!")
        print("=" * 70)
        print("\nPlease check the errors above and ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
