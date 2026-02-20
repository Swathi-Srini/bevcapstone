"""
Interactive Manual Route Creator
Click on the grid to create custom waypoints, then test the agent on your custom route.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from minimal_grid_town_env import MinimalGridTownEnv
import argparse


class ManualRouteCreator:
    def __init__(self, grid_size=6, block_spacing=30, lane_width=8):
        self.grid_size = grid_size
        self.block_spacing = block_spacing
        self.lane_width = lane_width
        self.waypoints = []
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_grid()
        
        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        print("\n" + "="*60)
        print("MANUAL ROUTE CREATOR")
        print("="*60)
        print("📍 LEFT CLICK: Add waypoint")
        print("🔙 RIGHT CLICK: Remove last waypoint")
        print("✅ MIDDLE CLICK (or press ENTER): Finish route")
        print("❌ Press 'Q': Quit without saving")
        print("="*60 + "\n")
        
    def setup_grid(self):
        """Draw the grid layout"""
        self.ax.clear()
        self.ax.set_xlim(-self.block_spacing/2, self.grid_size * self.block_spacing + self.block_spacing/2)
        self.ax.set_ylim(-self.block_spacing/2, self.grid_size * self.block_spacing + self.block_spacing/2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        self.ax.set_title('Click to Create Custom Route Waypoints', fontsize=14, fontweight='bold')
        
        # Draw grid intersections
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = i * self.block_spacing
                y = j * self.block_spacing
                self.ax.plot(x, y, 'ko', markersize=8, alpha=0.5)
                self.ax.text(x, y-3, f'({i},{j})', ha='center', va='top', fontsize=8, alpha=0.6)
        
        # Draw existing waypoints
        if len(self.waypoints) > 0:
            waypoints_array = np.array(self.waypoints)
            self.ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], 'ro-', 
                        markersize=10, linewidth=2, label='Waypoints', zorder=5)
            
            # Number the waypoints
            for i, (x, y) in enumerate(self.waypoints):
                self.ax.text(x, y+2, str(i+1), ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color='red', zorder=6)
            
            # Show interpolated smooth path if we have enough points
            if len(self.waypoints) >= 2:
                smooth_path = self.interpolate_path(waypoints_array)
                self.ax.plot(smooth_path[:, 0], smooth_path[:, 1], 'b-', 
                           linewidth=2, alpha=0.5, label='Smooth Path')
            
            self.ax.legend(loc='upper right')
        
        # Show instructions on plot
        info_text = f"Waypoints: {len(self.waypoints)}"
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add waypoint
            x, y = event.xdata, event.ydata
            self.waypoints.append([x, y])
            print(f"✓ Added waypoint {len(self.waypoints)}: ({x:.1f}, {y:.1f})")
            self.setup_grid()
            
        elif event.button == 3:  # Right click - remove last waypoint
            if len(self.waypoints) > 0:
                removed = self.waypoints.pop()
                print(f"✗ Removed waypoint: ({removed[0]:.1f}, {removed[1]:.1f})")
                self.setup_grid()
            else:
                print("⚠ No waypoints to remove")
                
        elif event.button == 2:  # Middle click - finish
            if len(self.waypoints) >= 2:
                plt.close(self.fig)
            else:
                print("⚠ Need at least 2 waypoints to create a route")
    
    def interpolate_path(self, waypoints, num_points=200):
        """Create smooth path through waypoints using cubic spline interpolation"""
        if len(waypoints) < 2:
            return waypoints
        
        # Remove duplicate waypoints (within 0.1m tolerance)
        unique_waypoints = [waypoints[0]]
        for i in range(1, len(waypoints)):
            if np.linalg.norm(waypoints[i] - unique_waypoints[-1]) > 0.1:
                unique_waypoints.append(waypoints[i])
        
        waypoints = np.array(unique_waypoints)
        
        if len(waypoints) < 2:
            return waypoints
        
        # Calculate cumulative distance along waypoints
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])
            distances[i] = distances[i-1] + max(dist, 0.01)  # Ensure minimum spacing
        
        # Normalize distances
        if distances[-1] > 0:
            t = distances / distances[-1]
        else:
            t = np.linspace(0, 1, len(waypoints))
        
        # Ensure strictly increasing
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + 0.001
        
        try:
            # Create cubic splines for x and y
            cs_x = CubicSpline(t, waypoints[:, 0], bc_type='natural')
            cs_y = CubicSpline(t, waypoints[:, 1], bc_type='natural')
            
            # Generate smooth path
            t_smooth = np.linspace(0, 1, num_points)
            smooth_path = np.column_stack([cs_x(t_smooth), cs_y(t_smooth)])
            
            return smooth_path
        except Exception as e:
            print(f"⚠ Spline interpolation failed: {e}, using linear interpolation")
            return waypoints
    
    def create_route(self):
        """Show interactive interface and return the created route"""
        plt.show()
        
        if len(self.waypoints) < 2:
            print("\n❌ Route creation cancelled or insufficient waypoints")
            return None
        
        # Create smooth path
        waypoints_array = np.array(self.waypoints)
        smooth_path = self.interpolate_path(waypoints_array, num_points=500)
        
        print(f"\n✅ Route created with {len(self.waypoints)} waypoints")
        print(f"   Smooth path: {len(smooth_path)} points")
        print(f"   Total length: {self.calculate_length(smooth_path):.1f} meters")
        
        return smooth_path
    
    def calculate_length(self, path):
        """Calculate total path length"""
        if len(path) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))


def test_agent_on_custom_route(route_points, model_path="models/ppo/best_model", episodes=3):
    """Test trained agent on custom route"""
    print("\n" + "="*60)
    print(f"Testing Agent on Custom Route ({episodes} episodes)")
    print("="*60)
    
    # Create environment with custom route
    def make_env():
        env = MinimalGridTownEnv(
            grid_size=(6, 6),
            block_spacing=30,
            lane_width=8,
            start_node=(0, 0),
            goal_node=(5, 5),
            max_steps=2000,
            # Reduced penalties
            lambda_d=5.0,
            lambda_lat=0.1,
            lambda_E=0.0001,
            lambda_brake=0.05,
            lambda_jerk=0.005,
        )
        # Override the route with custom one
        env.route.points = route_points
        env.route_length = np.sum(np.linalg.norm(np.diff(route_points, axis=0), axis=1))
        return env
    
    # Load model and normalization
    env = DummyVecEnv([make_env])
    try:
        env = VecNormalize.load(f"{model_path}_vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False
        print(f"✓ Loaded VecNormalize statistics")
    except (FileNotFoundError, AssertionError) as e:
        print(f"⚠ VecNormalize not compatible or not found ({type(e).__name__}), using unnormalized environment")
    
    model = PPO.load(model_path, env=env)
    print(f"✓ Loaded model from {model_path}")
    
    # Run episodes
    results = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        max_progress = 0
        
        while not done and step < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            # Track progress
            if 'route_progress' in info[0]:
                max_progress = max(max_progress, info[0]['route_progress'])
            
            step += 1
        
        results.append({
            'episode': ep + 1,
            'steps': step,
            'progress': max_progress * 100,
            'reward': total_reward
        })
        
        print(f"  Episode {ep+1}: {step:4d} steps | "
              f"Progress: {max_progress*100:5.1f}% | "
              f"Reward: {total_reward:7.1f}")
    
    # Summary
    avg_progress = np.mean([r['progress'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print("\n" + "-"*60)
    print(f"Average Progress: {avg_progress:.1f}%")
    print(f"Average Steps: {avg_steps:.0f}")
    print("="*60 + "\n")
    
    return results


def visualize_agent_on_custom_route(route_points, model_path="models/ppo/best_model"):
    """Visualize agent following custom route with matplotlib animation"""
    print("\n" + "="*60)
    print("Visualizing Agent on Custom Route")
    print("="*60)
    
    # Create environment with custom route
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30,
        lane_width=8,
        start_node=(0, 0),
        goal_node=(5, 5),
        max_steps=2000,
        lambda_d=5.0,
        lambda_lat=0.1,
        lambda_E=0.0001,
        lambda_brake=0.05,
        lambda_jerk=0.005,
    )
    
    # Override the route
    env.route.points = route_points
    env.route_length = np.sum(np.linalg.norm(np.diff(route_points, axis=0), axis=1))
    
    # Wrap and load model
    env_vec = DummyVecEnv([lambda: env])
    try:
        env_vec = VecNormalize.load(f"{model_path}_vec_normalize.pkl", env_vec)
        env_vec.training = False
        env_vec.norm_reward = False
    except FileNotFoundError:
        pass
    
    model = PPO.load(model_path, env=env_vec)
    
    # Run episode and collect trajectory
    obs = env_vec.reset()
    trajectory = []
    done = False
    step = 0
    
    while not done and step < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_vec.step(action)
        
        # Get vehicle state
        x, y = env.vehicle.x, env.vehicle.y
        heading = env.vehicle.psi
        v = env.vehicle.v
        trajectory.append([x, y, heading, v])
        step += 1
    
    trajectory = np.array(trajectory)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Trajectory
    ax1.plot(route_points[:, 0], route_points[:, 1], 'b-', linewidth=3, alpha=0.5, label='Target Route')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Agent Trajectory')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, label='End')
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Custom Route Following', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right plot: Velocity profile
    time_steps = np.arange(len(trajectory)) * 0.1  # dt = 0.1s
    ax2.plot(time_steps, trajectory[:, 3], 'b-', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Velocity Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Completed in {step} steps ({step*0.1:.1f} seconds)")
    print(f"  Final progress: {info[0].get('route_progress', 0)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and test custom routes')
    parser.add_argument('--model', type=str, default='models/ppo/best_model',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes')
    parser.add_argument('--visualize', action='store_true',
                       help='Show animated visualization')
    args = parser.parse_args()
    
    # Create custom route interactively
    creator = ManualRouteCreator(grid_size=6, block_spacing=30, lane_width=8)
    custom_route = creator.create_route()
    
    if custom_route is not None:
        # Test agent on custom route
        test_agent_on_custom_route(custom_route, model_path=args.model, episodes=args.episodes)
        
        # Optional visualization
        if args.visualize:
            visualize_agent_on_custom_route(custom_route, model_path=args.model)
    else:
        print("\n❌ No route created. Exiting.")
