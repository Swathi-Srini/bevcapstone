"""
Rule-Based Path Following Controller
Demonstrates simple lateral control with PID for custom routes
"""

import numpy as np
import matplotlib.pyplot as plt
from minimal_grid_town_env import MinimalGridTownEnv
from scipy.interpolate import CubicSpline
import argparse


class SimplePathFollower:
    """Simple rule-based controller using lateral error feedback"""
    
    def __init__(self, kp=0.5, target_speed=10.0, lookahead=5.0):
        self.kp = kp  # Proportional gain for steering
        self.target_speed = target_speed
        self.lookahead = lookahead  # Lookahead distance for steering
        
    def compute_action(self, obs):
        """
        Compute control action from observation
        obs = [v, d, heading_error, distance_to_goal]
        """
        v, d, heading_error, distance_to_goal = obs
        
        # Steering: proportional to lateral deviation + heading error
        steering = -self.kp * d - 0.3 * heading_error
        steering = np.clip(steering, -0.6, 0.6)  # Limit to max steering
        
        # Speed control: simple throttle/brake logic
        speed_error = self.target_speed - v
        
        if speed_error > 0:  # Too slow
            throttle = np.clip(speed_error / 5.0, 0, 1)
            brake = 0
        else:  # Too fast
            throttle = 0
            brake = np.clip(-speed_error / 5.0, 0, 1)
        
        # Slow down corner (based on lateral deviation)
        if abs(d) > 2.0:  # Large lateral deviation suggests sharp corner ahead
            throttle *= 0.5
            if speed_error < -2.0:
                brake = 0.3
        
        return np.array([steering, throttle, brake])


def create_custom_route_interactive():
    """Interactive route creation (same as simple_route_test.py)"""
    waypoints = []
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def setup_plot():
        ax.clear()
        ax.set_xlim(-15, 165)
        ax.set_ylim(-15, 165)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Click to Create Route Waypoints', fontsize=14, fontweight='bold')
        
        # Grid
        for i in range(6):
            for j in range(6):
                x, y = i * 30, j * 30
                ax.plot(x, y, 'ko', markersize=8, alpha=0.5)
        
        # Waypoints
        if len(waypoints) > 0:
            wpts = np.array(waypoints)
            ax.plot(wpts[:, 0], wpts[:, 1], 'ro-', markersize=12, linewidth=3)
            
            for i, (x, y) in enumerate(waypoints):
                ax.text(x, y+2, str(i+1), ha='center', va='bottom',
                       fontsize=12, fontweight='bold', color='red')
            
            if len(waypoints) >= 2:
                smooth = interpolate_waypoints(wpts)
                if smooth is not None:
                    ax.plot(smooth[:, 0], smooth[:, 1], 'b-', linewidth=2, alpha=0.6)
        
        fig.canvas.draw()
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # Left click
            waypoints.append([event.xdata, event.ydata])
            print(f"✓ Waypoint {len(waypoints)}: ({event.xdata:.1f}, {event.ydata:.1f})")
            setup_plot()
        elif event.button == 3:  # Right click
            if waypoints:
                removed = waypoints.pop()
                print(f"✗ Removed: ({removed[0]:.1f}, {removed[1]:.1f})")
                setup_plot()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("\n" + "="*60)
    print("RULE-BASED CONTROLLER: Custom Route Mode")
    print("="*60)
    print("📍 LEFT CLICK: Add waypoint")
    print("🔙 RIGHT CLICK: Remove last waypoint")
    print("✅ CLOSE WINDOW: Start controller")
    print("="*60 + "\n")
    
    setup_plot()
    plt.show()
    
    if len(waypoints) < 2:
        return None
    
    return interpolate_waypoints(np.array(waypoints), num_points=500)


def interpolate_waypoints(waypoints, num_points=300):
    """Interpolate smooth path through waypoints"""
    if len(waypoints) < 2:
        return None
    
    # Remove duplicates
    unique = [waypoints[0]]
    for i in range(1, len(waypoints)):
        if np.linalg.norm(waypoints[i] - unique[-1]) > 0.1:
            unique.append(waypoints[i])
    
    waypoints = np.array(unique)
    if len(waypoints) < 2:
        return None
    
    dists = np.zeros(len(waypoints))
    for i in range(1, len(waypoints)):
        dists[i] = dists[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
    
    if dists[-1] < 0.01:
        return waypoints
    
    t = dists / dists[-1]
    for i in range(1, len(t)):
        if t[i] <= t[i-1]:
            t[i] = t[i-1] + 0.001
    
    try:
        cs_x = CubicSpline(t, waypoints[:, 0], bc_type='natural')
        cs_y = CubicSpline(t, waypoints[:, 1], bc_type='natural')
        t_smooth = np.linspace(0, 1, num_points)
        return np.column_stack([cs_x(t_smooth), cs_y(t_smooth)])
    except:
        return waypoints


def test_rule_based_controller(custom_route=None, kp=0.5, target_speed=10.0, episodes=1, visualize=True):
    """Test rule-based controller on custom or default route"""
    
    print("\n" + "="*60)
    print(f"RULE-BASED CONTROLLER TEST")
    print("="*60)
    print(f"Parameters: Kp={kp}, Target Speed={target_speed} m/s")
    print("="*60 + "\n")
    
    # Create environment
    env = MinimalGridTownEnv(
        grid_size=(6, 6),
        block_spacing=30,
        lane_width=8,
        start_node=(0, 0),
        goal_node=(5, 5),
        max_steps=3000,
        lambda_d=5.0,
        lambda_lat=0.1,
        lambda_E=0.0001,
        lambda_brake=0.05,
        lambda_jerk=0.005,
    )
    
    # Override with custom route if provided
    if custom_route is not None:
        env.route.points = custom_route
        env.route_length = np.sum(np.linalg.norm(np.diff(custom_route, axis=0), axis=1))
        print(f"✓ Using custom route: {len(custom_route)} points, {env.route_length:.1f}m")
    else:
        print(f"✓ Using default route: (0,0) → (5,5)")
    
    controller = SimplePathFollower(kp=kp, target_speed=target_speed)
    
    # Run episodes
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        trajectory = []
        
        while not done and step < 3000:
            action = controller.compute_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            trajectory.append([env.vehicle.x, env.vehicle.y, env.vehicle.yaw, env.vehicle.v])
            step += 1
        
        trajectory = np.array(trajectory)
        progress = info.get('route_progress', 0)
        
        print(f"Episode {ep+1}:")
        print(f"  Steps: {step}")
        print(f"  Progress: {progress*100:.1f}%")
        print(f"  Final position: ({trajectory[-1, 0]:.1f}, {trajectory[-1, 1]:.1f})")
        print(f"  Avg speed: {np.mean(trajectory[:, 3]):.1f} m/s")
        print(f"  Distance traveled: {np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)):.1f}m")
        
        # Visualize
        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Trajectory
            ax1.plot(env.route.points[:, 0], env.route.points[:, 1], 'b-', 
                    linewidth=3, alpha=0.5, label='Target Route')
            ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
                    linewidth=2, label='Rule-Based Controller')
            ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Start')
            ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=12, label='End')
            ax1.set_xlabel('X (meters)', fontsize=12)
            ax1.set_ylabel('Y (meters)', fontsize=12)
            ax1.set_title(f'Rule-Based Path Following ({progress*100:.1f}% complete)', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Velocity
            time = np.arange(len(trajectory)) * 0.1
            ax2.plot(time, trajectory[:, 3], 'b-', linewidth=2)
            ax2.axhline(target_speed, color='r', linestyle='--', label=f'Target: {target_speed} m/s')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.set_title('Velocity Profile', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rule-based path following controller')
    parser.add_argument('--custom', action='store_true', help='Draw custom route interactively')
    parser.add_argument('--kp', type=float, default=0.5, help='Proportional gain for steering')
    parser.add_argument('--speed', type=float, default=10.0, help='Target speed (m/s)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    custom_route = None
    if args.custom:
        custom_route = create_custom_route_interactive()
        if custom_route is None:
            print("❌ No route created, exiting")
            exit(0)
    
    test_rule_based_controller(
        custom_route=custom_route,
        kp=args.kp,
        target_speed=args.speed,
        episodes=args.episodes,
        visualize=not args.no_viz
    )
