"""
MinimalGridTownEnv: A deterministic Gymnasium environment for route-constrained,
energy-aware autonomous driving research.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

from graph_pathfinding import GridGraph
from route_geometry import RouteGeometry
from vehicle_dynamics import VehicleDynamics
from energy_model import EnergyModel


class MinimalGridTownEnv(gym.Env):
    """
    A minimal custom Gymnasium environment for reinforcement learning research
    on route-constrained, energy-aware autonomous driving.
    
    The environment simulates a vehicle driving through a grid-based city,
    following a pre-computed route while optimizing for speed, energy efficiency,
    and lane-keeping.
    
    Key Features:
    - 2D grid-based road network with Dijkstra pathfinding
    - Kinematic bicycle vehicle model
    - Physics-based energy model
    - Left-hand driving rules (India-style)
    - Deterministic dynamics
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        # Grid parameters
        grid_size: Tuple[int, int] = (6, 6),
        block_spacing: float = 30.0,
        
        # Route parameters
        start_node: Optional[Tuple[int, int]] = None,
        goal_node: Optional[Tuple[int, int]] = None,
        interpolation_step: float = 0.5,
        
        # Road parameters
        lane_width: float = 3.0,
        road_width: float = 8.0,
        
        # Vehicle parameters
        wheelbase: float = 2.7,
        max_steering: float = 0.6,
        k_throttle: float = 3.0,
        k_brake: float = 5.0,
        max_velocity: float = 20.0,
        
        # Energy model parameters
        mass: float = 1500.0,
        frontal_area: float = 2.5,
        drag_coefficient: float = 0.3,
        rolling_resistance: float = 0.01,
        
        # Simulation parameters
        dt: float = 0.1,
        max_steps: int = 1000,
        
        # Reward parameters
        alpha: float = 50.0,          # Progress reward weight
        beta: float = 1.0,            # Speed reward weight
        lambda_E: float = 0.01,       # Energy penalty weight
        lambda_d: float = 2.0,        # Lateral deviation penalty weight
        lambda_lat: float = 0.5,      # Lateral acceleration penalty weight
        lambda_brake: float = 1.0,    # Brake penalty weight
        lambda_jerk: float = 0.1,     # Jerk penalty weight
        
        # Rendering
        render_mode: Optional[str] = None
    ):
        """
        Initialize the MinimalGridTownEnv.
        
        Args:
            grid_size: (rows, cols) number of intersections
            block_spacing: Distance between adjacent intersections (meters)
            start_node: Starting intersection (row, col), default is (0, 0)
            goal_node: Goal intersection (row, col), default is bottom-right
            interpolation_step: Spatial resolution for route interpolation (meters)
            lane_width: Width of a single lane (meters)
            road_width: Total road width (meters)
            wheelbase: Vehicle wheelbase (meters)
            max_steering: Maximum steering angle (radians)
            k_throttle: Throttle gain (m/s²)
            k_brake: Brake gain (m/s²)
            max_velocity: Maximum velocity (m/s)
            mass: Vehicle mass (kg)
            frontal_area: Frontal area (m²)
            drag_coefficient: Aerodynamic drag coefficient
            rolling_resistance: Rolling resistance coefficient
            dt: Simulation timestep (seconds)
            max_steps: Maximum episode steps
            alpha: Progress reward weight
            beta: Speed reward weight
            lambda_E: Energy penalty weight
            lambda_d: Lateral deviation penalty weight
            lambda_lat: Lateral acceleration penalty weight
            lambda_brake: Brake penalty weight
            lambda_jerk: Jerk penalty weight
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Store configuration
        self.grid_size = grid_size
        self.block_spacing = block_spacing
        self.lane_width = lane_width
        self.road_width = road_width
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Reward weights
        self.alpha = alpha
        self.beta = beta
        self.lambda_E = lambda_E
        self.lambda_d = lambda_d
        self.lambda_lat = lambda_lat
        self.lambda_brake = lambda_brake
        self.lambda_jerk = lambda_jerk
        
        # Build road network
        self.graph = GridGraph(grid_size, block_spacing)
        
        # Set start and goal nodes
        if start_node is None:
            self.start_node = (0, 0)
        else:
            self.start_node = start_node
        
        if goal_node is None:
            self.goal_node = (grid_size[0] - 1, grid_size[1] - 1)
        else:
            self.goal_node = goal_node
        
        # Compute route
        node_path = self.graph.dijkstra(self.start_node, self.goal_node)
        node_positions = [self.graph.get_node_position(node) for node in node_path]
        
        # Build route geometry
        self.route = RouteGeometry(
            node_positions,
            interpolation_step=interpolation_step,
            lane_width=lane_width,
            road_width=road_width
        )
        
        # Initialize vehicle dynamics
        self.vehicle = VehicleDynamics(
            wheelbase=wheelbase,
            dt=dt,
            max_steering=max_steering,
            k_throttle=k_throttle,
            k_brake=k_brake,
            max_velocity=max_velocity
        )
        
        # Initialize energy model
        self.energy = EnergyModel(
            mass=mass,
            frontal_area=frontal_area,
            drag_coefficient=drag_coefficient,
            rolling_resistance=rolling_resistance
        )
        
        # Define action space: [steering, throttle, brake]
        # steering ∈ [-1, 1], throttle ∈ [0, 1], brake ∈ [0, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space: [v, d, heading_error, distance_to_goal]
        # FIXED: Removed curvature_ahead to eliminate directional bias
        # FIXED: Bounded observations for better neural network training
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -road_width, -np.pi, 0.0]),
            high=np.array([max_velocity, road_width, np.pi, self.route.total_length]),
            dtype=np.float32
        )
        
        # Episode state
        self.step_count = 0
        self.s = 0.0  # Arc-length along route
        self.s_prev = 0.0
        self.route_idx = 0
        
        # Rendering
        self.fig = None
        self.ax = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed (for compatibility, but environment is deterministic)
            options: Additional options
            
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode counters
        self.step_count = 0
        self.s = 0.0
        self.s_prev = 0.0
        self.route_idx = 0
        
        # Reset vehicle to start position
        start_pos = self.route.route_points[0]
        start_yaw = np.arctan2(
            self.route.route_tangents[0][1],
            self.route.route_tangents[0][0]
        )
        # FIXED: Start with forward velocity to avoid "stay still" policy
        self.vehicle.reset(start_pos[0], start_pos[1], start_yaw, v=3.0)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: [steering, throttle, brake]
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Extract actions
        steering, throttle, brake = action
        
        # Update vehicle dynamics
        a_long, jerk = self.vehicle.step(steering, throttle, brake)
        
        # Get vehicle state
        pos = self.vehicle.get_position()
        v = self.vehicle.v
        
        # Calculate lateral acceleration (centripetal acceleration from turning)
        # a_lat = v² / r, where r = L / tan(δ) for bicycle model
        # So: a_lat = v² × tan(δ) / L
        if abs(self.vehicle.steering_angle) > 0.01:  # Avoid division issues
            a_lateral = (v ** 2) * abs(np.tan(self.vehicle.steering_angle)) / self.vehicle.L
        else:
            a_lateral = 0.0
        
        # Project onto route
        self.route_idx, s_proj, d, route_heading = self.route.project_point(pos)
        
        # Enforce monotonic progress
        self.s_prev = self.s
        self.s = max(self.s_prev, s_proj)
        
        # Compute heading error
        heading_error = self.vehicle.yaw - route_heading
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Compute energy consumption
        power = self.energy.compute_power(v, a_long)
        
        # Compute progress made this step
        progress_made = self.s - self.s_prev
        
        # CRITICAL FIX: Scale progress reward by how well we're following the route
        # If we're off the path, we shouldn't get full progress reward!
        # This prevents the exploit: "drive straight off-route and get progress anyway"
        # Forgiving for normal driving, strict for going off-route
        d_abs = abs(d)
        if d_abs < 1.0:
            progress_scale = 1.0  # Full reward - normal driving
        elif d_abs < 1.5:
            # Linear decrease from 1.0 to 0.5 (still gets some credit)
            progress_scale = 1.0 - (d_abs - 1.0) / 0.5 * 0.5
        elif d_abs < 2.0:
            # Linear decrease from 0.5 to 0 (minimal credit)
            progress_scale = 0.5 - (d_abs - 1.5) / 0.5 * 0.5
        else:
            # Too far off - NO progress reward, will hit lane departure soon anyway
            progress_scale = 0.0
        
        progress_made_scaled = progress_made * progress_scale
        
        # Compute reward
        reward = self._compute_reward(v, d, heading_error, power, brake, jerk, progress_made_scaled, a_lateral)
        
        # Goal completion bonus (huge reward for success)
        goal_reached = False
        if self.s >= self.route.total_length - 1.0:
            reward += 1000.0  # Big bonus!
            goal_reached = True
        
        # Update step count
        self.step_count += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Lane departure
        if abs(d) > self.lane_width / 2.0:
            terminated = True
        
        # Goal reached
        if goal_reached:
            terminated = True
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['power'] = power
        info['jerk'] = jerk
        info['lateral_deviation'] = d
        info['heading_error'] = heading_error
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            [v, d, heading_error, distance_to_goal]
        """
        # Get vehicle state
        pos = self.vehicle.get_position()
        v = self.vehicle.v
        yaw = self.vehicle.yaw
        
        # Project onto route
        route_idx, s_proj, d, route_heading = self.route.project_point(pos)
        
        # Heading error
        heading_error = yaw - route_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Distance to goal
        distance_to_goal = self.route.total_length - self.s
        
        obs = np.array([
            v,
            d,
            heading_error,
            distance_to_goal
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(
        self,
        v: float,
        d: float,
        heading_error: float,
        power: float,
        brake: float,
        jerk: float,
        progress_made: float,
        a_lateral: float
    ) -> float:
        """
        Compute reward function.
        
        R = alpha * progress + beta * v - lambda_E * P - lambda_d * d^2 
            - heading_penalty - lambda_lat * a_lat^2 - lambda_brake * brake - lambda_jerk * jerk^2
        
        Args:
            v: Velocity (m/s)
            d: Lateral deviation (meters)
            heading_error: Heading error from route direction (radians)
            power: Propulsion power (W)
            brake: Brake input [0, 1]
            jerk: Jerk (m/s³)
            progress_made: Distance traveled along route this step (m)
            a_lateral: Lateral acceleration from turning (m/s²)
            
        Returns:
            Reward value
        """
        # Progress reward (IMPORTANT: incentivizes forward movement along route)
        # Use alpha to scale progress along path (s - s_prev)
        r_progress = self.alpha * progress_made
        
        # Speed reward (helps exploration, much smaller than progress)
        r_speed = self.beta * v
        
        # Energy penalty
        r_energy = -self.lambda_E * power
        
        # Lateral deviation penalty - Use lambda_d to scale the penalty
        # Quadratic penalty that grows with distance from center
        r_lateral = -self.lambda_d * d**2
        
        # Heading alignment reward (encourages turning to follow route)
        # Gentle penalty for heading misalignment
        r_heading = -2.0 * abs(heading_error)  # ~3.1 penalty at 90° misalignment
        
        # Lateral acceleration penalty (physics constraint!)
        # Penalize excessive centrifugal forces
        # Comfortable: < 5 m/s², Max realistic: ~8 m/s², Emergency: ~15 m/s²
        a_lat_comfort = 5.0  # m/s²
        if a_lateral > a_lat_comfort:
            # Strong quadratic penalty for uncomfortable/dangerous lateral acceleration
            r_lat_acc = -self.lambda_lat * (a_lateral - a_lat_comfort) ** 2
        else:
            r_lat_acc = 0.0
        
        # Brake penalty
        r_brake = -self.lambda_brake * brake
        
        # Jerk penalty
        r_jerk = -self.lambda_jerk * jerk**2
        
        # CRITICAL FIX: Penalty for staying still (eliminates "do nothing" exploit)
        # If velocity is too low AND no progress, heavily penalize
        if v < 0.5 and progress_made < 0.01:
            r_still_penalty = -10.0  # Strong penalty for being stationary
        else:
            r_still_penalty = 0.0
        
        reward = r_progress + r_speed + r_energy + r_lateral + r_heading + r_lat_acc + r_brake + r_jerk + r_still_penalty
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            Dictionary with state information
        """
        return {
            'step': self.step_count,
            's': self.s,
            'route_progress': self.s / self.route.total_length if self.route.total_length > 0 else 0.0,
            'x': self.vehicle.x,
            'y': self.vehicle.y,
            'yaw': self.vehicle.yaw,
            'velocity': self.vehicle.v
        }
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            None for "human" mode, RGB array for "rgb_array" mode
        """
        if self.render_mode is None:
            return None
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrow
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        
        # Draw grid roads
        edges = self.graph.get_all_edges()
        for node1, node2 in edges:
            pos1 = self.graph.get_node_position(node1)
            pos2 = self.graph.get_node_position(node2)
            self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                        'k-', linewidth=2, alpha=0.3, zorder=1)
        
        # Draw route centerline
        centerline = self.route.get_centerline_points(
            [self.graph.get_node_position(node) 
             for node in self.graph.dijkstra(self.start_node, self.goal_node)]
        )
        self.ax.plot(centerline[:, 0], centerline[:, 1], 
                    'b--', linewidth=2, label='Route centerline', zorder=2)
        
        # Draw left lane centerline (where vehicle should follow)
        self.ax.plot(self.route.route_points[:, 0], self.route.route_points[:, 1],
                    'g-', linewidth=2, label='Left lane center', zorder=3)
        
        # Draw vehicle
        vehicle_length = 4.5
        vehicle_width = 2.0
        
        # Vehicle as a rectangle
        vehicle_rect = Rectangle(
            (self.vehicle.x - vehicle_length/2, self.vehicle.y - vehicle_width/2),
            vehicle_length, vehicle_width,
            angle=np.degrees(self.vehicle.yaw),
            fill=True, facecolor='red', edgecolor='darkred', linewidth=2, zorder=5
        )
        self.ax.add_patch(vehicle_rect)
        
        # Vehicle heading indicator
        arrow_length = 3.0
        dx = arrow_length * np.cos(self.vehicle.yaw)
        dy = arrow_length * np.sin(self.vehicle.yaw)
        self.ax.arrow(self.vehicle.x, self.vehicle.y, dx, dy,
                     head_width=1.0, head_length=0.5, fc='yellow', ec='orange', 
                     linewidth=2, zorder=6)
        
        # Draw start and goal
        start_pos = self.graph.get_node_position(self.start_node)
        goal_pos = self.graph.get_node_position(self.goal_node)
        self.ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, 
                    label='Start', zorder=4)
        self.ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, 
                    label='Goal', zorder=4)
        
        # Set axis properties
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        self.ax.set_title(f'MinimalGridTownEnv (Step: {self.step_count}, v: {self.vehicle.v:.2f} m/s)',
                         fontsize=14)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Set axis limits with padding
        padding = 10.0
        x_min = -padding
        x_max = (self.grid_size[1] - 1) * self.block_spacing + padding
        y_min = -padding
        y_max = (self.grid_size[0] - 1) * self.block_spacing + padding
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def close(self):
        """Close the environment and cleanup resources."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None
