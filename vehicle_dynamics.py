"""
Kinematic bicycle model for vehicle dynamics.
"""

import numpy as np
from typing import Tuple


class VehicleDynamics:
    """
    Kinematic bicycle model for vehicle motion.
    
    State: [x, y, yaw, velocity, acceleration]
    Actions: [steering, throttle, brake]
    """
    
    def __init__(
        self,
        wheelbase: float = 2.7,
        dt: float = 0.1,
        max_steering: float = 0.6,
        k_throttle: float = 3.0,
        k_brake: float = 5.0,
        max_velocity: float = 50.0
    ):
        """
        Initialize vehicle dynamics.
        
        Args:
            wheelbase: Distance between front and rear axles (meters)
            dt: Timestep for integration (seconds)
            max_steering: Maximum steering angle in radians
            k_throttle: Throttle gain (m/s²)
            k_brake: Brake gain (m/s²)
            max_velocity: Maximum velocity (m/s)
        """
        self.L = wheelbase
        self.dt = dt
        self.max_steering = max_steering
        self.k_throttle = k_throttle
        self.k_brake = k_brake
        self.max_velocity = max_velocity
        
        # State variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.a_prev = 0.0
        self.v_prev = 0.0
    
    def reset(self, x: float, y: float, yaw: float, v: float = 0.0):
        """
        Reset vehicle state.
        
        Args:
            x: Initial x position (meters)
            y: Initial y position (meters)
            yaw: Initial heading angle (radians)
            v: Initial velocity (m/s)
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a_prev = 0.0
        self.v_prev = v
    
    def step(
        self,
        steering: float,
        throttle: float,
        brake: float
    ) -> Tuple[float, float]:
        """
        Update vehicle state using kinematic bicycle model.
        
        Args:
            steering: Steering input in [-1, 1] (mapped to [-max_steering, max_steering])
            throttle: Throttle input in [0, 1]
            brake: Brake input in [0, 1]
            
        Returns:
            (a_long, jerk): longitudinal acceleration and jerk
        """
        # Clip actions to valid ranges
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        # Map steering to actual angle
        steering_angle = steering * self.max_steering
        
        # Compute longitudinal acceleration
        a_long = self.k_throttle * throttle - self.k_brake * brake
        
        # Store previous velocity for jerk calculation
        self.v_prev = self.v
        
        # Store steering angle for lateral acceleration calculation
        self.steering_angle = steering_angle
        
        # Kinematic bicycle model (Euler integration)
        self.x += self.v * np.cos(self.yaw) * self.dt
        self.y += self.v * np.sin(self.yaw) * self.dt
        self.yaw += (self.v / self.L) * np.tan(steering_angle) * self.dt
        self.v += a_long * self.dt
        
        # Enforce non-negative velocity
        self.v = max(self.v, 0.0)
        
        # Enforce maximum velocity
        self.v = min(self.v, self.max_velocity)
        
        # Normalize yaw to [-pi, pi]
        self.yaw = np.arctan2(np.sin(self.yaw), np.cos(self.yaw))
        
        # Compute actual acceleration from velocity change
        a_actual = (self.v - self.v_prev) / self.dt
        
        # Compute jerk
        jerk = (a_actual - self.a_prev) / self.dt
        
        # Update previous acceleration
        self.a_prev = a_actual
        
        return a_long, jerk
    
    def get_state(self) -> np.ndarray:
        """
        Get current vehicle state.
        
        Returns:
            [x, y, yaw, v, a_prev]
        """
        return np.array([self.x, self.y, self.yaw, self.v, self.a_prev])
    
    def get_position(self) -> np.ndarray:
        """
        Get current position.
        
        Returns:
            [x, y]
        """
        return np.array([self.x, self.y])
