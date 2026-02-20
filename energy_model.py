"""
Energy model for vehicle propulsion on flat road.
"""

import numpy as np


class EnergyModel:
    """
    Computes propulsion power for vehicle on flat road.
    
    Considers:
    - Aerodynamic drag
    - Rolling resistance
    - Inertial force
    
    Assumes flat road (gradient = 0).
    """
    
    def __init__(
        self,
        mass: float = 1500.0,
        frontal_area: float = 2.5,
        drag_coefficient: float = 0.3,
        rolling_resistance: float = 0.01,
        air_density: float = 1.225,
        gravity: float = 9.81
    ):
        """
        Initialize energy model with vehicle and environment parameters.
        
        Args:
            mass: Vehicle mass (kg)
            frontal_area: Frontal area (m²)
            drag_coefficient: Aerodynamic drag coefficient (dimensionless)
            rolling_resistance: Rolling resistance coefficient (dimensionless)
            air_density: Air density (kg/m³)
            gravity: Gravitational acceleration (m/s²)
        """
        self.m = mass
        self.A = frontal_area
        self.Cd = drag_coefficient
        self.Crr = rolling_resistance
        self.rho = air_density
        self.g = gravity
    
    def compute_power(self, v: float, a_long: float) -> float:
        """
        Compute mechanical power required for propulsion.
        
        Power is computed only when vehicle is accelerating or maintaining speed
        against resistive forces (drag and rolling resistance).
        
        Args:
            v: Vehicle velocity (m/s)
            a_long: Longitudinal acceleration (m/s²)
            
        Returns:
            Power in Watts (always non-negative)
        """
        # Aerodynamic drag force
        F_drag = 0.5 * self.rho * self.Cd * self.A * v**2
        
        # Rolling resistance force
        F_roll = self.Crr * self.m * self.g
        
        # Inertial force
        F_inertial = self.m * a_long
        
        # Total force required
        F_total = F_drag + F_roll + F_inertial
        
        # Propulsion force (clamped to non-negative)
        # Negative force means regenerative braking, which we don't model in power
        F_propulsion = max(F_total, 0.0)
        
        # Mechanical power
        # P = F * v
        power = F_propulsion * v
        
        return power
    
    def compute_forces(self, v: float, a_long: float) -> dict:
        """
        Compute individual force components for analysis.
        
        Args:
            v: Vehicle velocity (m/s)
            a_long: Longitudinal acceleration (m/s²)
            
        Returns:
            Dictionary with force components:
            - 'drag': Aerodynamic drag force (N)
            - 'roll': Rolling resistance force (N)
            - 'inertial': Inertial force (N)
            - 'total': Total force (N)
            - 'propulsion': Propulsion force (N, clamped to non-negative)
        """
        F_drag = 0.5 * self.rho * self.Cd * self.A * v**2
        F_roll = self.Crr * self.m * self.g
        F_inertial = self.m * a_long
        F_total = F_drag + F_roll + F_inertial
        F_propulsion = max(F_total, 0.0)
        
        return {
            'drag': F_drag,
            'roll': F_roll,
            'inertial': F_inertial,
            'total': F_total,
            'propulsion': F_propulsion
        }
