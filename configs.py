"""
Configuration presets for MinimalGridTownEnv.

This file contains several pre-configured environment setups
for different research scenarios and use cases.
"""

from minimal_grid_town_env import MinimalGridTownEnv


# ============================================================================
# BASIC CONFIGURATIONS
# ============================================================================

def get_default_config():
    """Default balanced configuration."""
    return {
        'grid_size': (6, 6),
        'block_spacing': 30.0,
        'dt': 0.1,
        'max_steps': 1000,
        'render_mode': None
    }


def get_small_test_config():
    """Small environment for quick testing and debugging."""
    return {
        'grid_size': (4, 4),
        'block_spacing': 20.0,
        'dt': 0.1,
        'max_steps': 500,
        'render_mode': None
    }


def get_large_city_config():
    """Large city environment for complex scenarios."""
    return {
        'grid_size': (10, 10),
        'block_spacing': 50.0,
        'dt': 0.1,
        'max_steps': 3000,
        'render_mode': None
    }


# ============================================================================
# REWARD-TUNED CONFIGURATIONS
# ============================================================================

def get_speed_optimized_config():
    """Configuration emphasizing speed over energy efficiency."""
    config = get_default_config()
    config.update({
        'alpha': 2.0,           # High speed reward
        'lambda_E': 0.001,      # Very low energy penalty
        'lambda_d': 1.0,        # Moderate lane-keeping
        'lambda_lat': 0.3,      # Low lateral acceleration penalty
        'lambda_brake': 0.5,    # Low brake penalty
        'lambda_jerk': 0.05     # Low jerk penalty
    })
    return config


def get_energy_optimized_config():
    """Configuration emphasizing energy efficiency over speed."""
    config = get_default_config()
    config.update({
        'alpha': 0.5,           # Low speed reward
        'lambda_E': 0.05,       # High energy penalty
        'lambda_d': 2.0,        # High lane-keeping
        'lambda_lat': 0.5,      # Moderate lateral acceleration
        'lambda_brake': 0.5,    # Moderate brake penalty
        'lambda_jerk': 0.2      # High jerk penalty (smooth driving)
    })
    return config


def get_comfort_optimized_config():
    """Configuration emphasizing passenger comfort (low jerk, low lateral acc)."""
    config = get_default_config()
    config.update({
        'alpha': 0.8,           # Moderate speed reward
        'lambda_E': 0.01,       # Moderate energy penalty
        'lambda_d': 2.0,        # High lane-keeping
        'lambda_lat': 1.0,      # High lateral acceleration penalty
        'lambda_brake': 2.0,    # High brake penalty
        'lambda_jerk': 0.5      # Very high jerk penalty
    })
    return config


def get_strict_lane_keeping_config():
    """Configuration with strict lane-keeping requirements."""
    config = get_default_config()
    config.update({
        'lane_width': 2.5,      # Narrower lanes
        'alpha': 1.0,
        'lambda_E': 0.01,
        'lambda_d': 5.0,        # Very high lateral deviation penalty
        'lambda_lat': 0.5,
        'lambda_brake': 1.0,
        'lambda_jerk': 0.1
    })
    return config


# ============================================================================
# VEHICLE-SPECIFIC CONFIGURATIONS
# ============================================================================

def get_sports_car_config():
    """Configuration for a sports car (lighter, more agile)."""
    config = get_default_config()
    config.update({
        'mass': 1200.0,             # Lighter
        'wheelbase': 2.5,           # Shorter wheelbase
        'max_steering': 0.7,        # More agile steering
        'k_throttle': 4.0,          # More powerful acceleration
        'k_brake': 6.0,             # Better braking
        'max_velocity': 30.0,       # Higher top speed
        'drag_coefficient': 0.25,   # More aerodynamic
        'frontal_area': 2.0         # Smaller
    })
    return config


def get_suv_config():
    """Configuration for an SUV (heavier, less agile)."""
    config = get_default_config()
    config.update({
        'mass': 2000.0,             # Heavier
        'wheelbase': 3.0,           # Longer wheelbase
        'max_steering': 0.5,        # Less agile steering
        'k_throttle': 2.5,          # Less powerful acceleration
        'k_brake': 4.5,             # Moderate braking
        'max_velocity': 20.0,       # Lower top speed
        'drag_coefficient': 0.35,   # Less aerodynamic
        'frontal_area': 3.0         # Larger
    })
    return config


def get_electric_vehicle_config():
    """Configuration for an electric vehicle."""
    config = get_default_config()
    config.update({
        'mass': 1800.0,             # Heavy (battery)
        'k_throttle': 4.5,          # Instant torque
        'k_brake': 7.0,             # Regenerative braking
        'drag_coefficient': 0.24,   # Very aerodynamic
        'rolling_resistance': 0.008, # Low rolling resistance
        'lambda_E': 0.02            # Higher energy penalty (range anxiety)
    })
    return config


# ============================================================================
# SCENARIO-SPECIFIC CONFIGURATIONS
# ============================================================================

def get_training_config():
    """Configuration optimized for RL training."""
    config = get_default_config()
    config.update({
        'grid_size': (6, 6),
        'block_spacing': 30.0,
        'dt': 0.1,
        'max_steps': 1000,
        'render_mode': None,        # No rendering during training
        # Balanced rewards for learning
        'alpha': 1.0,
        'lambda_E': 0.01,
        'lambda_d': 2.0,
        'lambda_lat': 0.5,
        'lambda_brake': 1.0,
        'lambda_jerk': 0.1
    })
    return config


def get_evaluation_config():
    """Configuration for evaluating trained agents."""
    config = get_training_config()
    config.update({
        'render_mode': 'human',     # Enable visualization
        'max_steps': 2000           # Longer episodes
    })
    return config


def get_short_route_config():
    """Configuration with a short route for quick episodes."""
    config = get_default_config()
    config.update({
        'grid_size': (4, 4),
        'start_node': (0, 0),
        'goal_node': (1, 3),        # Nearby goal
        'max_steps': 300
    })
    return config


def get_long_route_config():
    """Configuration with a long route for extended episodes."""
    config = get_default_config()
    config.update({
        'grid_size': (8, 8),
        'block_spacing': 40.0,
        'start_node': (0, 0),
        'goal_node': (7, 7),        # Far goal
        'max_steps': 2500
    })
    return config


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def create_environment(config_name: str = 'default'):
    """
    Create an environment from a predefined configuration.
    
    Args:
        config_name: Name of the configuration preset
        
    Returns:
        Configured MinimalGridTownEnv instance
        
    Available configs:
        - 'default': Balanced default configuration
        - 'small_test': Small environment for testing
        - 'large_city': Large city environment
        - 'speed_optimized': Optimized for speed
        - 'energy_optimized': Optimized for energy efficiency
        - 'comfort_optimized': Optimized for passenger comfort
        - 'strict_lane_keeping': Strict lane-keeping requirements
        - 'sports_car': Sports car parameters
        - 'suv': SUV parameters
        - 'electric_vehicle': Electric vehicle parameters
        - 'training': Optimized for RL training
        - 'evaluation': For evaluating trained agents
        - 'short_route': Short route for quick episodes
        - 'long_route': Long route for extended episodes
    """
    configs = {
        'default': get_default_config,
        'small_test': get_small_test_config,
        'large_city': get_large_city_config,
        'speed_optimized': get_speed_optimized_config,
        'energy_optimized': get_energy_optimized_config,
        'comfort_optimized': get_comfort_optimized_config,
        'strict_lane_keeping': get_strict_lane_keeping_config,
        'sports_car': get_sports_car_config,
        'suv': get_suv_config,
        'electric_vehicle': get_electric_vehicle_config,
        'training': get_training_config,
        'evaluation': get_evaluation_config,
        'short_route': get_short_route_config,
        'long_route': get_long_route_config
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    config = configs[config_name]()
    return MinimalGridTownEnv(**config)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("Available environment configurations:")
    print("-" * 50)
    
    configs = [
        ('default', 'Balanced default configuration'),
        ('small_test', 'Small environment for testing'),
        ('large_city', 'Large city environment'),
        ('speed_optimized', 'Optimized for speed'),
        ('energy_optimized', 'Optimized for energy efficiency'),
        ('comfort_optimized', 'Optimized for passenger comfort'),
        ('strict_lane_keeping', 'Strict lane-keeping'),
        ('sports_car', 'Sports car parameters'),
        ('suv', 'SUV parameters'),
        ('electric_vehicle', 'Electric vehicle parameters'),
        ('training', 'Optimized for RL training'),
        ('evaluation', 'For evaluating trained agents'),
        ('short_route', 'Short route'),
        ('long_route', 'Long route')
    ]
    
    for name, description in configs:
        print(f"  {name:25s} - {description}")
    
    print("\nExample usage:")
    print("  from configs import create_environment")
    print("  env = create_environment('speed_optimized')")
    print("  obs, info = env.reset()")
