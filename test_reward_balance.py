"""Calculate expected rewards to understand agent behavior."""

import numpy as np

# Scenario: Agent moving at 10 m/s, perfectly on path
v = 10.0
d = 0.0
heading_error = 0.0
progress_made = v * 0.1  # 1m per step
curvature = 0.0
power = 100.0  # approximate
brake = 0.0
jerk = 0.0

# Compute reward components
r_progress = 50.0 * progress_made
r_speed = 0.7 * v
r_energy = -0.0005 * power
r_lateral = 0.0  # d=0
r_heading = 0.0  # aligned
r_speed_curve = 0.0  # straight
r_lat_acc = 0.0
r_brake = 0.0
r_jerk = 0.0

total = r_progress + r_speed + r_energy + r_lateral + r_heading + r_speed_curve + r_lat_acc + r_brake + r_jerk

print("Scenario: Moving perfectly at 10 m/s")
print(f"  Progress reward: +{r_progress:.2f}")
print(f"  Speed reward:    +{r_speed:.2f}")
print(f"  Energy penalty:  {r_energy:.2f}")
print(f"  Other penalties:  0.00")
print(f"  TOTAL:           {total:.2f}")
print()

# Scenario: Staying still
v_still = 0.0
progress_still = 0.0
r_progress_still = 50.0 * progress_still
r_speed_still = 0.7 * v_still
r_stay_still = -1.0  # penalty

total_still = r_progress_still + r_speed_still + r_stay_still

print("Scenario: Staying completely still")
print(f"  Progress reward: +{r_progress_still:.2f}")
print(f"  Speed reward:    +{r_speed_still:.2f}")
print(f"  Stay-still penalty: -1.00")  
print(f"  TOTAL PER STEP:  {total_still:.2f}")
print(f"  TOTAL 1000 STEPS: {total_still * 1000:.2f}")
print()

print("=" * 60)
print("Comparison:")
print(f"  Moving:  +{total:.2f} per step")
print(f"  Still:   {total_still:.2f} per step")
print(f"  Difference: {total - total_still:.2f}")
print(f"  Moving should be WAY better!")
