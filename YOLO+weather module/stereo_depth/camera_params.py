"""Camera calibration and intrinsic parameters for stereo depth estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class CameraPosition(Enum):
    """Camera mounting positions on ego vehicle."""
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    REAR = "rear"


@dataclass
class CameraConfig:
    """Configuration for a single camera mounting."""
    
    position: CameraPosition
    x_offset: float  # Forward offset in metres
    z_offset: float  # Height above ground in metres
    yaw: float      # Yaw angle in degrees
    pitch: float    # Pitch angle in degrees (downward)
    role: str       # Camera role (e.g., "stereo_left", "monocular_side")
    
    def __post_init__(self):
        """Validate camera configuration."""
        if not isinstance(self.position, CameraPosition):
            self.position = CameraPosition(self.position.lower())


class CameraParameters:
    """
    Camera intrinsic and extrinsic parameters.
    
    Based on technical spec Section 2 and 3:
    - All cameras share identical intrinsic parameters
    - Stereo baseline B = 0.5 m (left and right cameras)
    - Focal length derived from FOV and resolution
    """
    
    # Intrinsic parameters (from tech spec Table 3)
    IMAGE_WIDTH = 1200  # pixels
    IMAGE_HEIGHT = 900  # pixels
    FOV_HORIZONTAL = 60  # degrees
    FOCAL_LENGTH = 1000  # pixels
    HEIGHT_ABOVE_GROUND = 1.4  # metres
    DOWNWARD_PITCH = -5  # degrees
    
    # Stereo baseline (from tech spec Section 2.3)
    STEREO_BASELINE = 0.5  # metres
    
    # Stereo SGM parameters (from tech spec Table 5)
    NUM_DISPARITIES = 192  # divisible by 16
    BLOCK_SIZE = 5
    P1 = 600  # 8 * 3 * 5^2
    P2 = 2400  # 32 * 3 * 5^2
    DISP12_MAX_DIFF = 1
    UNIQUENESS_RATIO = 10
    DEPTH_RANGE = (1.0, 30.0)  # metres
    
    # Camera mounting positions (from tech spec Table 2)
    CAMERA_CONFIGS = {
        CameraPosition.FRONT: CameraConfig(
            position=CameraPosition.FRONT,
            x_offset=2.0,
            z_offset=1.4,
            yaw=0,
            pitch=-5,
            role="stereo_left_primary"
        ),
        CameraPosition.LEFT: CameraConfig(
            position=CameraPosition.LEFT,
            x_offset=0.0,
            z_offset=1.4,
            yaw=-90,
            pitch=-5,
            role="monocular_side"
        ),
        CameraPosition.RIGHT: CameraConfig(
            position=CameraPosition.RIGHT,
            x_offset=0.0,
            z_offset=1.4,
            yaw=90,
            pitch=-5,
            role="monocular_side"
        ),
        CameraPosition.REAR: CameraConfig(
            position=CameraPosition.REAR,
            x_offset=-2.0,
            z_offset=1.4,
            yaw=180,
            pitch=-5,
            role="monocular_rear"
        ),
    }
    
    # Stereo camera right offset (0.25 m either side of centreline)
    STEREO_RIGHT_OFFSET = -0.25  # metres (negative = right side)
    
    def __init__(self):
        """Initialize camera parameters and compute derived quantities."""
        self.fx = self.FOCAL_LENGTH
        self.fy = self.FOCAL_LENGTH
        self.cx = self.IMAGE_WIDTH / 2.0
        self.cy = self.IMAGE_HEIGHT / 2.0
        
        # Intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Baseline for stereo pair (0.5 m between front cameras)
        self.baseline = self.STEREO_BASELINE
        
        # Depth precision at reference distance (30 m)
        # σZ = Z^2 * σd / (f * B)
        # σd ≈ 0.2 px with SGM
        self.disparity_error_px = 0.2
        self._compute_depth_precision()
    
    def _compute_depth_precision(self):
        """Compute stereo depth error as function of distance."""
        self.depth_error_coeffs = {
            10: self._compute_depth_error_at_distance(10),
            20: self._compute_depth_error_at_distance(20),
            30: self._compute_depth_error_at_distance(30),
        }
    
    def _compute_depth_error_at_distance(self, distance_m: float) -> float:
        """
        Compute depth error at given distance.
        
        From tech spec Section 3.1, Equation 3:
        σZ = Z^2 · σd / (f · B)
        
        Args:
            distance_m: Distance in metres
            
        Returns:
            Depth error in metres
        """
        sigma_z = (distance_m ** 2) * self.disparity_error_px / (
            self.fx * self.baseline
        )
        return sigma_z
    
    def get_depth_precision(self, distance_m: float) -> float:
        """
        Get depth precision at given distance.
        
        Args:
            distance_m: Distance in metres
            
        Returns:
            Depth error standard deviation in metres
        """
        return self._compute_depth_error_at_distance(distance_m)
    
    def get_camera_config(self, position: CameraPosition) -> CameraConfig:
        """Get configuration for specific camera position."""
        return self.CAMERA_CONFIGS[position]
    
    def get_all_camera_configs(self) -> dict[CameraPosition, CameraConfig]:
        """Get all camera configurations."""
        return self.CAMERA_CONFIGS.copy()
    
    def get_stereo_pair_cameras(self) -> Tuple[CameraConfig, CameraConfig]:
        """
        Get stereo pair camera configurations (front cameras).
        
        Returns:
            Tuple of (left_camera, right_camera) configurations
        """
        left_config = self.CAMERA_CONFIGS[CameraPosition.FRONT]
        
        # Right camera has same config but offset
        right_config = CameraConfig(
            position=CameraPosition.FRONT,
            x_offset=left_config.x_offset,
            z_offset=left_config.z_offset,
            yaw=left_config.yaw,
            pitch=left_config.pitch,
            role="stereo_right"
        )
        
        return left_config, right_config
    
    def fov_to_focal_length(self) -> float:
        """
        Verify focal length from FOV.
        
        From tech spec Equation 1:
        f = (w/2) / tan(FOVh/2)
        
        Returns:
            Focal length in pixels
        """
        fov_rad = np.radians(self.FOV_HORIZONTAL)
        f = (self.IMAGE_WIDTH / 2.0) / np.tan(fov_rad / 2.0)
        return f
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix K."""
        return self.K.copy()
    
    def get_stereo_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get stereo pair intrinsic matrices.
        
        Returns:
            Tuple of (K_left, K_right)
        """
        K_left = self.K.copy()
        
        # Right camera has same intrinsics but principal point offset
        K_right = self.K.copy()
        # Note: In practice, disparity adjustment is handled during rectification
        
        return K_left, K_right
    
    def print_summary(self):
        """Print camera parameters summary."""
        print("=" * 60)
        print("CAMERA PARAMETERS SUMMARY")
        print("=" * 60)
        print(f"\nINTRINSIC PARAMETERS:")
        print(f"  Image Resolution: {self.IMAGE_WIDTH} x {self.IMAGE_HEIGHT} px")
        print(f"  Focal Length: {self.FOCAL_LENGTH} px")
        print(f"  Principal Point: ({self.cx:.1f}, {self.cy:.1f}) px")
        print(f"  FOV (horizontal): {self.FOV_HORIZONTAL}°")
        print(f"  Camera Height: {self.HEIGHT_ABOVE_GROUND} m")
        print(f"  Downward Pitch: {self.DOWNWARD_PITCH}°")
        
        print(f"\nSTEREO PARAMETERS:")
        print(f"  Baseline: {self.baseline} m")
        print(f"  Disparity Range: 1-192 pixels")
        
        print(f"\nDEPTH PRECISION:")
        for dist, error in self.depth_error_coeffs.items():
            rel_error = (error / dist) * 100
            print(f"  @ {dist}m: ±{error:.3f}m ({rel_error:.1f}%)")
        
        print(f"\nCAMERA CONFIGURATIONS:")
        for pos, config in self.CAMERA_CONFIGS.items():
            print(f"  {pos.value.upper()}: offset=({config.x_offset}m, {config.z_offset}m), "
                  f"yaw={config.yaw}°, pitch={config.pitch}°, role={config.role}")
        print("=" * 60)
