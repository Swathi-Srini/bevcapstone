"""Depth estimation utility functions and coordinate transformations."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .camera_params import CameraParameters, CameraPosition


class DisparityToDepth:
    """
    Disparity to depth conversion with error analysis.
    Based on tech spec Section 3.2 and 3.1.
    """
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize disparity converter.
        
        Args:
            camera_params: Camera calibration parameters
        """
        self.camera_params = camera_params
    
    def disparity_to_depth(self, disparity_px: float | np.ndarray) -> float | np.ndarray:
        """
        Convert disparity to depth.
        
        From tech spec Equation 5:
        Z = f * B / d_px
        
        Args:
            disparity_px: Disparity in pixels
            
        Returns:
            Depth in metres
        """
        return (
            self.camera_params.fx * self.camera_params.baseline / disparity_px
        )
    
    def depth_to_disparity(self, depth_m: float | np.ndarray) -> float | np.ndarray:
        """
        Convert depth to disparity.
        
        Inverse of disparity_to_depth:
        d_px = f * B / Z
        
        Args:
            depth_m: Depth in metres
            
        Returns:
            Disparity in pixels
        """
        return (
            self.camera_params.fx * self.camera_params.baseline / depth_m
        )
    
    def depth_error_at_distance(self, distance_m: float) -> float:
        """
        Compute depth measurement error at given distance.
        
        From tech spec Section 3.1, Equation 3:
        σZ = Z^2 · σd / (f · B)
        
        Where σd ≈ 0.2 px (SGM accuracy)
        
        Args:
            distance_m: Distance in metres
            
        Returns:
            Depth error standard deviation in metres
        """
        return self.camera_params.get_depth_precision(distance_m)


class CoordinateTransform:
    """
    Coordinate system transformations for camera and ego frames.
    Based on tech spec Section 5.4 and 5.5.
    """
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize coordinate transformer.
        
        Args:
            camera_params: Camera calibration parameters
        """
        self.camera_params = camera_params
        
        # BEV grid parameters (from tech spec Section 5.1)
        self.bev_grid_size = 64  # pixels
        self.bev_lateral_range = 10.0  # metres
        self.bev_forward_range = 17.5  # metres
        self.bev_rear_range = 2.5  # metres
        self.bev_ego_row = 56
        self.bev_ego_col = 32
        self.metres_per_pixel = (
            2 * self.bev_lateral_range / self.bev_grid_size
        )
    
    def pixel_to_camera_3d(
        self,
        u: float | np.ndarray,
        v: float | np.ndarray,
        depth: float | np.ndarray
    ) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """
        Back-project pixel to 3D in camera frame.
        
        From tech spec Section 3.2, Equations 5-6:
        Z = f * B / d_px
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        
        Args:
            u: Pixel column(s)
            v: Pixel row(s)
            depth: Depth in metres
            
        Returns:
            (X, Y, Z) in camera frame (metres)
        """
        cp = self.camera_params
        
        X = (u - cp.cx) * depth / cp.fx
        Y = (v - cp.cy) * depth / cp.fy
        Z = depth
        
        return X, Y, Z
    
    def camera_to_ego_frame(
        self,
        X_cam: float | np.ndarray,
        Z_cam: float | np.ndarray,
        camera_yaw_deg: float
    ) -> Tuple[float | np.ndarray, float | np.ndarray]:
        """
        Transform camera coordinates to ego vehicle frame.
        
        Based on tech spec Section 5.5, Table 8.
        Camera yaw determines rotation to ego frame.
        
        Args:
            X_cam: Lateral coordinate in camera frame (metres)
            Z_cam: Forward coordinate in camera frame (metres)
            camera_yaw_deg: Camera yaw offset in degrees
            
        Returns:
            (X_ego, Z_ego) in ego frame (metres)
        """
        # Convert yaw to radians
        yaw_rad = np.radians(camera_yaw_deg)
        
        # Rotation matrix (2D)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # From tech spec Table 8 transformations:
        # For front camera (yaw=0): X_ego = X_cam, Z_ego = Z_cam
        # For left camera (yaw=-90): X_ego = -Z_cam, Z_ego = X_cam
        # For right camera (yaw=+90): X_ego = Z_cam, Z_ego = -X_cam
        # For rear camera (yaw=180): X_ego = -X_cam, Z_ego = -Z_cam
        
        X_ego = cos_yaw * X_cam - sin_yaw * Z_cam
        Z_ego = sin_yaw * X_cam + cos_yaw * Z_cam
        
        return X_ego, Z_ego
    
    def ego_to_bev_grid(
        self,
        X_ego: float | np.ndarray,
        Z_ego: float | np.ndarray
    ) -> Tuple[float | np.ndarray, float | np.ndarray]:
        """
        Transform ego frame coordinates to BEV grid coordinates.
        
        From tech spec Section 5.4, Equations 9-10:
        col = (X_ego + 10.0) / 20.0 * 64
        row = (1 - (Z_ego + 2.5) / 20.0) * 64
        
        Args:
            X_ego: Lateral coordinate in ego frame (metres)
            Z_ego: Forward coordinate in ego frame (metres)
            
        Returns:
            (col, row) in BEV grid coordinates
        """
        # From equations 9-10 in tech spec Section 5.4
        lateral_range = self.bev_lateral_range
        total_range = self.bev_forward_range + self.bev_rear_range
        
        col = ((X_ego + lateral_range) / (2 * lateral_range)) * self.bev_grid_size
        row = (1 - (Z_ego + self.bev_rear_range) / total_range) * self.bev_grid_size
        
        return col, row
    
    def camera_to_bev_grid(
        self,
        u: float | np.ndarray,
        v: float | np.ndarray,
        depth: float | np.ndarray,
        camera_yaw_deg: float
    ) -> Tuple[float | np.ndarray, float | np.ndarray]:
        """
        Full pipeline: pixel -> 3D camera -> ego frame -> BEV grid.
        
        Args:
            u: Pixel column(s)
            v: Pixel row(s)
            depth: Depth in metres
            camera_yaw_deg: Camera yaw in degrees
            
        Returns:
            (col, row) in BEV grid
        """
        # Step 1: Pixel to camera 3D
        X_cam, Y_cam, Z_cam = self.pixel_to_camera_3d(u, v, depth)
        
        # Step 2: Camera to ego frame
        X_ego, Z_ego = self.camera_to_ego_frame(X_cam, Z_cam, camera_yaw_deg)
        
        # Step 3: Ego frame to BEV grid
        col, row = self.ego_to_bev_grid(X_ego, Z_ego)
        
        return col, row


class PhysicalSizeEstimator:
    """
    Estimate physical object dimensions from bounding box and depth.
    Based on tech spec Section 5.7.
    """
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize size estimator.
        
        Args:
            camera_params: Camera calibration parameters
        """
        self.camera_params = camera_params
    
    def estimate_physical_size(
        self,
        bbox_width_px: float | np.ndarray,
        bbox_height_px: float | np.ndarray,
        depth: float | np.ndarray
    ) -> Tuple[float | np.ndarray, float | np.ndarray]:
        """
        Estimate physical object dimensions from bounding box.
        
        From tech spec Section 5.7, Equation 14:
        w_physical = w_bbox * Z / f
        l_physical = h_bbox * Z / f
        
        Args:
            bbox_width_px: Bounding box width in pixels
            bbox_height_px: Bounding box height in pixels
            depth: Depth in metres
            
        Returns:
            (width, length) in metres
        """
        w_physical = bbox_width_px * depth / self.camera_params.fx
        l_physical = bbox_height_px * depth / self.camera_params.fy
        
        return w_physical, l_physical
    
    def draw_physical_size_on_grid(
        self,
        center_col: float,
        center_row: float,
        width_m: float,
        length_m: float,
        metres_per_pixel: float = 0.3125
    ) -> np.ndarray:
        """
        Create a mask of physical object footprint on BEV grid.
        
        Args:
            center_col: Object center column in grid
            center_row: Object center row in grid
            width_m: Object width in metres
            length_m: Object length in metres
            metres_per_pixel: Grid scale (metres per pixel)
            
        Returns:
            Binary mask of object footprint
        """
        # Convert metres to pixels
        width_px = width_m / metres_per_pixel
        length_px = length_m / metres_per_pixel
        
        # Create rectangle
        # Note: In actual implementation, this would be drawn on the grid
        return {
            'center_col': center_col,
            'center_row': center_row,
            'width_px': width_px,
            'length_px': length_px,
            'area_px': width_px * length_px
        }


class MonocularDepth:
    """
    Monocular depth estimation via ground-plane projection.
    Based on tech spec Section 5.6.
    """
    
    def __init__(self, camera_params: CameraParameters):
        """
        Initialize monocular depth estimator.
        
        Args:
            camera_params: Camera calibration parameters
        """
        self.camera_params = camera_params
    
    def estimate_depth_from_ground_plane(
        self,
        v: float | np.ndarray,
        camera_pitch_deg: float,
        camera_height_m: float
    ) -> float | np.ndarray:
        """
        Estimate depth from ground plane projection.
        
        From tech spec Section 5.6, Equations 11-12:
        φ = θ_pitch + Δv * FOV_v / h_img
        Z = H / tan(φ)
        
        Args:
            v: Pixel row (vertical coordinate)
            camera_pitch_deg: Camera pitch in degrees
            camera_height_m: Camera height above ground in metres
            
        Returns:
            Estimated depth in metres
        """
        # FOV vertical (derived from horizontal FOV and aspect ratio)
        aspect_ratio = (
            self.camera_params.IMAGE_HEIGHT / self.camera_params.IMAGE_WIDTH
        )
        fov_v = self.camera_params.FOV_HORIZONTAL * aspect_ratio
        
        # Vertical pixel offset from image centre
        delta_v = v - self.camera_params.cy
        
        # Elevation angle
        pitch_rad = np.radians(camera_pitch_deg)
        fov_v_rad = np.radians(fov_v)
        
        phi = pitch_rad + delta_v * (fov_v_rad / self.camera_params.IMAGE_HEIGHT)
        
        # Depth from ground plane
        Z = camera_height_m / np.tan(phi)
        
        return Z
    
    def estimate_lateral_offset(
        self,
        u: float | np.ndarray,
        depth: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Estimate lateral offset from image column and depth.
        
        From tech spec Section 5.6, Equation 13:
        β = Δu * FOV_h / w_img
        X = Z * tan(β)
        
        Args:
            u: Pixel column
            depth: Estimated depth
            
        Returns:
            Lateral offset in metres
        """
        # Horizontal pixel offset from image centre
        delta_u = u - self.camera_params.cx
        
        # Horizontal angle
        fov_h_rad = np.radians(self.camera_params.FOV_HORIZONTAL)
        beta = delta_u * (fov_h_rad / self.camera_params.IMAGE_WIDTH)
        
        # Lateral offset
        X = depth * np.tan(beta)
        
        return X
