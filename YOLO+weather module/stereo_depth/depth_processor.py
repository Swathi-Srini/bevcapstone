"""High-level depth processing pipeline for autonomous driving perception."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera_params import CameraParameters, CameraPosition
from .stereo_matcher import StereoMatcher
from .depth_utils import (
    CoordinateTransform,
    DisparityToDepth,
    MonocularDepth,
    PhysicalSizeEstimator,
)


class DepthProcessor:
    """
    High-level depth processing pipeline.
    
    Coordinates stereo depth estimation, monocular depth, YOLO integration,
    and BEV grid construction.
    """
    
    def __init__(self, camera_params: Optional[CameraParameters] = None):
        """
        Initialize depth processor.
        
        Args:
            camera_params: Camera calibration. If None, uses defaults.
        """
        self.camera_params = camera_params or CameraParameters()
        
        # Initialize component modules
        self.stereo_matcher = StereoMatcher(self.camera_params)
        self.disparity_converter = DisparityToDepth(self.camera_params)
        self.coordinate_transform = CoordinateTransform(self.camera_params)
        self.size_estimator = PhysicalSizeEstimator(self.camera_params)
        self.monocular_depth = MonocularDepth(self.camera_params)
        
        # BEV grid (from tech spec Section 5.1)
        self.bev_grid_size = 64
        self.bev_grid = self._initialize_bev_grid()
        
        # Cache for depth maps and point clouds
        self.last_stereo_depth = None
        self.last_point_cloud = None
        self.last_yolo_detections = None
    
    def _initialize_bev_grid(self) -> np.ndarray:
        """Initialize empty BEV grid."""
        return np.zeros((self.bev_grid_size, self.bev_grid_size), dtype=np.float32)
    
    def process_stereo_pair(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        compute_point_cloud: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Process stereo image pair for depth estimation.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            compute_point_cloud: Whether to generate 3D point cloud
            
        Returns:
            Dictionary with 'depth', 'disparity', optionally 'point_cloud'
        """
        result = {}
        
        # Compute disparity map
        disparity = self.stereo_matcher.compute_disparity(left_img, right_img)
        result['disparity'] = disparity
        
        # Convert to depth
        depth = self.stereo_matcher.disparity_to_depth(disparity)
        result['depth'] = depth
        self.last_stereo_depth = depth
        
        # Optionally generate point cloud
        if compute_point_cloud:
            point_cloud = self.stereo_matcher.get_point_cloud(
                left_img, right_img, use_filtering=True
            )
            result['point_cloud'] = point_cloud
            self.last_point_cloud = point_cloud
        
        return result
    
    def process_yolo_detection(
        self,
        detection: Dict,
        depth_map: np.ndarray,
        camera_position: CameraPosition
    ) -> Dict:
        """
        Process YOLO detection with depth information.
        
        Args:
            detection: YOLO detection dict with keys:
                - 'bbox': [x1, y1, x2, y2] bounding box
                - 'confidence': confidence score
                - 'class_id': object class
            depth_map: Depth map from stereo
            camera_position: Which camera this detection is from
            
        Returns:
            Enhanced detection with 3D information
        """
        enhanced = detection.copy()
        
        x1, y1, x2, y2 = detection['bbox']
        
        # Get camera config
        camera_config = self.camera_params.get_camera_config(camera_position)
        
        # Use bottom-center of bounding box for ground plane projection
        bbox_center_u = (x1 + x2) / 2.0
        bbox_bottom_v = y2  # Bottom of bounding box
        
        # Get depth at detection
        u_int = int(np.clip(bbox_center_u, 0, depth_map.shape[1] - 1))
        v_int = int(np.clip(bbox_bottom_v, 0, depth_map.shape[0] - 1))
        
        depth_at_detection = depth_map[v_int, u_int]
        enhanced['depth'] = depth_at_detection
        
        # Convert detection to camera 3D
        if camera_position == CameraPosition.FRONT:
            # Use stereo depth directly
            X_cam, Y_cam, Z_cam = self.coordinate_transform.pixel_to_camera_3d(
                bbox_center_u, bbox_bottom_v, depth_at_detection
            )
        else:
            # Use monocular depth for side/rear cameras
            Z_cam = self.monocular_depth.estimate_depth_from_ground_plane(
                bbox_bottom_v,
                camera_config.pitch,
                camera_config.z_offset
            )
            X_cam = self.monocular_depth.estimate_lateral_offset(
                bbox_center_u, Z_cam
            )
            Y_cam = 0.0  # Ground plane
        
        # Transform to ego frame
        X_ego, Z_ego = self.coordinate_transform.camera_to_ego_frame(
            X_cam, Z_cam, camera_config.yaw
        )
        enhanced['ego_position'] = {'X': X_ego, 'Z': Z_ego}
        
        # Transform to BEV grid
        col_grid, row_grid = self.coordinate_transform.ego_to_bev_grid(
            X_ego, Z_ego
        )
        enhanced['bev_position'] = {'col': col_grid, 'row': row_grid}
        
        # Estimate physical dimensions
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        width_m, length_m = self.size_estimator.estimate_physical_size(
            bbox_width, bbox_height, depth_at_detection
        )
        enhanced['physical_size'] = {'width_m': width_m, 'length_m': length_m}
        
        return enhanced
    
    def process_detections_to_bev(
        self,
        detections_by_camera: Dict[CameraPosition, List[Dict]],
        depth_maps: Dict[CameraPosition, np.ndarray]
    ) -> np.ndarray:
        """
        Integrate multi-camera YOLO detections into BEV grid.
        
        Args:
            detections_by_camera: Detections per camera
            depth_maps: Depth maps per camera
            
        Returns:
            BEV grid with obstacles marked
        """
        bev_grid = self._initialize_bev_grid()
        
        # Process each camera's detections
        for camera_pos, detections in detections_by_camera.items():
            if camera_pos not in depth_maps:
                continue
            
            depth_map = depth_maps[camera_pos]
            
            for detection in detections:
                # Enhance detection with depth
                enhanced = self.process_yolo_detection(
                    detection, depth_map, camera_pos
                )
                
                # Draw on BEV grid
                bev_position = enhanced['bev_position']
                col = bev_position['col']
                row = bev_position['row']
                
                # Get physical size for drawing
                phys_size = enhanced['physical_size']
                width_px = phys_size['width_m'] / self.coordinate_transform.metres_per_pixel
                length_px = phys_size['length_m'] / self.coordinate_transform.metres_per_pixel
                
                # Draw rectangle on BEV grid (1.0 = occupied)
                self._draw_rectangle_on_grid(
                    bev_grid, col, row, width_px, length_px, value=1.0
                )
        
        self.bev_grid = bev_grid
        return bev_grid
    
    def _draw_rectangle_on_grid(
        self,
        grid: np.ndarray,
        center_col: float,
        center_row: float,
        width_px: float,
        length_px: float,
        value: float = 1.0
    ) -> None:
        """
        Draw rectangle on BEV grid (in-place).
        
        Args:
            grid: BEV grid to draw on
            center_col: Center column
            center_row: Center row
            width_px: Width in pixels
            length_px: Length in pixels
            value: Grid value to set
        """
        half_width = width_px / 2.0
        half_length = length_px / 2.0
        
        col_min = int(np.ceil(center_col - half_width))
        col_max = int(np.floor(center_col + half_width))
        row_min = int(np.ceil(center_row - half_length))
        row_max = int(np.floor(center_row + half_length))
        
        # Clamp to grid bounds
        col_min = max(0, col_min)
        col_max = min(grid.shape[1] - 1, col_max)
        row_min = max(0, row_min)
        row_max = min(grid.shape[0] - 1, row_max)
        
        if col_min <= col_max and row_min <= row_max:
            grid[row_min:row_max+1, col_min:col_max+1] = value
    
    def depth_to_visualization(
        self,
        depth: np.ndarray,
        method: str = 'turbo'
    ) -> np.ndarray:
        """
        Convert depth map to colored visualization.
        
        Args:
            depth: Depth map in metres
            method: Colormap ('turbo', 'jet', 'viridis')
            
        Returns:
            Colored depth visualization (uint8 BGR)
        """
        return self.stereo_matcher.get_depth_visualization(depth)
    
    def disparity_to_visualization(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to colored visualization.
        
        Args:
            disparity: Disparity map in pixels
            
        Returns:
            Colored disparity visualization (uint8 BGR)
        """
        return self.stereo_matcher.get_disparity_visualization(disparity)
    
    def bev_to_visualization(self) -> np.ndarray:
        """
        Create colored visualization of BEV grid.
        
        Returns:
            Colored BEV visualization (uint8 BGR)
        """
        # Normalize BEV grid to 0-255
        bev_vis = (self.bev_grid * 255).astype(np.uint8)
        
        # Convert to BGR (grayscale -> 3 channel)
        bev_colored = cv2.cvtColor(bev_vis, cv2.COLOR_GRAY2BGR)
        
        # Highlight obstacles in red
        red_channel = bev_colored[:, :, 2]
        red_channel[self.bev_grid > 0.5] = 255
        
        return bev_colored
    
    def print_calibration_summary(self):
        """Print full calibration and configuration summary."""
        print("\n" + "=" * 70)
        print("DEPTH PROCESSING PIPELINE - FULL CALIBRATION SUMMARY")
        print("=" * 70 + "\n")
        
        self.camera_params.print_summary()
        print()
        self.stereo_matcher.print_summary()
        
        print("\nBEV GRID CONFIGURATION:")
        print(f"  Grid Size: {self.bev_grid_size} x {self.bev_grid_size}")
        print(f"  Lateral Range: ±{self.coordinate_transform.bev_lateral_range} m")
        print(f"  Forward Range: {self.coordinate_transform.bev_forward_range} m ahead")
        print(f"  Rear Range: -{self.coordinate_transform.bev_rear_range} m")
        print(f"  Metres per Pixel: {self.coordinate_transform.metres_per_pixel:.4f}")
        print(f"  Ego Position: row={self.coordinate_transform.bev_ego_row}, "
              f"col={self.coordinate_transform.bev_ego_col}")
        
        print("\nBEV GRID VALUES:")
        print("  0.0 = Free drivable space")
        print("  0.5 = Route centreline (HD map)")
        print("  0.8 = Road boundary (HD map)")
        print("  1.0 = Physically occupied (ego or obstacle)")
        
        print("=" * 70 + "\n")
