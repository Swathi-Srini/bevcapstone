"""Stereo matching using Semi-Global Matching (SGM) algorithm."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .camera_params import CameraParameters


class StereoMatcher:
    """
    Stereo depth estimation using OpenCV's StereoSGBM.
    
    Implements Semi-Global Matching algorithm with parameters from
    technical spec Section 3.3 (Table 5).
    """
    
    def __init__(self, camera_params: Optional[CameraParameters] = None):
        """
        Initialize stereo matcher with specified parameters.
        
        Args:
            camera_params: CameraParameters instance. If None, uses defaults.
        """
        self.camera_params = camera_params or CameraParameters()
        self._initialize_matcher()
        self.last_disparity = None
        self.last_depth = None
    
    def _initialize_matcher(self):
        """Initialize OpenCV StereoSGBM matcher with spec parameters."""
        # From tech spec Table 5
        # Resolve SGBM mode constant across OpenCV versions
        if hasattr(cv2, 'StereoSGBM') and hasattr(cv2.StereoSGBM, 'MODE_SGBM_3WAY'):
            sgbm_mode = cv2.StereoSGBM.MODE_SGBM_3WAY
        elif hasattr(cv2, 'STEREO_SGBM_MODE_SGBM_3WAY'):
            sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
        else:
            # Fallback numeric value for SGBM_3WAY (common value = 2)
            sgbm_mode = 2

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.camera_params.NUM_DISPARITIES,
            blockSize=self.camera_params.BLOCK_SIZE,
            P1=self.camera_params.P1,
            P2=self.camera_params.P2,
            disp12MaxDiff=self.camera_params.DISP12_MAX_DIFF,
            uniquenessRatio=self.camera_params.UNIQUENESS_RATIO,
            speckleWindowSize=50,
            speckleRange=2,
            mode=sgbm_mode
        )
        
        # Left matcher for filtering
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.camera_params.NUM_DISPARITIES,
            blockSize=self.camera_params.BLOCK_SIZE,
            P1=self.camera_params.P1,
            P2=self.camera_params.P2,
            disp12MaxDiff=self.camera_params.DISP12_MAX_DIFF,
            uniquenessRatio=self.camera_params.UNIQUENESS_RATIO,
            speckleWindowSize=50,
            speckleRange=2,
            mode=sgbm_mode
        )
        
        # Right matcher and WLS filter (optional, requires opencv-contrib-python)
        self.right_matcher = None
        self.wls_filter = None
        self.use_wls_filter = False
        
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createRightMatcher'):
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
                self.wls_filter.setLambda(80000)
                self.wls_filter.setSigmaColor(1.5)
                self.use_wls_filter = True
                print("[INFO] WLS filter enabled (opencv-contrib-python available)")
            else:
                print("[INFO] WLS filter disabled (opencv-contrib-python not available)")
        except Exception as e:
            print(f"[INFO] WLS filter unavailable: {e}")
            print("  To enable: pip install opencv-contrib-python")
    
    def compute_disparity(
        self, 
        left_img: np.ndarray,
        right_img: np.ndarray,
        use_filtering: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from stereo pair.
        
        Args:
            left_img: Left camera image (BGR or grayscale)
            right_img: Right camera image (BGR or grayscale)
            use_filtering: Whether to apply WLS filtering for better results
            
        Returns:
            Disparity map (H x W), values in pixels (can be fractional)
        """
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Compute left and right disparity (raw int16 scaled by 16)
        left_disp_raw = self.left_matcher.compute(left_gray, right_gray)
        
        # Right disparity computation depends on availability of right_matcher
        if self.right_matcher is not None:
            right_disp_raw = self.right_matcher.compute(right_gray, left_gray)
        else:
            right_disp_raw = None
        
        if use_filtering and self.use_wls_filter and self.wls_filter is not None and right_disp_raw is not None:
            try:
                # Apply Weighted Least Squares filtering
                # WLS filter expects disparities in the matcher output format (int16 scaled by 16)
                filtered_disp = self.wls_filter.filter(
                    left_disp_raw, left_gray, disparity_map_right=right_disp_raw
                )
                # Convert to float disparity in pixels (divide by 16)
                disparity = np.float32(filtered_disp) / 16.0
            except Exception as e:
                print(f"Warning: WLS filtering failed ({e}), using unfiltered disparity")
                disparity = np.float32(left_disp_raw) / 16.0
        else:
            disparity = np.float32(left_disp_raw) / 16.0
        
        self.last_disparity = disparity
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        From tech spec Section 3.2, Equation 5:
        Z = f * B / d_px
        
        Where:
        - f = focal length (pixels)
        - B = baseline (metres)
        - d_px = disparity (pixels)
        
        Args:
            disparity: Disparity map (pixels)
            
        Returns:
            Depth map (metres), same shape as input
        """
        # Avoid division by zero
        mask = disparity > 0
        depth = np.full_like(disparity, np.inf)
        
        # Z = f * B / d (tech spec eq. 5)
        depth[mask] = (
            self.camera_params.fx * self.camera_params.baseline / disparity[mask]
        )
        
        self.last_depth = depth
        return depth
    
    def compute_depth(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        use_filtering: bool = True
    ) -> np.ndarray:
        """
        Compute depth map directly from stereo pair.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            use_filtering: Whether to apply post-processing
            
        Returns:
            Depth map in metres
        """
        disparity = self.compute_disparity(left_img, right_img, use_filtering)
        depth = self.disparity_to_depth(disparity)
        return depth
    
    def get_point_cloud(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        use_filtering: bool = True
    ) -> np.ndarray:
        """
        Generate 3D point cloud from stereo pair.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            use_filtering: Whether to apply post-processing
            
        Returns:
            Point cloud (N x 3), where N = H * W, columns are [X, Y, Z]
        """
        depth = self.compute_depth(left_img, right_img, use_filtering)
        
        h, w = depth.shape
        
        # Create coordinate grids
        x_indices = np.arange(w, dtype=np.float32)
        y_indices = np.arange(h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices)
        
        # Back-project from pixels to 3D using intrinsic matrix
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = Z (from disparity)
        
        X = (x_grid - self.camera_params.cx) * depth / self.camera_params.fx
        Y = (y_grid - self.camera_params.cy) * depth / self.camera_params.fy
        Z = depth
        
        # Filter out invalid points (inf, -inf, nan)
        valid_mask = np.isfinite(Z)
        
        # Stack and extract valid points
        points = np.stack([X, Y, Z], axis=2)
        point_cloud = points[valid_mask]
        
        return point_cloud
    
    def get_disparity_visualization(self, disparity: np.ndarray) -> np.ndarray:
        """
        Create visualization of disparity map.
        
        Args:
            disparity: Disparity map
            
        Returns:
            Colored disparity visualization (uint8, BGR)
        """
        # Normalize to 0-255
        disp_vis = disparity.copy()
        valid_mask = disp_vis > 0
        
        if np.any(valid_mask):
            min_disp = np.min(disp_vis[valid_mask])
            max_disp = np.max(disp_vis[valid_mask])
            
            if max_disp > min_disp:
                disp_vis[valid_mask] = (
                    (disp_vis[valid_mask] - min_disp) / (max_disp - min_disp) * 255
                )
            else:
                disp_vis[valid_mask] = 128
        
        disp_vis = np.uint8(disp_vis)
        
        # Apply colormap
        colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        
        # Set invalid pixels to black
        colored[~valid_mask] = 0
        
        return colored
    
    def get_depth_visualization(self, depth: np.ndarray) -> np.ndarray:
        """
        Create visualization of depth map.
        
        Args:
            depth: Depth map in metres
            
        Returns:
            Colored depth visualization (uint8, BGR)
        """
        depth_vis = depth.copy()
        
        # Filter valid pixels
        valid_mask = np.isfinite(depth_vis)
        
        if np.any(valid_mask):
            min_depth = np.min(depth_vis[valid_mask])
            max_depth = np.max(depth_vis[valid_mask])
            # Cap display range to at most +30m from the minimum depth to improve contrast
            max_depth = min(max_depth, min_depth + 30.0)
            
            # Avoid division by zero if max_depth == min_depth
            if max_depth > min_depth:
                depth_vis[valid_mask] = (
                    (depth_vis[valid_mask] - min_depth) / (max_depth - min_depth) * 255
                )
            else:
                depth_vis[valid_mask] = 128
            depth_vis[depth_vis > 255] = 255
            depth_vis[depth_vis < 0] = 0
        
        depth_vis = np.uint8(depth_vis)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        
        # Set invalid pixels to black
        colored[~valid_mask] = 0
        
        return colored
    
    def print_summary(self):
        """Print matcher configuration summary."""
        print("=" * 60)
        print("STEREO MATCHER CONFIGURATION")
        print("=" * 60)
        print(f"Algorithm: Semi-Global Matching (SGM)")
        print(f"  numDisparities: {self.camera_params.NUM_DISPARITIES}")
        print(f"  blockSize: {self.camera_params.BLOCK_SIZE}")
        print(f"  P1: {self.camera_params.P1}")
        print(f"  P2: {self.camera_params.P2}")
        print(f"  disp12MaxDiff: {self.camera_params.DISP12_MAX_DIFF}")
        print(f"  uniquenessRatio: {self.camera_params.UNIQUENESS_RATIO}")
        print(f"  Mode: STEREO_SGBM_MODE_SGBM_3WAY")
        print(f"  Post-processing: WLS Filter (lambda=80000, sigma=1.5)")
        print("=" * 60)
