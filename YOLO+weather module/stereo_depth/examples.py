"""
Example usage of stereo depth estimation with YOLO detections.

This module demonstrates how to integrate stereo depth estimation with YOLO
object detection to create Bird's Eye View (BEV) representations for autonomous
driving perception.

Based on technical specification Section 5 (BEV Grid Construction).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from stereo_depth import (
    CameraParameters,
    CameraPosition,
    DepthProcessor,
    StereoMatcher,
)


class BEVPerceptionPipeline:
    """
    End-to-end BEV perception pipeline combining stereo depth and YOLO.
    """
    
    def __init__(self, yolo_model_path: Optional[str] = None):
        """
        Initialize perception pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model weights (e.g., 'yolov8n.pt')
        """
        # Initialize depth processing
        self.depth_processor = DepthProcessor()
        self.camera_params = self.depth_processor.camera_params
        
        # Initialize YOLO (requires ultralytics)
        self.yolo_model = None
        if yolo_model_path:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_model_path)
            except ImportError:
                print("Warning: ultralytics not installed. YOLO detection will be skipped.")
        
        # Detection confidence threshold (from tech spec)
        self.confidence_threshold = 0.4
        
        # Obstacle class IDs (COCO classes from tech spec)
        # Cars=2, Trucks=7, Buses=5, Motorcycles=3, Pedestrians=0
        self.obstacle_classes = {0, 2, 3, 5, 7}
    
    def process_stereo_images(
        self,
        left_image_path: str,
        right_image_path: str
    ) -> Dict:
        """
        Process stereo image pair.
        
        Args:
            left_image_path: Path to left camera image
            right_image_path: Path to right camera image
            
        Returns:
            Dictionary with depth maps and visualizations
        """
        # Load images
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)
        
        if left_img is None or right_img is None:
            raise FileNotFoundError("Could not load stereo images")
        
        # Process stereo pair
        result = self.depth_processor.process_stereo_pair(
            left_img, right_img, compute_point_cloud=True
        )
        
        # Generate visualizations
        depth_vis = self.depth_processor.depth_to_visualization(result['depth'])
        disp_vis = self.depth_processor.disparity_to_visualization(result['disparity'])
        
        result['depth_visualization'] = depth_vis
        result['disparity_visualization'] = disp_vis
        
        return result
    
    def detect_yolo_obstacles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect obstacles using YOLO.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections with keys:
            - 'bbox': [x1, y1, x2, y2]
            - 'confidence': confidence score
            - 'class_id': object class
            - 'class_name': object class name
        """
        if self.yolo_model is None:
            return []
        
        # Run YOLO inference
        results = self.yolo_model(image, conf=self.confidence_threshold)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls.item())
            
            # Filter to obstacle classes only
            if class_id not in self.obstacle_classes:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf.item())
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': results.names.get(class_id, f'class_{class_id}')
            })
        
        return detections
    
    def build_bev_grid_from_stereo(
        self,
        left_image_path: str,
        right_image_path: str
    ) -> np.ndarray:
        """
        Build complete BEV grid from stereo images with YOLO detections.
        
        Args:
            left_image_path: Path to left camera image
            right_image_path: Path to right camera image
            
        Returns:
            BEV grid as 64x64 numpy array
        """
        # Process stereo depth
        stereo_result = self.process_stereo_images(left_image_path, right_image_path)
        
        # Load images for YOLO
        left_img = cv2.imread(left_image_path)
        
        # Detect obstacles with YOLO
        detections = self.detect_yolo_obstacles(left_img)
        
        # Create depth maps dict for each camera
        depth_maps = {
            CameraPosition.FRONT: stereo_result['depth']
        }
        
        # Create detections dict by camera
        detections_by_camera = {
            CameraPosition.FRONT: detections
        }
        
        # Build BEV grid
        bev_grid = self.depth_processor.process_detections_to_bev(
            detections_by_camera, depth_maps
        )
        
        return bev_grid
    
    def save_results(
        self,
        output_dir: str,
        depth_vis: np.ndarray,
        disp_vis: np.ndarray,
        bev_vis: np.ndarray,
        bev_grid: np.ndarray
    ) -> None:
        """
        Save processing results to disk.
        
        Args:
            output_dir: Output directory path
            depth_vis: Depth visualization
            disp_vis: Disparity visualization
            bev_vis: BEV grid visualization
            bev_grid: BEV grid numpy array
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save visualizations
        cv2.imwrite(
            str(output_path / 'depth_map.jpg'),
            depth_vis
        )
        cv2.imwrite(
            str(output_path / 'disparity_map.jpg'),
            disp_vis
        )
        cv2.imwrite(
            str(output_path / 'bev_grid.jpg'),
            bev_vis
        )
        
        # Save BEV grid as numpy file
        np.save(
            str(output_path / 'bev_grid.npy'),
            bev_grid
        )
        
        print(f"Results saved to {output_dir}")


# ============================================================================
# Example Usage
# ============================================================================

def example_depth_estimation():
    """Example: Compute depth from stereo pair."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Stereo Depth Estimation")
    print("=" * 70)
    
    # Initialize processor
    processor = DepthProcessor()
    processor.print_calibration_summary()
    
    # Create dummy images for demonstration
    left_img = np.random.randint(0, 256, (900, 1200, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 256, (900, 1200, 3), dtype=np.uint8)
    
    # Process stereo pair
    result = processor.process_stereo_pair(left_img, right_img)
    
    print(f"\nDepth map shape: {result['depth'].shape}")
    print(f"Depth range: {np.nanmin(result['depth']):.2f} to {np.nanmax(result['depth']):.2f} m")
    print(f"Valid depth pixels: {np.isfinite(result['depth']).sum()} / {result['depth'].size}")


def example_yolo_integration():
    """Example: Integrate YOLO detections with depth."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: YOLO Detection + Depth Integration")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = BEVPerceptionPipeline()
    
    # Create dummy detection
    dummy_detection = {
        'bbox': [100, 150, 300, 400],  # [x1, y1, x2, y2]
        'confidence': 0.85,
        'class_id': 2,  # car
        'class_name': 'car'
    }
    
    # Create dummy depth map
    depth_map = np.ones((900, 1200), dtype=np.float32) * 20.0  # All 20m away
    
    # Process detection with depth
    processor = pipeline.depth_processor
    enhanced = processor.process_yolo_detection(
        dummy_detection,
        depth_map,
        CameraPosition.FRONT
    )
    
    print(f"Original detection: {dummy_detection['bbox']}")
    print(f"Depth at detection: {enhanced['depth']:.2f} m")
    print(f"BEV position: col={enhanced['bev_position']['col']:.1f}, "
          f"row={enhanced['bev_position']['row']:.1f}")
    print(f"Physical size: {enhanced['physical_size']['width_m']:.2f}m x "
          f"{enhanced['physical_size']['length_m']:.2f}m")


def example_bev_grid_construction():
    """Example: Build BEV grid from detections."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: BEV Grid Construction")
    print("=" * 70)
    
    # Initialize processor
    processor = DepthProcessor()
    
    # Create dummy detections
    detections = [
        {
            'bbox': [100, 150, 300, 400],
            'confidence': 0.85,
            'class_id': 2,
            'class_name': 'car'
        },
        {
            'bbox': [500, 200, 700, 450],
            'confidence': 0.92,
            'class_id': 7,
            'class_name': 'truck'
        }
    ]
    
    # Create dummy depth map
    depth_map = np.ones((900, 1200), dtype=np.float32) * 15.0
    
    # Build BEV grid
    detections_by_camera = {CameraPosition.FRONT: detections}
    depth_maps = {CameraPosition.FRONT: depth_map}
    
    bev_grid = processor.process_detections_to_bev(
        detections_by_camera, depth_maps
    )
    
    print(f"BEV grid shape: {bev_grid.shape}")
    print(f"Occupied pixels: {(bev_grid > 0.5).sum()}")
    print(f"Free space pixels: {(bev_grid < 0.1).sum()}")
    
    # Visualize
    bev_vis = processor.bev_to_visualization()
    print(f"BEV visualization shape: {bev_vis.shape}")


def example_coordinate_transformations():
    """Example: Test coordinate transformations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Coordinate Transformations")
    print("=" * 70)
    
    from stereo_depth import CoordinateTransform
    from stereo_depth import CameraParameters
    
    camera_params = CameraParameters()
    transform = CoordinateTransform(camera_params)
    
    # Test pixel -> 3D camera -> ego frame -> BEV
    u, v = 600, 450  # Pixel coordinates (centre of image)
    depth = 20.0     # metres
    
    # Step 1: Pixel to camera 3D
    X_cam, Y_cam, Z_cam = transform.pixel_to_camera_3d(u, v, depth)
    print(f"\nPixel ({u}, {v}) at {depth}m depth:")
    print(f"  Camera 3D: X={X_cam:.2f}, Y={Y_cam:.2f}, Z={Z_cam:.2f}")
    
    # Step 2: Camera to ego frame
    X_ego, Z_ego = transform.camera_to_ego_frame(X_cam, Z_cam, yaw_deg=0)
    print(f"  Ego frame: X={X_ego:.2f}, Z={Z_ego:.2f}")
    
    # Step 3: Ego to BEV grid
    col, row = transform.ego_to_bev_grid(X_ego, Z_ego)
    print(f"  BEV grid: col={col:.1f}, row={row:.1f}")


if __name__ == "__main__":
    # Run all examples
    example_depth_estimation()
    example_yolo_integration()
    example_bev_grid_construction()
    example_coordinate_transformations()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
