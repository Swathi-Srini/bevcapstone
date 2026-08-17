# Integration Guide: Stereo Depth + YOLO Detection

How to integrate stereo depth estimation with the existing YOLO detection module for complete 3D perception.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Autonomous Driving Stack                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │          Multi-Camera Input (4 cameras)              │ │
│  │  Front Left | Front Right | Left | Right | Rear    │ │
│  └──────────────────────────────────────────────────────┘ │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │        Stereo Depth Estimation Module                │ │
│  │  • Semi-Global Matching (SGM)                        │ │
│  │  • Disparity → Depth conversion                      │ │
│  │  • Monocular ground-plane projection                │ │
│  └──────────────────────────────────────────────────────┘ │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │        YOLO Object Detection Module                  │ │
│  │  • Multi-camera YOLO inference                       │ │
│  │  • Bounding box generation                           │ │
│  │  • Class filtering (obstacles only)                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │    3D Detection Fusion (New)                          │ │
│  │  • Depth-aware box lifting                           │ │
│  │  • Coordinate transformations                         │ │
│  │  • Physical size estimation                           │ │
│  └──────────────────────────────────────────────────────┘ │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │         BEV Grid Construction                        │ │
│  │  • 64×64 occupancy grid                              │ │
│  │  • Obstacle placement                                │ │
│  │  • Route visualization                               │ │
│  └──────────────────────────────────────────────────────┘ │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │    Policy Network (PPO)                              │ │
│  │  • CNN branch: BEV grid processing                   │ │
│  │  • MLP branch: Scalar state processing               │ │
│  │  • Action output: throttle, steering, brake          │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Update YOLO Utils for Depth Integration

Create a new file `yolo/yolo_depth_integration.py`:

```python
"""
YOLO + Stereo Depth Integration Module

Combines YOLO detections with stereo depth for 3D obstacle perception.
"""

from typing import Dict, List, Optional
import numpy as np
import cv2

from stereo_depth import (
    DepthProcessor, 
    CameraPosition,
    CoordinateTransform,
    PhysicalSizeEstimator
)


class YOLODepthIntegration:
    """Integrate YOLO detections with stereo depth maps."""
    
    def __init__(self, yolo_model=None):
        """
        Initialize integration module.
        
        Args:
            yolo_model: Ultralytics YOLO model instance
        """
        self.yolo_model = yolo_model
        self.depth_processor = DepthProcessor()
        
        # Obstacle class IDs from COCO (cars, trucks, buses, motorcycles, pedestrians)
        self.obstacle_classes = {0, 2, 3, 5, 7}
        self.confidence_threshold = 0.4
    
    def process_frame(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        use_yolo: bool = True
    ) -> Dict:
        """
        Process single frame: compute depth + detect obstacles.
        
        Args:
            left_img: Left camera image (BGR)
            right_img: Right camera image (BGR)
            use_yolo: Whether to run YOLO detection
            
        Returns:
            Dictionary with:
            - depth_map: Stereo depth
            - detections: Enhanced 3D detections
            - bev_grid: Bird's eye view grid
            - visualizations: Depth/BEV visualizations
        """
        result = {}
        
        # Step 1: Compute stereo depth
        depth_result = self.depth_processor.process_stereo_pair(
            left_img, right_img, compute_point_cloud=False
        )
        depth_map = depth_result['depth']
        result['depth_map'] = depth_map
        
        # Step 2: Run YOLO detection
        detections = []
        if use_yolo and self.yolo_model is not None:
            detections = self._detect_obstacles_yolo(left_img)
        
        result['detections'] = detections
        
        # Step 3: Build BEV grid from detections
        if detections:
            detections_by_camera = {CameraPosition.FRONT: detections}
            depth_maps = {CameraPosition.FRONT: depth_map}
            
            bev_grid = self.depth_processor.process_detections_to_bev(
                detections_by_camera, depth_maps
            )
        else:
            bev_grid = self.depth_processor._initialize_bev_grid()
        
        result['bev_grid'] = bev_grid
        
        # Step 4: Generate visualizations
        result['visualizations'] = {
            'depth': self.depth_processor.depth_to_visualization(depth_map),
            'disparity': self.depth_processor.disparity_to_visualization(
                depth_result['disparity']
            ),
            'bev': self.depth_processor.bev_to_visualization()
        }
        
        return result
    
    def _detect_obstacles_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect obstacles using YOLO.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections with obstacle classes filtered
        """
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model(image, conf=self.confidence_threshold)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            
            # Filter to obstacle classes
            if class_id not in self.obstacle_classes:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf.item())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': results.names.get(class_id, f'class_{class_id}')
            })
        
        return detections


class RealTimePerception:
    """Real-time multi-camera perception pipeline."""
    
    def __init__(self, yolo_model=None):
        """Initialize real-time perception."""
        self.integration = YOLODepthIntegration(yolo_model)
        self.camera_configs = {
            'front_left': None,
            'front_right': None,
            'side_left': None,
            'side_right': None,
            'rear': None
        }
    
    def set_camera_feed(self, camera_name: str, video_source):
        """Set camera feed (video file or camera index)."""
        self.camera_configs[camera_name] = video_source
    
    def process_frame_from_cameras(self) -> Optional[Dict]:
        """Process frame from camera feeds."""
        # Load from cameras or video files
        # This would integrate with actual camera/video input
        pass


def create_yolo_depth_detector(yolo_weights_path: str):
    """
    Create integrated YOLO + Depth detector.
    
    Args:
        yolo_weights_path: Path to YOLO weights file
        
    Returns:
        YOLODepthIntegration instance
    """
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_weights_path)
        return YOLODepthIntegration(yolo_model)
    except ImportError:
        print("Warning: ultralytics not installed")
        return YOLODepthIntegration(None)
```

## Step 2: Update the Environment Integration

Update your environment's perception module to use stereo depth:

```python
# In your environment step function
def _get_perception_observation(self):
    """Get BEV observation with stereo depth."""
    
    # Process front stereo cameras
    left_img = self.get_camera_image('front_left')
    right_img = self.get_camera_image('front_right')
    
    # Run integrated perception
    perception_result = self.perception_module.process_frame(
        left_img, right_img, use_yolo=True
    )
    
    # Extract BEV grid
    bev_grid = perception_result['bev_grid']
    
    # Return as observation
    return {
        'bev_grid': bev_grid,
        'depth_map': perception_result['depth_map'],
        'detections': perception_result['detections']
    }
```

## Step 3: Example Training Integration

```python
"""
Example integration with RL training pipeline.
"""

import numpy as np
from yolo.yolo_depth_integration import create_yolo_depth_detector


class AutonomousDrivingEnvWithStereo:
    """Environment with stereo depth perception."""
    
    def __init__(self, yolo_weights='yolov8n.pt'):
        """Initialize environment with stereo depth."""
        self.perception = create_yolo_depth_detector(yolo_weights)
        
        # State space
        self.observation_space = {
            'bev_grid': (64, 64),  # BEV occupancy grid
            'scalar_state': 6       # [v, s, d, theta_err, kappa, d_goal]
        }
        
        # Action space
        self.action_space = 3  # [steering, throttle, brake]
    
    def reset(self):
        """Reset environment."""
        # Initialize cameras and state
        pass
    
    def step(self, action):
        """Execute action step."""
        # Apply control
        steering, throttle, brake = action
        self.vehicle.apply_control(steering, throttle, brake)
        
        # Get stereo perception
        left_img = self._get_left_camera()
        right_img = self._get_right_camera()
        
        perception_result = self.perception.process_frame(left_img, right_img)
        
        # Extract BEV for observation
        bev_grid = perception_result['bev_grid']
        
        # Compute scalar state
        scalar_state = self._compute_scalar_state()
        
        # Compute reward
        reward = self._compute_reward(action, perception_result)
        
        # Check termination
        done = self._check_termination(perception_result)
        
        return {
            'bev_grid': bev_grid,
            'scalar_state': scalar_state
        }, reward, done, {}
    
    def _compute_reward(self, action, perception_result):
        """Compute reward with depth information."""
        # Reward components can now use depth information
        # e.g., penalize collisions based on depth
        reward = 0.0
        
        # Progress reward
        reward += self._compute_progress_reward()
        
        # Depth-aware collision penalty
        depth_map = perception_result['depth_map']
        collision_risk = self._assess_collision_risk(depth_map)
        reward -= collision_risk * 10.0
        
        return reward
    
    def _assess_collision_risk(self, depth_map):
        """Assess collision risk from depth map."""
        # High risk if something is very close
        min_depth = np.nanmin(depth_map)
        return 1.0 / (1.0 + min_depth)  # Risk decreases with distance


# Example training loop
def train_with_stereo_perception(num_episodes=1000):
    """Train agent with stereo depth perception."""
    env = AutonomousDrivingEnvWithStereo(yolo_weights='yolov8n.pt')
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Policy action
            action = policy(obs)  # Your PPO policy
            
            # Step environment (includes stereo depth processing)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

## Step 4: Performance Considerations

### Processing Time Breakdown

Expected timing per frame (900×1200 images):

| Component | Time | Notes |
|-----------|------|-------|
| Stereo SGM matching | 30-50ms | CPU-based, GPU variant available |
| Disparity filtering | 10-20ms | Optional, improves quality |
| YOLO inference | 20-40ms | Depends on model size |
| BEV construction | 5-10ms | Fast, mostly CPU |
| **Total** | **65-120ms** | **~8-15 FPS** |

### Optimization Strategies

1. **Reduce Resolution:**
```python
# Process at half resolution
h, w = left_img.shape[:2]
left_small = cv2.resize(left_img, (w//2, h//2))
right_small = cv2.resize(right_img, (w//2, h//2))
result = processor.process_stereo_pair(left_small, right_small)
```

2. **Skip Point Cloud:**
```python
# Don't compute expensive 3D points
result = processor.process_stereo_pair(
    left_img, right_img, 
    compute_point_cloud=False
)
```

3. **Batch Processing:**
```python
# Process multiple frames in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(processor.process_stereo_pair, l_img, r_img)
        for l_img, r_img in image_pairs
    ]
    results = [f.result() for f in futures]
```

## Step 5: Testing Integration

```python
# test_integration.py

from yolo.yolo_depth_integration import create_yolo_depth_detector
import cv2

# Create detector
detector = create_yolo_depth_detector('yolov8n.pt')

# Load test images
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Process
result = detector.process_frame(left_img, right_img, use_yolo=True)

# Check results
print(f"Detections: {len(result['detections'])}")
print(f"BEV grid shape: {result['bev_grid'].shape}")
print(f"Depth range: {result['depth_map'].min():.2f} to {result['depth_map'].max():.2f}m")

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(result['visualizations']['depth'], cv2.COLOR_BGR2RGB))
axes[0].set_title('Depth Map')
axes[1].imshow(cv2.cvtColor(result['visualizations']['disparity'], cv2.COLOR_BGR2RGB))
axes[1].set_title('Disparity Map')
axes[2].imshow(cv2.cvtColor(result['visualizations']['bev'], cv2.COLOR_BGR2RGB))
axes[2].set_title('BEV Grid')

plt.tight_layout()
plt.savefig('integration_test.png')
print("Saved integration_test.png")
```

## Next Steps

1. **Test with real camera data** - Use `visualize_stereo.py` with actual stereo images
2. **Calibrate cameras** - Run calibration if using physical cameras
3. **Integrate with RL training** - Update your environment to use perception
4. **Optimize performance** - Profile and optimize based on hardware
5. **Validate accuracy** - Test depth estimates against ground truth

