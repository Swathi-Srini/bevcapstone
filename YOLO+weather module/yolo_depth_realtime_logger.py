"""
Real-time YOLO + Depth Logger for MetaDrive Manual Driving

Integrates with MetaDrive to log detections + depth while manually driving.
Place this in your MetaDrive manual drive script.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from stereo_depth import DepthProcessor
import csv
from pathlib import Path


class RealtimeYOLODepthLogger:
    """
    Real-time YOLO + Depth estimation for MetaDrive manual driving.
    Logs each detection frame with depth estimates.
    """
    
    def __init__(self, model_path='../yolov8n.pt', conf_threshold=0.3, 
                 output_dir='./depth_logs'):
        """
        Initialize logger.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
            output_dir: Directory to save logs
        """
        self.yolo = YOLO(model_path)
        self.processor = DepthProcessor()
        self.conf_threshold = conf_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Detections log
        self.frame_count = 0
        self.detections = []
        
        # CSV writer setup
        self.csv_path = self.output_dir / 'yolo_depth_realtime.csv'
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = None
        
        print(f"✓ Logger initialized")
        print(f"✓ Output: {self.output_dir}")
    
    def estimate_depth_monocular(self, bbox, image_shape):
        """Estimate depth from bounding box using monocular cues."""
        x1, y1, x2, y2 = bbox
        h_bbox = y2 - y1
        y_center = (y1 + y2) / 2
        
        # Normalized vertical position
        v_norm = y_center / image_shape[0]
        
        # Simple heuristic for ground plane
        if v_norm > 0.7:
            depth = 2.0 + (1.0 - v_norm) * 5
        elif v_norm > 0.5:
            depth = 8.0 + (1.0 - v_norm) * 10
        else:
            depth = 20.0 + (1.0 - v_norm) * 30
        
        return float(depth)
    
    def process_frame(self, image, step_num):
        """
        Process single frame with YOLO + depth.
        
        Args:
            image: Input image (numpy array)
            step_num: Current step/frame number
            
        Returns:
            List of detections
        """
        # Run YOLO
        results = self.yolo(image, conf=self.conf_threshold, verbose=False)
        
        frame_detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                
                # Estimate depth
                depth = self.estimate_depth_monocular((x1, y1, x2, y2), image.shape)
                
                detection = {
                    'step': step_num,
                    'class': cls_name,
                    'confidence': round(conf, 3),
                    'x1': round(float(x1), 1),
                    'y1': round(float(y1), 1),
                    'x2': round(float(x2), 1),
                    'y2': round(float(y2), 1),
                    'width': round(float(x2 - x1), 1),
                    'height': round(float(y2 - y1), 1),
                    'depth_m': round(depth, 2),
                }
                
                # Initialize CSV on first detection
                if self.csv_writer is None:
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=detection.keys())
                    self.csv_writer.writeheader()
                
                # Write to CSV
                self.csv_writer.writerow(detection)
                self.csv_file.flush()
                
                frame_detections.append(detection)
        
        self.frame_count += 1
        
        # Print progress every 50 frames
        if self.frame_count % 50 == 0:
            print(f"  Step {step_num}: {len(frame_detections)} detections | Total: {len(self.detections) + len(frame_detections)}")
        
        return frame_detections
    
    def close(self):
        """Finalize logging."""
        if self.csv_file:
            self.csv_file.close()
        print(f"\n✓ Logging complete: {self.csv_path}")


# ============================================================================
# INTEGRATION EXAMPLE - Add this to your MetaDrive manual drive script
# ============================================================================

"""
Example: How to use in your manual drive script

from yolo_depth_logger import RealtimeYOLODepthLogger

# In your main loop:
logger = RealtimeYOLODepthLogger(
    model_path='./yolov8n.pt',
    conf_threshold=0.3,
    output_dir='./depth_logs'
)

for step in range(num_steps):
    # Your existing code to get image from MetaDrive
    image = env.render('rgb_array')  # or however you get the image
    
    # Process with YOLO + depth
    detections = logger.process_frame(image, step)
    
    # Your driving code...
    action = get_manual_action()
    obs, reward, done, info = env.step(action)

# At the end
logger.close()
"""
