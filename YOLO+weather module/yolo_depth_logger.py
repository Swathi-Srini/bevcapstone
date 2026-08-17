"""
YOLO + Depth Estimation Logger for MetaDrive Manual Driving

Processes frames from MetaDrive manual driving, detects objects with YOLO,
estimates depth for each detection, and logs results.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from stereo_depth import DepthProcessor, CameraParameters
import csv
from datetime import datetime


class YOLODepthLogger:
    """
    Process MetaDrive frames with YOLO + depth estimation.
    Logs detections with depth values to CSV.
    """
    
    def __init__(self, model_path='../yolov8n.pt', conf_threshold=0.3):
        """Initialize YOLO model and depth processor."""
        self.yolo = YOLO(model_path)
        self.processor = DepthProcessor()
        self.camera_params = CameraParameters()
        self.conf_threshold = conf_threshold
        
        # Output tracking
        self.detections_log = []
        self.frame_count = 0
        
        print("✓ YOLO model loaded")
        print("✓ Depth processor initialized")
        print(f"✓ Confidence threshold: {conf_threshold}")
    
    def estimate_depth_from_bbox(self, bbox, image_shape):
        """
        Estimate depth using monocular cues.
        For objects in image center, estimate based on object size.
        """
        x1, y1, x2, y2 = bbox
        h_bbox = y2 - y1
        y_center = (y1 + y2) / 2
        
        # Estimate depth from vertical position and size
        # Objects lower in frame (higher y) are typically closer
        # This is a heuristic for ground plane estimation
        
        # Get image height
        img_height = image_shape[0]
        
        # Normalized vertical position (0 = top, 1 = bottom)
        v_norm = y_center / img_height
        
        # Simple heuristic: objects near bottom are closer
        # Depth range: 2m to 50m
        if v_norm > 0.7:
            estimated_depth = 2.0 + (1.0 - v_norm) * 5  # Very close
        elif v_norm > 0.5:
            estimated_depth = 8.0 + (1.0 - v_norm) * 10  # Medium range
        else:
            estimated_depth = 20.0 + (1.0 - v_norm) * 30  # Far range
        
        return float(estimated_depth)
    
    def process_frame(self, frame_path, step_num):
        """
        Process single frame: YOLO detection + depth estimation.
        
        Args:
            frame_path: Path to image file
            step_num: Frame step number
            
        Returns:
            List of detections with depth values
        """
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"✗ Failed to load {frame_path}")
            return []
        
        # Run YOLO detection
        results = self.yolo(image, conf=self.conf_threshold, verbose=False)
        
        frame_detections = []
        
        # Process each detection
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # Extract box info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]
                
                # Estimate depth from bounding box
                depth = self.estimate_depth_from_bbox((x1, y1, x2, y2), image.shape)
                
                # Store detection
                detection = {
                    'step': step_num,
                    'class': cls_name,
                    'confidence': round(conf, 3),
                    'x1': round(float(x1), 1),
                    'y1': round(float(y1), 1),
                    'x2': round(float(x2), 1),
                    'y2': round(float(y2), 1),
                    'bbox_width': round(float(x2 - x1), 1),
                    'bbox_height': round(float(y2 - y1), 1),
                    'depth_m': round(depth, 2),  # Estimated depth in meters
                }
                
                frame_detections.append(detection)
                self.detections_log.append(detection)
        
        return frame_detections
    
    def process_directory(self, image_dir, output_csv='yolo_depth_log.csv'):
        """
        Process all frames in directory.
        
        Args:
            image_dir: Directory containing frame images (step_*.png)
            output_csv: Output CSV file path
        """
        image_dir = Path(image_dir)
        
        # Find all frame images
        frame_files = sorted(image_dir.glob('step_*.png'))
        
        if not frame_files:
            print(f"✗ No frames found in {image_dir}")
            return
        
        print(f"\n📽️  Processing {len(frame_files)} frames...")
        print(f"📁 Input: {image_dir}")
        print(f"📄 Output: {output_csv}\n")
        
        # Process each frame
        for i, frame_path in enumerate(frame_files):
            # Extract step number
            step_num = int(frame_path.stem.split('_')[1])
            
            # Process frame
            detections = self.process_frame(frame_path, step_num)
            
            # Print progress
            if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                print(f"✓ Processed {i + 1}/{len(frame_files)} frames ({len(self.detections_log)} total detections)")
        
        # Save to CSV
        self.save_to_csv(output_csv)
        
        # Print summary
        self.print_summary()
    
    def save_to_csv(self, output_path):
        """Save detections to CSV file."""
        if not self.detections_log:
            print("✗ No detections to save")
            return
        
        # CSV headers
        headers = [
            'step', 'class', 'confidence',
            'x1', 'y1', 'x2', 'y2',
            'bbox_width', 'bbox_height',
            'depth_m'
        ]
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.detections_log)
        
        print(f"\n✓ Saved {len(self.detections_log)} detections to {output_path}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.detections_log:
            print("✗ No detections found")
            return
        
        print("\n" + "="*60)
        print("📊 DETECTION SUMMARY")
        print("="*60)
        
        # Count by class
        classes = {}
        for det in self.detections_log:
            cls = det['class']
            classes[cls] = classes.get(cls, 0) + 1
        
        print(f"\n📍 Total Detections: {len(self.detections_log)}")
        print(f"📹 Frames with Detections: {len(set(d['step'] for d in self.detections_log))}")
        
        print("\n🎯 By Class:")
        for cls, count in sorted(classes.items(), key=lambda x: x[1], reverse=True):
            avg_conf = np.mean([d['confidence'] for d in self.detections_log if d['class'] == cls])
            avg_depth = np.mean([d['depth_m'] for d in self.detections_log if d['class'] == cls])
            print(f"   {cls:15} : {count:3} detections | Avg Conf: {avg_conf:.2f} | Avg Depth: {avg_depth:.1f}m")
        
        # Depth statistics
        depths = [d['depth_m'] for d in self.detections_log]
        print(f"\n📏 Depth Statistics (meters):")
        print(f"   Min Depth   : {min(depths):.2f}m")
        print(f"   Max Depth   : {max(depths):.2f}m")
        print(f"   Mean Depth  : {np.mean(depths):.2f}m")
        print(f"   Median Depth: {np.median(depths):.2f}m")
        print(f"   Std Dev     : {np.std(depths):.2f}m")
        
        # Confidence statistics
        confs = [d['confidence'] for d in self.detections_log]
        print(f"\n📊 Confidence Statistics:")
        print(f"   Min Conf    : {min(confs):.3f}")
        print(f"   Max Conf    : {max(confs):.3f}")
        print(f"   Mean Conf   : {np.mean(confs):.3f}")
        
        print("\n" + "="*60 + "\n")
    
    def get_detections_at_step(self, step_num):
        """Get all detections for a specific frame step."""
        return [d for d in self.detections_log if d['step'] == step_num]


def main():
    """Main entry point."""
    import sys
    
    # Configuration
    FRAME_DIR = "../../manual_drive_output"  # Relative path from script
    OUTPUT_CSV = "../../manual_drive_output/yolo_depth_log.csv"
    MODEL_PATH = "../../yolov8n.pt"
    CONFIDENCE = 0.3
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    frame_dir = (script_dir / FRAME_DIR).resolve()
    output_csv = (script_dir / OUTPUT_CSV).resolve()
    model_path = (script_dir / MODEL_PATH).resolve()
    
    print("\n" + "="*60)
    print("🚗 YOLO + DEPTH ESTIMATION LOGGER")
    print("="*60)
    print(f"📂 Input Directory: {frame_dir}")
    print(f"📊 Output CSV: {output_csv}")
    print(f"🤖 YOLO Model: {model_path}")
    print("="*60 + "\n")
    
    # Initialize logger
    logger = YOLODepthLogger(
        model_path=str(model_path),
        conf_threshold=CONFIDENCE
    )
    
    # Process frames
    logger.process_directory(frame_dir, str(output_csv))


if __name__ == '__main__':
    main()
