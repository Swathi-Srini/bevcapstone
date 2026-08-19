"""
Real-Time YOLO + Depth Logger for MetaDrive Manual Driving
Logs every detection with depth estimation directly to terminal
Based on Technical Spec Section 3: Stereo Depth Estimation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from datetime import datetime

class RealTimeDepthLogger:
    """
    Real-time depth estimation from YOLO detections.
    
    Uses monocular depth formula:
    Z = f·B / d_px = 500 / d_px (from technical spec Equation 5)
    
    For each YOLO detection:
    - Estimate depth from bounding box size
    - Log to terminal in real-time
    """
    
    # Technical Spec Parameters (Section 2-3)
    FOCAL_LENGTH = 1000.0  # f (pixels)
    BASELINE = 0.5  # B (meters) - stereo baseline
    IMG_WIDTH = 1200  # pixels
    IMG_HEIGHT = 900  # pixels
    FOV_H = 60  # degrees (horizontal)

    # Depth formula: Z = f*B / d_px = 500 / d_px
    DEPTH_CONSTANT = FOCAL_LENGTH * BASELINE  # 500
    OBJECT_WIDTH_METERS = {
        'car': 1.8,
        'truck': 2.4,
        'bus': 2.8,
        'motorcycle': 0.7,
        'bicycle': 0.6,
        'person': 0.5,
        'bench': 1.5,
        'traffic light': 0.4,
        'surfboard': 0.6,
        'boat': 1.8,
    }
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
        """Initialize YOLO model."""
        self.yolo = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.step = 0
        self.total_detections = 0
        
        print("\n" + "="*70)
        print("🚗 REAL-TIME YOLO + DEPTH LOGGER")
        print("="*70)
        print(f"✓ YOLO model loaded: {model_path}")
        print(f"✓ Confidence threshold: {conf_threshold}")
        print(f"✓ Focal length (f): {self.FOCAL_LENGTH} px")
        print(f"✓ Stereo baseline (B): {self.BASELINE} m")
        print(f"✓ Depth formula: Z = {self.DEPTH_CONSTANT} / d_px")
        print("="*70)
        print(f"{'Step':<8} {'Class':<12} {'Conf':<8} {'Bbox':<20} {'Depth (m)':<12}")
        print("-"*70)

    def estimate_depth_from_bbox_size(self, bbox_width, bbox_height, image_width=None, class_name='car'):
        """
        Estimate depth using the same perspective relationship used by the document.
        With a single live MetaDrive RGB camera, this is the valid monocular approximation:
        Z = f * W_obj / W_px
        and the corresponding disparity proxy is d_px = f * B / Z.
        """
        if image_width is None:
            image_width = self.IMG_WIDTH

        bbox_width = max(1.0, float(bbox_width))
        object_width = self.OBJECT_WIDTH_METERS.get(class_name.lower(), 1.8)

        # Perspective size relation: Z ≈ f * W_obj / W_px
        depth_m = self.FOCAL_LENGTH * object_width / bbox_width
        estimated_disparity = self.DEPTH_CONSTANT / max(depth_m, 1e-3)

        # Keep the result in a practical range for the live simulation
        depth_m = float(np.clip(depth_m, 5.0, 200.0))
        estimated_disparity = float(np.clip(estimated_disparity, 2.5, 100.0))

        return depth_m, estimated_disparity
    
    def annotate_frame(self, frame, detections):
        """Draw detection boxes and labels directly on the image."""
        annotated = frame.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for det in detections:
            x1 = int(max(0, det['x1']))
            y1 = int(max(0, det['y1']))
            x2 = int(min(annotated.shape[1], det['x2']))
            y2 = int(min(annotated.shape[0], det['y2']))
            cls_name = det['class']
            conf = det['confidence']
            depth_m = det['depth_m']

            color = (0, 255, 255)
            if cls_name in ('car', 'truck', 'bus'):
                color = (0, 255, 0)
            elif cls_name in ('person', 'bicycle'):
                color = (0, 165, 255)
            elif cls_name in ('traffic light', 'stop sign'):
                color = (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{cls_name} {conf:.2f} {depth_m:.1f}m",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    def process_frame(self, frame, step_num):
        """
        Process single frame: YOLO detection + depth estimation.
        
        Args:
            frame: Input image (numpy array)
            step_num: Current step/frame number
            
        Returns:
            List of detections with depth
        """
        self.step = step_num
        
        # Run YOLO detection
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        
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
                
                # Calculate bbox dimensions
                bbox_width = float(x2 - x1)
                bbox_height = float(y2 - y1)

                # MetaDrive exposes only a single RGB camera in the live loop, so the
                # valid real-time approximation is based on object size using the same
                # perspective/depth relationship from the document.
                depth_m, disparity_est = self.estimate_depth_from_bbox_size(
                    bbox_width,
                    bbox_height,
                    image_width=frame.shape[1],
                    class_name=cls_name,
                )
                
                # Create detection record
                detection = {
                    'step': step_num,
                    'class': cls_name,
                    'confidence': conf,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'depth_m': depth_m,
                    'disparity_est': disparity_est,
                }
                
                # Print to terminal in real-time
                bbox_str = f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                print(f"{step_num:<8} {cls_name:<12} {conf:<8.3f} {bbox_str:<20} {depth_m:<12.2f}")
                
                frame_detections.append(detection)
                self.total_detections += 1
        
        return frame_detections


def main():
    """
    Main entry point - integrate with your MetaDrive manual drive script.
    """
    
    # Initialize logger
    logger = RealTimeDepthLogger(
        model_path='yolov8n.pt',
        conf_threshold=0.3
    )
    
    print("\n✓ Ready to process frames")
    print("✓ Use in your MetaDrive script as:")
    print("   logger.process_frame(image, step)")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
