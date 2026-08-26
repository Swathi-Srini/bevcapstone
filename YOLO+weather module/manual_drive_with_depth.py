"""
LEGACY MetaDrive manual-drive monocular bounding-box-depth demo.

This script predates the active front StereoSGBM integration. It estimates
range from YOLO box dimensions and must not be used to represent stereo depth.
Use ``manual_drive_stereo_yolo_weather.py`` for the active workflow.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from realtime_depth_logger import RealTimeDepthLogger
from weather.weather_utils import prepare_image

try:
    import keyboard
except ImportError:
    print("Installing keyboard package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard", "-q"])
    import keyboard


def get_keyboard_action():
    """Get action from keyboard input."""
    steer = 0.0
    throttle = 0.0
    
    if keyboard.is_pressed('w'):
        throttle = 0.5
    if keyboard.is_pressed('s'):
        throttle = -0.3
    if keyboard.is_pressed('a'):
        steer = -0.5
    if keyboard.is_pressed('d'):
        steer = 0.5
    
    return np.array([steer, throttle], dtype=np.float32)


def main():
    """Main manual drive loop."""
    
    print("\n" + "="*70)
    print("MetaDrive Manual Drive with Real-Time Depth Logging")
    print("="*70)
    print("Controls:")
    print("  W/S - Throttle forward/backward")
    print("  A/D - Steer left/right")
    print("  Q   - Quit")
    print("="*70 + "\n")
    
    try:
        from metadrive import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
    except ImportError:
        print("✗ MetaDrive not installed")
        print("Install with: pip install metadrive-simulator")
        return 1
    
    # Initialize depth logger
    depth_logger = RealTimeDepthLogger(
        model_path='yolov8n.pt',
        conf_threshold=0.3
    )
    
    # MetaDrive config: use the RGB camera sensor for image observations.
    # This matches the MetaDrive 0.4.3 API and avoids the missing 'rgb_camera' KeyError.
    config = {
        "manual_control": False,
        "use_render": False,
        "window_size": (480, 270),
        "traffic_density": 0.2,
        "image_observation": True,
        "norm_pixel": False,
        "num_scenarios": 1,
        "horizon": 99999,
        "crash_vehicle_done": False,
        "out_of_road_done": False,
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": (RGBCamera, 480, 270),
        },
    }
    
    env = MetaDriveEnv(config)
    obs, info = env.reset()
    
    window_name = "Manual Drive with Depth Logging"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    
    step = 0
    
    try:
        print("Started manual driving. Watch terminal for depth detections.\n")
        
        while True:
            # Get keyboard action
            action = get_keyboard_action()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get image from observation
            if isinstance(obs, dict) and "image" in obs:
                frame = obs["image"]
                if isinstance(frame, np.ndarray):
                    frame = prepare_image(frame)
                else:
                    print("Warning: frame is not numpy array")
                    frame = np.zeros((270, 480, 3), dtype=np.uint8)
            else:
                frame = np.zeros((270, 480, 3), dtype=np.uint8)
            
            # Resize to standard size for YOLO
            frame_resized = cv2.resize(frame, (1200, 900))
            
            # Log depths for detections in this frame
            detections = depth_logger.process_frame(frame_resized, step)
            annotated_frame = depth_logger.annotate_frame(frame_resized.copy(), detections)
            
            # Display frame in window
            frame_display = cv2.resize(annotated_frame, (960, 540))
            cv2.putText(frame_display, f"Step: {step}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame_display)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
                print("\n\nQuitting...")
                break
            
            if done:
                print("\n\nEpisode ended.")
                break
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        env.close()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f"✓ Manual drive complete")
        print(f"✓ Total steps: {step}")
        print(f"✓ Total detections logged: {depth_logger.total_detections}")
        print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
