"""
Stereo depth visualization utility.

Interactive viewer for stereo depth maps with disparity and depth visualizations.

PowerShell example:
  python "e:\Capstone\Minimal_Grid_env\YOLO+weather module\stereo_depth\visualize_stereo.py" \
    "E:\path\to\left.jpg" "E:\path\to\right.jpg"
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# Ensure parent-of-package is on sys.path so absolute imports work
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from stereo_depth.depth_processor import DepthProcessor
from stereo_depth.camera_params import CameraPosition


class StereoVisualizer:
    """Interactive stereo depth map visualizer."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.processor = DepthProcessor()
        self.camera_params = self.processor.camera_params
        
        # Current images and results
        self.left_img = None
        self.right_img = None
        self.depth = None
        self.disparity = None
        self.point_cloud = None
    
    def load_images(self, left_path: str, right_path: str) -> bool:
        """Load stereo image pair."""
        try:
            self.left_img = cv2.imread(left_path)
            self.right_img = cv2.imread(right_path)
            
            if self.left_img is None or self.right_img is None:
                print(f"Error: Could not load images from {left_path} or {right_path}")
                return False
            
            print(f"✓ Loaded left image: {self.left_img.shape}")
            print(f"✓ Loaded right image: {self.right_img.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading images: {e}")
            return False
    
    def compute_depth_maps(self) -> bool:
        """Compute stereo depth maps."""
        if self.left_img is None or self.right_img is None:
            print("Error: Images not loaded")
            return False
        
        try:
            print("\nProcessing stereo pair...")
            print("  Computing disparity map (this may take 30-60 seconds)...")
            
            result = self.processor.process_stereo_pair(
                self.left_img,
                self.right_img,
                compute_point_cloud=True
            )
            
            self.disparity = result['disparity']
            self.depth = result['depth']
            self.point_cloud = result.get('point_cloud')
            
            print(f"✓ Disparity map: {self.disparity.shape}")
            print(f"  Range: {np.nanmin(self.disparity):.1f} to {np.nanmax(self.disparity):.1f} pixels")
            print(f"✓ Depth map: {self.depth.shape}")
            print(f"  Range: {np.nanmin(self.depth):.2f} to {np.nanmax(self.depth):.2f} m")
            
            if self.point_cloud is not None:
                print(f"✓ Point cloud: {self.point_cloud.shape[0]:,} points")
            
            return True
        except Exception as e:
            print(f"Error computing depth: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_interactive(self) -> None:
        """Launch interactive visualization window."""
        if self.depth is None:
            print("Error: Depth not computed")
            return
        
        print("\n" + "=" * 70)
        print("INTERACTIVE VISUALIZATION")
        print("=" * 70)
        print("\nControls:")
        print("  [d] - Toggle depth map")
        print("  [s] - Toggle disparity map")
        print("  [l] - Toggle left image")
        print("  [r] - Toggle right image")
        print("  [c] - Print calibration summary")
        print("  [q] - Quit")
        print()
        
        # Generate initial visualizations
        depth_vis = self.processor.depth_to_visualization(self.depth)
        disp_vis = self.processor.disparity_to_visualization(self.disparity)
        
        # Create display window
        cv2.namedWindow('Stereo Visualization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stereo Visualization', 1200, 900)
        
        current_display = depth_vis
        current_name = "Depth Map"
        
        while True:
            # Display current visualization with text overlay
            display_with_text = current_display.copy()
            cv2.putText(
                display_with_text, f"[{current_name}]  Press: d=depth, s=disparity, l=left, r=right, c=calib, q=quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.imshow('Stereo Visualization', display_with_text)
            
            # Wait for key press (1ms timeout)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('d'):
                current_display = depth_vis
                current_name = "Depth Map"
                print("Showing depth map")
            elif key == ord('s'):
                current_display = disp_vis
                current_name = "Disparity Map"
                print("Showing disparity map")
            elif key == ord('l'):
                current_display = self.left_img
                current_name = "Left Image"
                print("Showing left image")
            elif key == ord('r'):
                current_display = self.right_img
                current_name = "Right Image"
                print("Showing right image")
            elif key == ord('c'):
                self.processor.print_calibration_summary()
        
        cv2.destroyAllWindows()
        print("Visualization closed.")
    
    def save_results(self, output_dir: str = "stereo_results") -> None:
        """Save visualization results."""
        if self.depth is None:
            print("Error: No results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save depth visualization
            depth_vis = self.processor.depth_to_visualization(self.depth)
            cv2.imwrite(str(output_path / 'depth_map.png'), depth_vis)
            print(f"✓ Saved: {output_path / 'depth_map.png'}")
            
            # Save disparity visualization
            disp_vis = self.processor.disparity_to_visualization(self.disparity)
            cv2.imwrite(str(output_path / 'disparity_map.png'), disp_vis)
            print(f"✓ Saved: {output_path / 'disparity_map.png'}")
            
            # Save left and right images
            cv2.imwrite(str(output_path / 'left_image.png'), self.left_img)
            cv2.imwrite(str(output_path / 'right_image.png'), self.right_img)
            print(f"✓ Saved: {output_path / 'left_image.png'}")
            print(f"✓ Saved: {output_path / 'right_image.png'}")
            
            # Save depth and disparity as numpy arrays
            np.save(str(output_path / 'depth_map.npy'), self.depth)
            np.save(str(output_path / 'disparity_map.npy'), self.disparity)
            print(f"✓ Saved: {output_path / 'depth_map.npy'}")
            print(f"✓ Saved: {output_path / 'disparity_map.npy'}")
            
            # Save point cloud if available
            if self.point_cloud is not None:
                np.save(str(output_path / 'point_cloud.npy'), self.point_cloud)
                print(f"✓ Saved: {output_path / 'point_cloud.npy'}")
            
            # Save metadata
            with open(output_path / 'metadata.txt', 'w') as f:
                f.write("Stereo Depth Estimation Results\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Left image shape: {self.left_img.shape}\n")
                f.write(f"Right image shape: {self.right_img.shape}\n")
                f.write(f"Depth map shape: {self.depth.shape}\n")
                f.write(f"Depth range: {np.nanmin(self.depth):.2f} to {np.nanmax(self.depth):.2f} m\n")
                f.write(f"Disparity range: {np.nanmin(self.disparity):.1f} to {np.nanmax(self.disparity):.1f} px\n")
                if self.point_cloud is not None:
                    f.write(f"Point cloud: {self.point_cloud.shape[0]:,} points\n")
                f.write("\n")
                f.write("Camera Parameters:\n")
                f.write(f"  Focal length: {self.camera_params.FOCAL_LENGTH} px\n")
                f.write(f"  Baseline: {self.camera_params.STEREO_BASELINE} m\n")
                f.write(f"  Resolution: {self.camera_params.IMAGE_WIDTH}x{self.camera_params.IMAGE_HEIGHT} px\n")
                f.write(f"  FOV: {self.camera_params.FOV_HORIZONTAL}°\n")
            
            print(f"✓ Saved: {output_path / 'metadata.txt'}")
            print(f"\nAll results saved to: {output_path}")
        
        except Exception as e:
            print(f"Error saving results: {e}")


def show_in_windows(left_path: str, right_path: str) -> None:
    """Load and visualize stereo image pair."""
    print("\n" + "=" * 70)
    print("STEREO DEPTH VISUALIZATION")
    print("=" * 70 + "\n")
    
    # Initialize visualizer
    viz = StereoVisualizer()
    
    # Load images
    if not viz.load_images(left_path, right_path):
        return
    
    # Compute depth
    if not viz.compute_depth_maps():
        return
    
    # Print calibration summary
    viz.processor.print_calibration_summary()
    
    # Show interactive visualization
    viz.visualize_interactive()
    
    # Save results
    viz.save_results()


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python visualize_stereo.py <left_image> <right_image>")
        print("\nExample:")
        print("  python visualize_stereo.py left.jpg right.jpg")
        print("\nSupported formats: .jpg, .png, .bmp")
        sys.exit(1)
    
    left_path = sys.argv[1]
    right_path = sys.argv[2]
    
    # Verify files exist
    if not Path(left_path).exists():
        print(f"Error: Left image not found: {left_path}")
        sys.exit(1)
    if not Path(right_path).exists():
        print(f"Error: Right image not found: {right_path}")
        sys.exit(1)
    
    show_in_windows(left_path, right_path)


if __name__ == "__main__":
    main()
    depth = result.get("depth")
    pc = result.get("point_cloud")

    # Visualizations
    disp_vis = proc.disparity_to_visualization(disparity)
    depth_vis = proc.depth_to_visualization(depth)
    bev_vis = proc.bev_to_visualization()

    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / "disparity.png"), disp_vis)
    cv2.imwrite(str(save_dir / "depth.png"), depth_vis)
    cv2.imwrite(str(save_dir / "bev.png"), bev_vis)

    if pc is not None and pc.size > 0:
        ply_path = save_dir / "point_cloud.ply"
        save_point_cloud_ply(pc, ply_path)

    print(f"Saved outputs to: {save_dir}")


def show_in_windows(left_path: Path, right_path: Path):
    proc = DepthProcessor()

    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None or right is None:
        raise SystemExit("Failed to read input images. Check the paths.")

    result = proc.process_stereo_pair(left, right, compute_point_cloud=False)
    disparity = result.get("disparity")
    depth = result.get("depth")

    disp_vis = proc.disparity_to_visualization(disparity)
    depth_vis = proc.depth_to_visualization(depth)
    bev_vis = proc.bev_to_visualization()

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity", disp_vis)
    cv2.imshow("Depth", depth_vis)
    cv2.imshow("BEV", bev_vis)

    print("Press any key in an OpenCV window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Visualize stereo disparity, depth and BEV")
    ap.add_argument("left", help="Left image path")
    ap.add_argument("right", help="Right image path")
    ap.add_argument("--save", action="store_true", help="Save outputs to disk")
    ap.add_argument("--out", default=None, help="Output directory (optional)")
    args = ap.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)

    if args.save:
        out_dir = Path(args.out) if args.out else left_path.parent / "stereo_outputs"
        build_and_save(left_path, right_path, out_dir)
    else:
        show_in_windows(left_path, right_path)


if __name__ == "__main__":
    main()
