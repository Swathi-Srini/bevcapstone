"""Compare YOLO detections on a saved clean image and its weathered version.

Run from the repository root, for example:
    .\\venv\\Scripts\\python.exe ".\\YOLO+weather module\\benchmark_weather_perception.py" --image .\\sample.png --weather fog --level 0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from weather.weather_utils import apply_weather, prepare_image
from yolo.yolo_utils import annotate_image, ensure_yolo_model, run_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO detections before and after synthetic fog/rain on a saved image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB/BGR image path.")
    parser.add_argument("--weather", choices=("fog", "rain", "all"), default="fog")
    parser.add_argument("--level", type=float, default=0.5, help="Weather intensity from 0 to 1.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("weather_benchmark_output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.level <= 1.0:
        raise ValueError("--level must be between 0 and 1")

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    clean = prepare_image(image)
    weathered = apply_weather(clean, args.weather, args.level)

    model = ensure_yolo_model(args.device, args.yolo_model, None, args.confidence_threshold)
    clean_detections = run_yolo(model, clean)
    weathered_detections = run_yolo(model, weathered)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "clean_annotated.png"), annotate_image(clean, clean_detections))
    cv2.imwrite(str(args.output_dir / f"{args.weather}_annotated.png"), annotate_image(weathered, weathered_detections))

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "weather", "level", "detections", "mean_confidence"))
        writer.writeheader()
        for condition, detections in (("clean", clean_detections), ("weathered", weathered_detections)):
            mean_confidence = sum(item["confidence"] for item in detections) / len(detections) if detections else 0.0
            writer.writerow({
                "condition": condition,
                "weather": args.weather,
                "level": args.level,
                "detections": len(detections),
                "mean_confidence": f"{mean_confidence:.4f}",
            })

    print(f"Clean detections: {len(clean_detections)}")
    print(f"Weathered detections: {len(weathered_detections)}")
    print(f"Saved benchmark outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
