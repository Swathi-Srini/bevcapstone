"""Build the video table and the frame-montage figure from the final videos in report/videos/.

File names follow  front_camera__seed5169__<model>__<weather>__<outcome>.mp4 .  The number of steps is the number of frames (one frame per step, 10 steps = 1 s).
Output: report/figures/fig_seed5169_front_camera_frames.pdf/.png and report/tables/video_outcomes.tex
Run from the repository root:   venv\\Scripts\\python.exe report\\make_video_assets.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
VID = ROOT / "report" / "videos"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
MODELS = [("baseline_behaviour_cloning_no_traffic_demonstrations", "Baseline Driving Model\nTrained Without Traffic"),
          ("baseline_behaviour_cloning_sparse_traffic_demonstrations", "Baseline Driving Model\nTrained With Traffic"),
          ("headline_model_map_overlay_three_scalars", "Proposed Bird's-Eye-View\nDriving Model"),
          ("pixel_policy_front_image_only", "Image-Only Driving Model\n(reads camera pixels)")]
WEATHERS = [("clear", "Clear"), ("fog_1", "Fog 1.0"), ("fog_1.0", "Fog 1.0"), ("rain_0.6", "Rain 0.6"), ("rain_1", "Rain 1.0"), ("rain_1.0", "Rain 1.0")]
OUTCOME = {"success": "success", "crash": "collision", "off_road": "left the road", "timeout": "timeout"}


def find(model, weather_keys):
    for f in VID.glob(f"front_camera__seed5169__{model}__*.mp4"):
        parts = f.stem.split("__")
        if parts[3] in weather_keys:
            return f, parts[4]
    return None, None


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(index, 0), n - 1))
    ok, frame = cap.read()
    cap.release()
    return frame[..., ::-1] if ok else None


def main():
    # table
    lines = ["\\begin{tabular}{@{}lllrr@{}}\\toprule", "Model & Weather & Outcome & Steps & Duration (s)\\\\\\midrule"]
    for wkeys, wlabel in ((("clear",), "Clear"), (("fog_1", "fog_1.0"), "Fog 1.0"), (("rain_0.6",), "Rain 0.6"), (("rain_1", "rain_1.0"), "Rain 1.0")):
        for model, label in MODELS:
            f, outcome = find(model, wkeys)
            if f is None:
                continue
            n = frame_count(f)
            lines.append(f"{label.replace(chr(10), ' ')} & {wlabel} & {OUTCOME.get(outcome, outcome)} & {n} & {n / 10:.1f}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (TAB / "video_outcomes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # montage: four models x two weather conditions x four moments
    moments = [("step 100\n(10 s)", 100), ("step 200\n(20 s)", 200), ("step 240\n(24 s)", 240), ("last frame of\nthe episode", 10 ** 9)]
    rows = [(m, lab, wk, wl) for wk, wl in ((("clear",), "Clear"), (("rain_1", "rain_1.0"), "Rain 1.0")) for m, lab in MODELS]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.4, 11.2))
    for ri, (model, label, wkeys, wlabel) in enumerate(rows):
        f, outcome = find(model, wkeys)
        for ci, (mlabel, idx) in enumerate(moments):
            ax = axes[ri, ci]
            ax.axis("off")
            if f is not None:
                frame = read_frame(f, idx)
                if frame is not None:
                    ax.imshow(cv2.resize(frame, (400, 300)))
            if ri == 0:
                ax.set_title(mlabel, fontsize=7.5)
            if ci == 0 and f is not None:
                ax.text(-0.04, 0.5, f"{label}\n{wlabel}: {OUTCOME.get(outcome, outcome)}", transform=ax.transAxes, ha="right", va="center", fontsize=6.4)
    fig.suptitle("Front-camera frames of scenario 5169: four models, clear weather and heavy rain", fontsize=9, x=0.27, ha="left", y=0.995)
    fig.subplots_adjust(left=0.27, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.04)
    fig.savefig(FIG / "fig_seed5169_front_camera_frames.pdf", bbox_inches="tight", dpi=130)
    fig.savefig(FIG / "fig_seed5169_front_camera_frames.png", bbox_inches="tight", dpi=150)
    print("montage and table written")


if __name__ == "__main__":
    main()
