#!/bin/bash
# Front-camera recordings, ONE seed (1039: oncoming vehicle drifts into the ego lane), every model, clear / fog 0.5 / rain 0.5.  Resumable (--skip-existing).
# Log: logs/camera_videos.log   Flag: logs/camera_videos_done.flag   Output: paper/videos_camera/<model>__seed1039__<weather>.mp4 (+ INDEX.md)
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs paper/videos_camera; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
C="$PY -m bev_rl.adaptive.record_front_camera --out paper/videos_camera --skip-existing --every 3 --seeds 1039"
# batch 1: the models that were presented/compared + rule-based references
$C --weather clear:0.0 fog:0.5 rain:0.5 --arms repo_bc_sparse repo_bc_clear base_a ov_split ctrl_nocaut caut_c0 caut_auto safe_student stock_idm safe_teacher
# batch 2: earlier DAgger ablations (input configuration inferred from the catalogue)
$C --weather clear:0.0 fog:0.5 rain:0.5 --arms arch_bev_only arch_scalar_only ov_both ov_bev_only
$PY - <<'PYEOF'
import json
seen={}
for l in open("paper/videos_camera/index.jsonl"): r=json.loads(l); seen[r["file"]]=r
rows=sorted(seen.values(), key=lambda r:(r["weather"],r["arm"]))
L=["# Front-camera recordings (seed 1039), every model, clear / fog 0.5 / rain 0.5","",
 "Left: teammates' front-left camera frame with the GitHub weather augmentation + the YOLO detections on that frame. Right: the BEV the policy receives (bright = model input, steel-blue/dim = display-only road geometry).",
 "The policies never see the camera pixels: they drive from the simulator-derived BEV; weather = detector dropout from the measured retention table (no weather physics). 3.3 frames/s (one frame per 3 sim steps), real-time playback.","",
 "| file | model | weather | outcome | steps | mean speed m/s |","|---|---|---|---|---|---|"]
L+=[f"| {r['file']} | {r['arm']} | {r['weather']} | {r['outcome']} | {r['steps']} | {r['mean_speed']:.2f} |" for r in rows]
L+=["","caut_auto is identical to caut_c0 in clear weather and is shown only in fog/rain."]
open("paper/videos_camera/INDEX.md","w",encoding="utf-8").write("\n".join(L)+"\n"); print(len(rows),"camera videos indexed")
PYEOF
touch logs/camera_videos_done.flag
