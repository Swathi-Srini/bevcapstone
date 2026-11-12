#!/bin/bash
# PLAIN front-camera videos (format of bev_policy/outputs/videos/fog_seed1000.mp4), ONE seed (1039), every model, fog 0.5 / rain 0.5 / clear, then fog 1.0 / rain 1.0.
# Two parallel streams (fog | rain+clear). Resumable (--skip-existing). Logs: logs/plain_camera_{A,B}.log  Flag: logs/plain_camera_done.flag  Output: paper/videos_front_camera/
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs paper/videos_front_camera; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
R="$PY -m bev_rl.adaptive.record_camera_plain --out paper/videos_front_camera --skip-existing --seeds 1039"
$R --weather fog:0.5 fog:1.0 > logs/plain_camera_A.log 2>&1 &
$R --weather rain:0.5 clear:0.0 rain:1.0 > logs/plain_camera_B.log 2>&1 &
wait
$PY - <<'PYEOF'
import json
seen={}
for l in open("paper/videos_front_camera/index.jsonl"): r=json.loads(l); seen[r["file"]]=r
rows=sorted(seen.values(), key=lambda r:(r["weather"],r["arm"]))
L=["# Front-camera videos, seed 1039, every model (plain format of bev_policy/outputs/videos/fog_seed1000.mp4: raw 1200x900 front-left frame + weather augmentation, 10 fps real time, one frame per step, no overlays)","",
 "The policies do NOT see these pixels: they drive from the simulator-derived BEV (weather = detector dropout, no weather physics). Outcomes are listed here because the videos carry no text.","",
 "| file | model | weather | outcome | steps | mean speed m/s |","|---|---|---|---|---|---|"]
L+=[f"| {r['file']} | {r['arm']} | {r['weather']} | {r['outcome']} | {r['steps']} | {r['mean_speed']:.2f} |" for r in rows]
L+=["","caut_auto is identical to caut_c0 in clear weather and is shown only in fog/rain. Ablation models' input configuration is inferred from the model catalogue."]
open("paper/videos_front_camera/INDEX.md","w",encoding="utf-8").write("\n".join(L)+"\n"); print(len(rows),"videos indexed")
PYEOF
touch logs/plain_camera_done.flag
