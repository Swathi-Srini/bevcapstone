#!/bin/bash
# Record every model on paired seeds -> paper/videos/<model>__seed<N>__<weather>.mp4 (+ INDEX.md). Resumable (--skip-existing). Log: logs/videos.log, flag: logs/videos_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs paper/videos; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
V="$PY -m bev_rl.adaptive.record_videos --out paper/videos --skip-existing"
# 1) clear weather, 4 shared seeds (1012 ordinary, 1020 repo model leaves the road, 1039 oncoming sideswipe, 1040 roundabout merge); caut_auto == caut_c0 in clear so it appears in the weather block
$V --seeds 1012 1020 1039 1040 --weather clear:0.0 --arms repo_bc_sparse repo_bc_clear base_a ov_split ctrl_nocaut caut_c0 safe_student stock_idm safe_teacher
# 2) earlier DAgger ablations, 2 seeds
$V --seeds 1012 1020 --weather clear:0.0 --arms arch_bev_only arch_scalar_only ov_both ov_bev_only
# 3) detector-dropout weather block (fog 1.0, rain 1.0): the weather-relevant models
$V --seeds 1020 1039 --weather fog:1.0 rain:1.0 --arms repo_bc_sparse base_a ov_split ctrl_nocaut caut_c0 caut_auto
$PY - <<'PYEOF'
import json
rows=[json.loads(l) for l in open("paper/videos/index.jsonl")]; seen={}
for r in rows: seen[r["file"]]=r
rows=sorted(seen.values(), key=lambda r:(r["weather"],r["seed"],r["arm"]))
L=["# Videos of every model (10 fps = real time; left: top-down world, right: the BEV the policy receives)","",
   "Environment: sparse traffic (density 0.05), horizon 800, detector dropout for fog/rain from the measured retention table (NO weather physics). Identical scenario seed across models in each group.","",
   "| file | model | seed | weather | outcome | steps | mean speed m/s |","|---|---|---|---|---|---|---|"]
L+= [f"| {r['file']} | {r['arm']} | {r['seed']} | {r['weather']} | {r['outcome']} | {r['steps']} | {r['mean_speed']:.2f} |" for r in rows]
L+=["","Notes: caut_auto is identical to caut_c0 in clear weather (caution c = 0), so it is shown only in fog/rain. Rule-based references (stock_idm, safe_teacher) see ground truth. Ablation models' input configuration is inferred from the model catalogue."]
open("paper/videos/INDEX.md","w",encoding="utf-8").write("\n".join(L)+"\n"); print(len(rows),"videos indexed")
PYEOF
touch logs/videos_done.flag
