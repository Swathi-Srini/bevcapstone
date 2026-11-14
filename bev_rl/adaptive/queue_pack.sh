#!/bin/bash
# DELIVERABLE PACK (queued): ORIGINAL ("200-episode", ground-truth-trained) ov_split and caution checkpoints on the FULL camera pipeline.
#   fog 1.0, seeds 5100-5109, both models (20 episodes) + front-camera/BEV videos for seed 5100 in clear, fog 1.0 and rain 1.0 (4 more episodes). Density 0.05, horizon 800.
# Starts only when OV-Split has finished (logs/camera_training_ov.done) and no camera worker is running. Never touches OV-Split files.
# Starts 3 workers; then a 4-WORKER HEALTH TEST: a 4th worker is added only if memory/paging are healthy, and is removed again (its unfinished claims released) if
# paging, free memory or per-step speed become unhealthy. Thresholds: hard page faults avg < 300/s, free memory min > 250 MB, per-step time < 2.25 s (baseline 1.96 s with 3 workers).
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
OUT=bev_rl/adaptive/results/camera_eval_originals; L=logs/camera_pack.log; TASKS="fog:1.0@5100-5109;clear:0.0@5100;rain:1.0@5100"
mkdir -p logs "$OUT"; echo "$(date '+%F %T') pack queued" >> $L
health() { powershell -NoProfile -Command '$s=Get-Counter "\Memory\Pages/sec","\Memory\Available MBytes" -SampleInterval 5 -MaxSamples 12; $p=($s | % { $_.CounterSamples[0].CookedValue } | Measure-Object -Average).Average; $m=($s | % { $_.CounterSamples[1].CookedValue } | Measure-Object -Minimum).Minimum; "$([math]::Round($p)) $([math]::Round($m))"' | tr -d '\r'; }
nproc_cam() { powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger (collect|validate)|evaluate_camera_pipeline" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r'; }
until [ -f logs/camera_training_ov.done ]; do sleep 60; done
until [ "$(nproc_cam)" = "0" ]; do sleep 60; done
echo "$(date '+%F %T') OV-Split finished; starting 3 workers" >> $L
launch() { CAM_MAX_WORKERS=4 $PY -m bev_rl.adaptive.evaluate_camera_pipeline --worker $1 --out $OUT --tasks "$TASKS" --video-seeds 5100 >> logs/camera_pack_w$1.log 2>&1 & }
for W in 1 2 3; do launch $W; sleep 20; done
sleep 240; H=$(health); echo "$(date '+%F %T') health with 3 workers (pages/s, min free MB): $H" >> $L; P=${H% *}; M=${H#* }
if [ "$P" -lt 300 ] && [ "$M" -gt 250 ]; then
  T4=$(date +%s); launch 4; echo "$(date '+%F %T') 4th worker started (test)" >> $L; sleep 420; H=$(health); echo "$(date '+%F %T') health with 4 workers: $H" >> $L; P=${H% *}; M=${H#* }
  bad=0; [ "$P" -ge 300 ] || [ "$M" -le 250 ] && bad=1
  for chk in 1 2; do  # per-step time of episodes finished since the 4th worker started (needs >= 2)
    sleep 1200; SPS=$($PY - <<EOF
import json,glob,os
t4=$T4; v=[]
for f in glob.glob("$OUT/episodes/*.json"):
    if os.path.getmtime(f)>t4+300:
        d=json.load(open(f)); v.append(d["wall_s"]/max(1,len(d["steps_log"])))
print(round(sum(v)/len(v),2) if len(v)>=2 else 0)
EOF
); echo "$(date '+%F %T') per-step time since 4th worker (s): $SPS" >> $L
    awk "BEGIN{exit !($SPS>2.25)}" && bad=1; [ "$SPS" != "0" ] && break
  done
  if [ "$bad" = "1" ]; then
    echo "$(date '+%F %T') UNHEALTHY -> removing 4th worker" >> $L
    powershell -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "evaluate_camera_pipeline" -and $_.CommandLine -match "--worker 4" -and $_.CommandLine -notmatch "Get-Cim" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'
    sleep 3; for n in $(cat "$OUT/claims/_w4.log" 2>/dev/null); do [ -f "$OUT/episodes/$n.json" ] || rm -f "$OUT/claims/$n"; done
    echo "$(date '+%F %T') 4th worker removed; unfinished claims released" >> $L
  else echo "$(date '+%F %T') 4 workers HEALTHY" >> $L; fi
else echo "$(date '+%F %T') 4th worker NOT started (3-worker health not good enough)" >> $L; fi
wait
touch logs/camera_pack.done; echo "$(date '+%F %T') pack finished" >> $L
