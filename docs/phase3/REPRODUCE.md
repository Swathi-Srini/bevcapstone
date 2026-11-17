# Reproduce

Run from the repository root with the project environment (`requirements.txt`; MetaDrive, PyTorch, Ultralytics YOLOv8n weights). The camera pipeline is slow (about 1.7 s per step per episode on the development laptop), so the full evaluation takes days. Each stage is resumable: finished outputs are skipped and claim files prevent two workers taking the same episode.

```bash
# 1. Model B (oracle-caution reference), camera-pipeline DAgger
python -m bev_rl.adaptive.camera_dagger_severity freeze
python -m bev_rl.adaptive.camera_dagger_severity collect  --round 0 --worker 1
python -m bev_rl.adaptive.camera_dagger_severity train    --round 0
#   ... repeat collect / train for rounds 1 to 3, then
python -m bev_rl.adaptive.camera_dagger_severity select   --round 3

# 2. Teacher validation with the identity caution (privileged, 50 episodes)
python -m bev_rl.adaptive.validate_teacher_identity_caution --weather fog:1.0 --seeds 3100-3149

# 3. Severity estimator (fit, then held-out evaluation; the evaluation never fits)
python -m bev_rl.adaptive.severity_estimator fit  --out bev_rl/adaptive/results/severity_estimator/estimator.json
python -m bev_rl.adaptive.severity_estimator eval --model bev_rl/adaptive/results/severity_estimator/estimator.json

# 4. Paired A / C evaluation (20 seeds shown; the extension uses seeds 5220-5239 and --out results/severity_eval_ext)
python -m bev_rl.adaptive.evaluate_severity_policies --out bev_rl/adaptive/results/severity_eval \
    --tasks "clear:0.0@5200-5219;fog:1.0@5200-5219;rain:1.0@5200-5219" --arms A,C --worker 1

# 5. Analysis (read-only on the episode files)
python -m bev_rl.adaptive.analyze_severity_eval --episodes bev_rl/adaptive/results/severity_eval_20seed_snapshot \
    --out bev_rl/adaptive/results/severity_eval_20seed_analysis --label "20 paired seeds (pilot)"

# 6. Verify the snapshot against its manifest
cd bev_rl/adaptive/results/severity_eval_20seed_snapshot/episodes && sha256sum -c ../MANIFEST_sha256.txt

# 7. Paper figures and architecture diagrams
python report/paper_severity/make_figures.py
python report/make_architecture_v3.py

# 8. Optional: depth / detection / BEV stills for one scene under several weathers (needs about 2.5 GB free)
python -m bev_rl.adaptive.capture_pipeline_stills --seed 5219 --step 195 --out report/paper_severity/figures/capture
```

The driver script `bev_rl/adaptive/queue_severity_pipeline_v3.sh` chains the evaluation, the snapshot with verification, and the extension. The paper is built with `pdflatex` / `bibtex` in `report/paper_severity/` (see its `main.tex`).

Not included in the repository (see `FILE_MAP.md`): the DAgger training data (`.npz`, ignored by `.gitignore`), videos, logs, the YOLO weights file, and the partial extension results.
