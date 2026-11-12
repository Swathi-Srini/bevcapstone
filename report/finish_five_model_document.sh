#!/bin/bash
# Waits for the three image-only weather runs, then regenerates the five-model tables and figures and rebuilds all PDFs.
cd /d/bev_Fresh/bevcapstone || exit 1
R=bev_rl/adaptive/results/vanilla_camera
until [ -f $R/vcam__fog1__n200_s5000.json ] && [ -f $R/vcam__rain0.6__n200_s5000.json ] && [ -f $R/vcam__rain1__n200_s5000.json ]; do sleep 120; done
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe report/make_five_model_comparison.py > logs/five_model_regenerate.log 2>&1
powershell -File report/build_documents.ps1 > logs/five_model_rebuild.log 2>&1
touch logs/five_model_done.flag
