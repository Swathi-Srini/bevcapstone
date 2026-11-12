"""Collect every final trained model into ONE folder (trained_models/) with descriptive file names.

What this script does (and does not do)
    * COPIES the selected final checkpoint of each model into trained_models/<family>/<descriptive name>.
    * COPIES the training record (log.json / best.json / config.json) of each model into trained_models/training_records/.
    * Writes trained_models/MODEL_MANIFEST.json: for every copied file the original path, the new path, the SHA-256 hash, the size
      and a plain-language description, so that every model can be traced back to the run that produced it.
    * Does NOT move or delete the originals. The evaluation scripts still read bev_rl/adaptive/runs/<run>/best.pt, so the
      run directories stay in place as the training workspace (they also hold every intermediate iteration).

Run from the repository root:
    venv\\Scripts\\python.exe repository_tools\\organise_trained_models.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "trained_models"
RUNS = "bev_rl/adaptive/runs"

# (family folder, new descriptive file name, source path relative to the root, description)
MODELS = [
    # ---------------------------------------------------------------- baselines supplied by the teammates (GitHub repository)
    ("01_Baseline_Models_From_Teammates", "Baseline_Driving_Model_Trained_Without_Traffic.pt", "bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt",
     "Teammates' behaviour-cloning baseline. 5 clear-weather demonstrations of the intelligent driver model, NO other vehicles. Input: camera-derived bird's-eye view (1 channel) + 6 scalars."),
    ("01_Baseline_Models_From_Teammates", "Baseline_Driving_Model_Trained_With_Traffic.pt", "bev_policy/checkpoints/bc_sparse_traffic_5eps.pt",
     "Teammates' behaviour-cloning baseline. 5 demonstrations with sparse traffic (density 0.05). Input as above. Recommended starting point of the reinforcement-learning stage in the repository README."),
    ("02_Superseded_Early_Experiments", "Early_Camera_View_Version2_Model_Clear_Weather_Only.pt", "bev_policy/checkpoints/bcv2_clear_none.pt",
     "Early experiment (2026-11-02): behaviour cloning on the 7-channel camera bird's-eye view version 2, clear weather only. Superseded; the 7-channel pipeline was not used for the final results."),
    ("02_Superseded_Early_Experiments", "Early_Camera_View_Version2_Model_Mixed_Weather_Without_Weather_Input.pt", "bev_policy/checkpoints/bcv2_mixed_none.pt",
     "Early experiment: version-2 bird's-eye view, mixed clear/fog/rain demonstrations, no weather input. Superseded."),
    ("02_Superseded_Early_Experiments", "Early_Camera_View_Version2_Model_Mixed_Weather_With_Weather_Input.pt", "bev_policy/checkpoints/bcv2_mixed_explicit.pt",
     "Early experiment: version-2 bird's-eye view, mixed demonstrations, explicit weather input. Superseded."),
    # ---------------------------------------------------------------- dataset aggregation (DAgger) students
    ("03_Imitation_Trained_Driving_Models", "Proposed_BEV_Driving_Model.pt", f"{RUNS}/ov_split/best.pt",
     "PROPOSED MODEL (code name ov_split; called the headline model in earlier notes). Dataset-aggregation student. Input: bird's-eye view with the route centre line and road boundaries drawn in from the map + 3 scalars (speed, route progress, distance to goal). Lateral offset, heading error and curvature were moved out of the scalars into the map overlay."),
    ("03_Imitation_Trained_Driving_Models", "Ablation_Plain_BEV_With_Six_State_Inputs.pt", f"{RUNS}/base_a/best.pt",
     "Dataset-aggregation student (code name base_a) on the teammates' unmodified 1-channel bird's-eye view + all 6 scalars. The direct dataset-aggregation counterpart of the baseline behaviour-cloning models."),
    ("03_Imitation_Trained_Driving_Models", "Ablation_Map_Overlay_BEV_With_Six_State_Inputs.pt", f"{RUNS}/ov_both/best.pt",
     "Ablation (code name ov_both): map overlay AND all 6 scalars (information duplicated in both inputs)."),
    ("03_Imitation_Trained_Driving_Models", "Ablation_Map_Overlay_BEV_Without_State_Inputs.pt", f"{RUNS}/ov_bev_only/best.pt",
     "Ablation (code name ov_bev_only): map overlay, scalars hidden."),
    ("03_Imitation_Trained_Driving_Models", "Ablation_Plain_BEV_Without_State_Inputs.pt", f"{RUNS}/arch_bev_only/best.pt",
     "Ablation (code name arch_bev_only): unmodified bird's-eye view, scalars hidden."),
    ("03_Imitation_Trained_Driving_Models", "Ablation_Six_State_Inputs_Without_BEV.pt", f"{RUNS}/arch_scalar_only/best.pt",
     "Ablation (code name arch_scalar_only): bird's-eye view replaced by zeros, all 6 scalars visible."),
    ("03_Imitation_Trained_Driving_Models", "Proposed_BEV_Driving_Model_Trained_From_Collision_Aware_Teacher.pt", f"{RUNS}/sparse_overlay_split_safe_teacher/best.pt",
     "Same input design as the headline model, but labelled by the collision-aware teacher (intelligent driver model + predictive braking layer) instead of the stock driver model. Trained at densities 0, 0.05, 0.1."),
    ("03_Imitation_Trained_Driving_Models", "Caution_Study_Caution_Aware_Driving_Model.pt", f"{RUNS}/caut_a/best.pt",
     "Caution study (code name caut_a): proposed model's input design + one extra 'caution' input derived from measured detector reliability; trained with detector dropout and a caution-aware teacher."),
    ("03_Imitation_Trained_Driving_Models", "Caution_Study_Control_Model_Caution_Input_Fixed_To_Zero.pt", f"{RUNS}/ctrl_nocaut/best.pt",
     "Control of the caution study (code name ctrl_nocaut): identical data and architecture, caution input always 0."),
    ("03_Imitation_Trained_Driving_Models", "Weather_Degradation_Trained_Driving_Model.pt", f"{RUNS}/deg_base/best.pt",
     "Code name deg_base. Dataset aggregation where the student sees the calibrated degraded perception (missed detections and wrong ranges measured from the stereo+detector pipeline); labels from the collision-aware teacher on ground truth."),
    ("03_Imitation_Trained_Driving_Models", "Pose_Error_Study_Map_Only_Without_Error_Level_Input.pt", f"{RUNS}/pose_aug_unconditioned/best.pt",
     "Pose-error study (code name pose_aug_unconditioned): trained with random ego-pose error against the map, reliability input fixed to 0 (control)."),
    ("03_Imitation_Trained_Driving_Models", "Pose_Error_Study_Map_Only_With_Error_Level_Input.pt", f"{RUNS}/pose_aug_conditioned/best.pt",
     "Pose-error study (code name pose_aug_conditioned): same, but the pose-error level is given to the network as an input."),
    ("03_Imitation_Trained_Driving_Models", "Pose_Error_Study_Two_Cues_Without_Error_Level_Input.pt", f"{RUNS}/pose_cam_unconditioned/best.pt",
     "Pose-error study, two-cue variant (code name pose_cam_unconditioned): additional noisy lane-detector scalars (lateral offset, heading error); reliability input 0."),
    ("03_Imitation_Trained_Driving_Models", "Pose_Error_Study_Two_Cues_With_Error_Level_Input.pt", f"{RUNS}/pose_cam_conditioned/best.pt",
     "Pose-error study, two-cue variant (code name pose_cam_conditioned): both reliability levels given as input."),
    # ---------------------------------------------------------------- reinforcement learning
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_4_Fine_Tuned_Without_Weather_Input.pt", f"{RUNS}/ppo_plain/best.pt",
     "Code name ppo_plain. Proximal policy optimisation, reward version 3, starting from the degraded-perception student. 300 000 steps, 3 parallel environments, learning rate 1e-4. Validation success was 1.0 (12 episodes) at EVERY checkpoint including step 0, so validation could not discriminate; the saved 'best' is step 300000 only by the tie-break rule (later iterate wins)."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_4_With_Weather_Input_INCOMPLETE_Untrained.pt", f"{RUNS}/ppo_ada/best.pt",
     "Code name ppo_ada. The run DIED after step 0 (process stopped); this file is the untrained start (= the degraded-perception student with a zero-initialised weather branch). Do not report it as a trained model."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_3_Reward_Sweep_Base_Reward.zip", f"{RUNS}/rl_plain/best.zip",
     "Code name rl_plain. PPO (Stable-Baselines3 container) from the headline model, base reward, density 0.2 validation. Reward-combination sweep, 150 000 steps."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Energy_Term.zip", f"{RUNS}/rl_energy/best.zip",
     "Code name rl_energy. As rl_plain + energy penalty."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Safe_Speed_And_Crash_Scaling.zip", f"{RUNS}/rl_safe/best.zip",
     "Code name rl_safe. As rl_plain + visibility-coupled speed limit and risk-scaled crash penalty."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Headway_Term.zip", f"{RUNS}/rl_head/best.zip",
     "Code name rl_head. As rl_plain + headway (gap) term."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_3_Reward_Sweep_All_Reward_Terms.zip", f"{RUNS}/rl_full/best.zip",
     "Code name rl_full. As rl_plain + energy + safe speed + headway + crash scaling."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_1_Privileged_Teacher_Run_C2.zip", "bev_rl/adaptive/checkpoints/C2_final_1.5M.zip",
     "Code name C2. First reinforcement-learning attempt: PPO teacher on privileged simulator observations with an explicit weather input, warm-started from behaviour cloning of the driver model (bc_init), 1.5 M steps, speed cap 12 m/s, crash penalty 50. Reserved-seed evaluation: clear 2/10 success, fog 5/10, rain 9/50; failures were almost all off-road."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_1_Privileged_Teacher_Run_Probe_C.zip", "bev_rl/adaptive/checkpoints/probe_C_final_1M.zip",
     "Code name probe_C. Probe run of the privileged PPO teacher (same set-up as C2, 1 M steps, speed cap 12 m/s). Reserved-seed evaluation: clear 3/10 success, fog 3/10, rain 10/50; failures almost all off-road."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_1_Initialisation_Behaviour_Cloned_Privileged_Expert.pt", "bev_rl/adaptive/checkpoints/bc_init.pt",
     "Code name bc_init. Behaviour-cloned privileged expert used to warm-start the PPO teacher (avoids the stand-still optimum)."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_1_Privileged_Teacher_Run_C3.zip", "archive/removed_scripts_and_early_runs_2026-11-05/runs/C3/best.zip",
     "Early PPO attempt (code name C3): as C2 but speed cap 10 m/s and learning rate 5e-5, 1.5 M steps planned. Best validation checkpoint was at 100 000 steps (validation success 0.4). Kept for the failure record."),
    ("04_Reinforcement_Learning_Attempts", "Reinforcement_Learning_Attempt_2_Fine_Tuned_Proposed_Model_With_Weather_Input.zip", "archive/removed_scripts_and_early_runs_2026-11-05/runs/pilot_cond/best.zip",
     "Reinforcement-learning attempt 2 (code name pilot_cond): PPO fine-tuning of the proposed model (map overlay, three state inputs) with an added weather input, detector dropout at the measured retention, learning rate 5e-5, 3 parallel environments, speed cap 10 m/s, crash penalty 50. Planned 480 000 steps; the run ended after 120 000. Validation success 0.675 at step 0, 0.65 at 60 000, 0.675 at 120 000 (40 episodes)."),
    # ---------------------------------------------------------------- pixel policy and perception
    ("05_Image_Only_Model", "Image_Only_Driving_Model_Front_Camera.pt", f"{RUNS}/vcam/best.pt",
     "Code name vcam. Convolutional network that sees ONLY a front-camera image (no bird's-eye view, no scalars). Control experiment showing that image corruptions hurt a pixel policy."),
    ("06_Third_Party_Perception_Weights", "Third_Party_YOLOv8_Nano_Object_Detector_Weights.pt", "weights/yolov8n.pt",
     "Pretrained YOLO version 8 nano object detector (Ultralytics, third-party weights) used by the teammates' perception pipeline. Not trained in this project."),
]

RECORDS = {   # training record files to copy next to the models
    "ov_split": "Proposed_BEV_Driving_Model", "base_a": "Ablation_Plain_BEV_With_Six_State_Inputs", "ov_both": "Ablation_Map_Overlay_BEV_With_Six_State_Inputs", "ov_bev_only": "Ablation_Map_Overlay_BEV_Without_State_Inputs",
    "arch_bev_only": "Ablation_Plain_BEV_Without_State_Inputs", "arch_scalar_only": "Ablation_Six_State_Inputs_Without_BEV", "sparse_overlay_split_safe_teacher": "Proposed_BEV_Driving_Model_Trained_From_Collision_Aware_Teacher",
    "caut_a": "Caution_Study_Caution_Aware_Driving_Model", "ctrl_nocaut": "Caution_Study_Control_Model_Caution_Input_Fixed_To_Zero", "deg_base": "Weather_Degradation_Trained_Driving_Model", "pose_aug_unconditioned": "Pose_Error_Study_Map_Only_Without_Error_Level_Input",
    "pose_aug_conditioned": "Pose_Error_Study_Map_Only_With_Error_Level_Input", "pose_cam_unconditioned": "Pose_Error_Study_Two_Cues_Without_Error_Level_Input", "pose_cam_conditioned": "Pose_Error_Study_Two_Cues_With_Error_Level_Input",
    "ppo_plain": "Reinforcement_Learning_Attempt_4_Fine_Tuned_Without_Weather_Input", "ppo_ada": "Reinforcement_Learning_Attempt_4_With_Weather_Input_INCOMPLETE_Untrained", "rl_plain": "Reinforcement_Learning_Attempt_3_Reward_Sweep_Base_Reward",
    "rl_energy": "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Energy_Term", "rl_safe": "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Safe_Speed_And_Crash_Scaling", "rl_head": "Reinforcement_Learning_Attempt_3_Reward_Sweep_Plus_Headway_Term", "rl_full": "Reinforcement_Learning_Attempt_3_Reward_Sweep_All_Reward_Terms", "vcam": "Image_Only_Driving_Model_Front_Camera",
    "C2": "Reinforcement_Learning_Attempt_1_Privileged_Teacher_Run_C2",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(exist_ok=True)
    manifest, missing = [], []
    for family, new_name, src_rel, description in MODELS:
        src = ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        out_dir = DEST / family
        out_dir.mkdir(exist_ok=True)
        dst = out_dir / new_name
        shutil.copy2(src, dst)
        manifest.append({"file": f"{family}/{new_name}", "original_path": src_rel, "sha256": sha256(dst), "size_kilobytes": round(dst.stat().st_size / 1024, 1), "description": description})
    rec_dir = DEST / "Training_Records"
    rec_dir.mkdir(exist_ok=True)
    for run, nice in RECORDS.items():
        for fname in ("log.json", "best.json", "config.json", "train_log.json"):
            src = ROOT / RUNS / run / fname
            if src.exists():
                shutil.copy2(src, rec_dir / f"{nice}__{fname}")
    (DEST / "MODEL_MANIFEST.json").write_text(json.dumps({"models": manifest, "missing_sources": missing}, indent=1), encoding="utf-8")
    print(f"copied {len(manifest)} models; missing: {missing}")


if __name__ == "__main__":
    main()
