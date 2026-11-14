"""Plain front-camera video of the reinforcement-learning Attempt 4 model (fine-tuned from the proposed model, no weather input), same format as record_camera_plain.py.

    python -m bev_rl.adaptive.record_rl_attempt4_video --seeds 5169 --weather clear:0.0 fog:1.0 --out report/videos_share_raw
The checkpoint has the proposed model's design (1 grid channel with the map overlay, 6 scalars, no extra input), so it is run exactly like ov_split.
"""
import sys
import numpy as np
from bev_rl.adaptive import record_videos

CKPT = "trained_models/04_Reinforcement_Learning_Attempts/Reinforcement_Learning_Attempt_4_Fine_Tuned_Without_Weather_Input.pt"
_orig = record_videos.build


def build(arm, torch, env):
    if arm != "rl_attempt4": return _orig(arm, torch, env)
    from bev_policy.model import BEVScalarPolicy
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    inner = BEVScalarPolicy(int(ck["in_channels"]), 6, int(ck["weather_dim"])); inner.load_state_dict(ck["model_state"]); inner.eval()
    return {"kind": "ov", "net": (lambda b, s_, c: inner(b, s_)), "mu": np.asarray(ck["scalar_mean"], np.float32), "sd": np.asarray(ck["scalar_std"], np.float32)}


record_videos.build = build
from bev_rl.adaptive import record_camera_plain
if __name__ == "__main__":
    sys.argv += ["--arms", "rl_attempt4"]
    record_camera_plain.main()
