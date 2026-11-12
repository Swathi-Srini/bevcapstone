"""Checks that the evaluation env passes weather into the actor BEV: same scene/seed/actions, clear vs fog vs rain."""
import numpy as np, torch
from bev_rl.factory import make_v2_env
from bev_rl.policy import load_bc_checkpoint

ck = load_bc_checkpoint("bev_policy/checkpoints/bcv2_mixed_explicit.pt")
conds = [("clear", 0.0), ("fog", 0.6), ("rain", 0.6)]
env = make_v2_env([5001], ck["scalar_mean"], ck["scalar_std"], weather_conditions=conds, weather_input="explicit",
                  horizon=20, sequential=True, cam_scale=0.5)
out = {}
try:
    for k, l in conds:
        obs, info = env.reset(options={"weather": (k, l)})
        frames = []
        for _ in range(10):
            obs, *_r = env.step(np.array([0.0, 0.3], dtype=np.float32))
            frames.append(obs["bev"].copy())
        out[k] = (np.stack(frames), obs["weather"].copy(), info)
finally:
    env.close()
for k in ("fog", "rain"):
    d = np.abs(out[k][0] - out["clear"][0]).mean(axis=(0, 2, 3))
    print(k, "weather obs", out[k][1], "mean |BEV diff vs clear| per channel", np.round(d, 4))
print("clear weather obs", out["clear"][1])
