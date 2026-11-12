"""SMOKE TEST ONLY (not a result): wiring of degraded_env / run_episode / AdaptPPOEnv / PPO / export_actor with a FAKE depth-error model written to a scratch folder.
    python -m bev_rl.adaptive._smoke_degraded_pipeline
"""
import json
import time
from pathlib import Path

import numpy as np

SCR = Path("bev_rl/adaptive/results/_smoke"); SCR.mkdir(parents=True, exist_ok=True)
rel = [-0.3, -0.1, 0.0, 0.05, 0.2, 0.5]
cell = lambda pd, pv: {"n": [20, 20, 20], "p_detect": [pd] * 3, "p_valid": [pv] * 3, "rel_err": [rel] * 3, "pooled_rel_err": rel}
fake = {"range_bins_m": [0.0, 6.0, 12.0, 30.0], "cells": {"none:0.0": cell(0.95, 0.95), "fog:0.6": cell(0.9, 0.9), "fog:0.8": cell(0.6, 0.7), "fog:1.0": cell(0.3, 0.5),
                                                          "rain:0.3": cell(0.7, 0.8), "rain:0.6": cell(0.2, 0.5), "rain:1.0": cell(0.05, 0.3)}}
(SCR / "fake_model.json").write_text(json.dumps(fake))


def main():
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import degraded_env as D
    from bev_rl.adaptive.env import SCALE
    t0 = time.time()
    dm = D.DepthModel(SCR / "fake_model.json"); rng = np.random.default_rng(0)
    keys = {dm._key("fog", 0.7, rng) for _ in range(200)}; print("interp keys for fog 0.7:", sorted(keys)); assert keys <= {"fog:0.6", "fog:0.8"}
    print("clear key:", dm._key("clear", 0.0, rng), "| fog 1.0 sample:", dm.sample("fog", 1.0, 8.0, rng))
    # 1. degraded env + run_episode (student acts) + collect with SafeTeacher
    env = D.DegradedEnv(range(2000, 2005), [("fog", 1.0)], mode="dropout", traffic_density=0.1, horizon=120)
    env.mode = "calibrated"; env.depth = dm
    net = BEVScalarPolicy(1, 6, 4); net.eval(); mu = np.zeros(6, np.float32); sd = np.ones(6, np.float32)
    B, S, W, Y, info = D.run_episode(env, net, mu, sd, torch, 2001, ("fog", 1.0), True, beta=0.5, noise=0.1, rng=rng, collect=True)
    print("collect episode:", len(B), "samples", B[0].shape, B[0].dtype, S[0].shape, W[0], Y[0], "| outcome", info["outcome"], "min_gap", round(info["min_gap_m"], 2))
    assert B[0].shape == (64, 64) and len(B) == len(S) == len(W) == len(Y)
    *_, info2 = D.run_episode(env, net, mu, sd, torch, 2002, ("rain", 1.0), True)
    print("student episode outcome:", info2["outcome"], "steps", info2["steps"])
    n_obj = [len(env.degrade(env.truth_objects())) for _ in range(5)]; print("objects truth/degraded counts sample:", len(env.truth_objects()), n_obj)
    env.close()
    # 2. PPO env reward path
    from bev_rl.adaptive.ppo_adaptive import AdaptPPOEnv, export_actor
    e = AdaptPPOEnv(range(2000, 2005), [("fog", 1.0)], mu, sd, True, mode="dropout", traffic_density=0.1, horizon=80); e.mode = "calibrated"; e.depth = dm
    o, _ = e.reset(options={"scenario_seed": 2003, "weather": ("fog", 1.0)}); print("obs keys:", {k: v.shape for k, v in o.items()})
    R = 0.0
    for _ in range(40):
        o, r, te, tr, i = e.step(np.array([0.0, 0.3], np.float32)); R += r
        if te or tr: break
    print("40-step reward sum:", round(R, 2), "info", i["outcome"], round(i["min_gap_m"], 1)); e.close()
    # 3. short PPO + export equivalence
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from bev_rl.adaptive.ppo_bev import base_to_weather_ckpt
    from bev_rl.policy import BEVActorCritic, init_from_bc
    base = BEVScalarPolicy(1, 6, 0); ck = {"model_state": base.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist()}
    for variant in ("plain", "adaptive"):
        wd = 4 if variant == "adaptive" else 0
        venv = DummyVecEnv([lambda: AdaptPPOEnv(range(2000, 2005), [("fog", 1.0), ("clear", 0.0)], mu, sd, wd == 4, mode="dropout", traffic_density=0.05, horizon=60)])
        m = PPO(BEVActorCritic, venv, learning_rate=1e-4, n_steps=64, batch_size=32, n_epochs=2, device="cpu", verbose=0, policy_kwargs={"log_std_init": -2.0})
        init_from_bc(m.policy, base_to_weather_ckpt(ck) if wd else {"in_channels": 1, "weather_dim": 0, "model_state": ck["model_state"]})
        m.learn(128)
        obs = venv.reset(); act_sb3 = m.predict(obs, deterministic=True)[0]
        export_actor(m.policy, mu, sd, wd, SCR / f"export_{variant}.pt", torch)
        bc = BEVScalarPolicy(1, 6, wd); bc.load_state_dict(torch.load(SCR / f"export_{variant}.pt", weights_only=True)["model_state"]); bc.eval()
        with torch.no_grad():
            args = [torch.from_numpy(obs["bev"]), torch.from_numpy(obs["scalar"])] + ([torch.from_numpy(obs["weather"])] if wd else [])
            act_bc = bc(*args).numpy()
        print(variant, "SB3 vs exported max |diff|:", float(np.abs(act_sb3 - act_bc).max())); assert np.allclose(act_sb3, act_bc, atol=1e-5)
        venv.close()
    # 4. adaptive net at step 0 equals the plain net (net surgery keeps the function)
    plain, ada = BEVScalarPolicy(1, 6, 0), BEVScalarPolicy(1, 6, 4); ada.load_state_dict(base_to_weather_ckpt(ck)["model_state"]); plain.load_state_dict(ck["model_state"]); plain.eval(); ada.eval()
    b = torch.randn(2, 1, 64, 64); s_ = torch.randn(2, 6); w_ = torch.rand(2, 4)
    print("surgery max |diff|:", float((plain(b, s_) - ada(b, s_, w_)).abs().max())); assert torch.allclose(plain(b, s_), ada(b, s_, w_), atol=1e-6)
    print("SMOKE OK", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
