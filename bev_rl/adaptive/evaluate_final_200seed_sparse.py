"""Final large-n evaluation on a FRESH seed block (default 5000-5199; never used for training, validation or checkpoint selection).
Clear weather, traffic density 0.05, horizon 800, identical seeds for every arm (needed for paired McNemar / bootstrap).
Per episode: outcome, mean speed, Wh/km, route completion. Resumable: an arm whose results/final_200seed/<arm>.json exists is skipped.
Arms: ctrl_nocaut | base_ovsplit | repo_bc_sparse | repo_bc_clear | stock_idm | safe_teacher
    python -m bev_rl.adaptive.evaluate_final_200seed_sparse --n 200 --seed0 5000 --arms ctrl_nocaut base_ovsplit ...
"""
import argparse, json
from pathlib import Path
import numpy as np

OUT = Path("bev_rl/adaptive/results/final_200seed")
REPO = {"repo_bc_sparse": "bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", "repo_bc_clear": "bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt"}
ALL = ["ctrl_nocaut", "base_ovsplit", "repo_bc_sparse", "repo_bc_clear", "stock_idm", "safe_teacher"]


def row(s, i):
    return {"seed": s, "outcome": i["outcome"], "speed": float(i["mean_speed"]), "wh_per_km": float(i["energy_wh"] / max(i["distance_m"] / 1000.0, 1e-3)),
            "route": float(i["route_completion"]), "steps": int(i["steps"]) if "steps" in i else None}


def run_arm(arm, a, torch):
    from bev_policy.model import BEVScalarPolicy
    seeds = range(a.seed0, a.seed0 + a.n); W = ("clear", 0.0); rows = []
    if arm in ("ctrl_nocaut", "base_ovsplit"):
        from bev_rl.adaptive.caution import CautionEnv, episode
        run = "ctrl_nocaut" if arm == "ctrl_nocaut" else "ov_split"
        ck = torch.load(f"bev_rl/adaptive/runs/{run}/best.pt", map_location="cpu", weights_only=True)
        if run == "ov_split":
            inner = BEVScalarPolicy(1, 6, 0); inner.load_state_dict(ck["model_state"]); inner.eval(); net = lambda b, s_, c: inner(b, s_)
        else:
            net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.array(ck["scalar_mean"], np.float32), np.array(ck["scalar_std"], np.float32)
        env = CautionEnv(seeds, [W], dropout=True, traffic_density=a.density, horizon=800, sample_seed=7)
        for s in seeds:
            *_, i = episode(env, net, mu, sd, torch, s, W, 0.0); rows.append(row(s, i)); print(arm, rows[-1], flush=True)
    elif arm in REPO:
        from bev_rl.adaptive.caution import CautionEnv
        from bev_rl.adaptive.env import SCALE
        ck = torch.load(REPO[arm], map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
        env = CautionEnv(seeds, [W], dropout=True, traffic_density=a.density, horizon=800, sample_seed=7)
        for s in seeds:
            o, _ = env.reset(options={"scenario_seed": s, "weather": W}); env.drng = np.random.default_rng(s * 7 + 1)
            while True:
                bev = env.asm.build_bev_grid(env.truth_objects())          # raw GitHub BEV, no overlay, clear weather => keep_prob 1
                sc = ((o[:6] * SCALE - mu) / sd)[None].astype(np.float32)
                with torch.no_grad():
                    act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
                o, r, te, tr, i = env.step(act)
                if te or tr: break
            rows.append(row(s, i)); print(arm, rows[-1], flush=True)
    else:                                                                 # teachers (no learned weights)
        from metadrive.policy.idm_policy import IDMPolicy
        from bev_rl.adaptive.env import AdaptiveEnv
        from bev_rl.adaptive.safe_teacher import SafeTeacher
        env = AdaptiveEnv(seeds, [W], weather_input="zeroed", horizon=800, traffic_density=a.density)
        for s in seeds:
            env.reset(options={"scenario_seed": s, "weather": W})
            idm = IDMPolicy(env.md.agent, env.md.engine.global_seed)
            if arm == "stock_idm":
                while True:
                    o, r, te, tr, i = env.step(np.clip(np.asarray(idm.act(), np.float32)[:2], -1, 1))
                    if te or tr: break
            else:
                idm.enable_lane_change = False; pol = SafeTeacher(idm, t_soft_s=2.4, t_hard_s=1.3)
                while True:
                    o, r, te, tr, i = env.step(pol.act(env))
                    if te or tr: break
            rows.append(row(s, i)); print(arm, rows[-1], flush=True)
    env.close()
    oc = [r["outcome"] for r in rows]
    res = {"arm": arm, "seed0": a.seed0, "n": len(rows), "density": a.density,
           **{k: oc.count(k) / len(oc) for k in ("success", "crash", "off_road", "timeout")},
           "mean_speed": float(np.mean([r["speed"] for r in rows])), "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])),
           "route": float(np.mean([r["route"] for r in rows])), "episodes": rows}
    (OUT / f"{arm}.json").write_text(json.dumps(res, indent=1))
    print("SUMMARY", {k: v for k, v in res.items() if k != "episodes"}, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--arms", nargs="+", default=ALL, choices=ALL)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    import torch
    for arm in a.arms:
        if (OUT / f"{arm}.json").exists(): print("skip (exists)", arm); continue
        run_arm(arm, a, torch)

if __name__ == "__main__":
    main()
