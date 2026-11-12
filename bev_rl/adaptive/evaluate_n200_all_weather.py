"""Final n=200 ALL-WEATHER evaluation, ONE harness for all six presented models (records energy for every model).
Seeds 5000-5199 (fresh block; never used for training, validation or checkpoint selection), density 0.05, horizon 800, weather = detector dropout only
(keep-prob from results/retention_table.json). Identical seeds for every model AND an identical per-episode dropout stream (rng seeded s*7+1), so paired tests are valid.

Models: repo_bc_sparse, repo_bc_clear (GitHub BEV, 6 scalars, no overlay), ov_split, ctrl_nocaut (c fixed 0), caut_c0 (caut_a, c=0), caut_auto (caut_a, c=1-keep_prob).
Resumable: results/n200_all_weather/<arm>__<weather>.json per cell (a cell is skipped if it exists); <cell>.partial.json is written every 20 episodes.
Reuse (valid because clear weather has keep_prob 1, identical env settings): clear cells of repo_bc_sparse, repo_bc_clear, ov_split, ctrl_nocaut are converted from
results/final_200seed/*.json; caut_auto@clear is copied from caut_c0@clear (c = 0 in clear, same network => identical by construction).
    python -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --arms ov_split repo_bc_sparse ...
"""
import argparse, json
from pathlib import Path
import numpy as np

WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
ALL = ["ov_split", "repo_bc_sparse", "repo_bc_clear", "ctrl_nocaut", "caut_c0", "caut_auto", "safe_student"]   # safe_student = runs/sparse_overlay_split_safe_teacher (SafeTeacher labels, same input design as ov_split)
REPO = {"repo_bc_sparse": "bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", "repo_bc_clear": "bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt"}
CLEAR_FILE = {"ov_split": "base_ovsplit", "repo_bc_sparse": "repo_bc_sparse", "repo_bc_clear": "repo_bc_clear", "ctrl_nocaut": "ctrl_nocaut"}
FINAL = Path("bev_rl/adaptive/results/final_200seed")


def wname(w): return f"{w[0]}_{w[1]}"


def summarise(arm, w, a, rows, note=None):
    oc = [r["outcome"] for r in rows]; n = len(oc)
    return {"arm": arm, "weather": f"{w[0]}:{w[1]}", "seed0": a.seed0, "n": n, "density": a.density, "note": note,
            **{k: oc.count(k) / n for k in ("success", "crash", "off_road", "timeout")},
            "mean_speed": float(np.mean([r["speed"] for r in rows])), "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])),
            "route": float(np.mean([r["route"] for r in rows])), "episodes": rows}


def row(s, i):
    return {"seed": s, "outcome": i["outcome"], "speed": float(i["mean_speed"]), "wh_per_km": float(i["energy_wh"] / max(i["distance_m"] / 1000.0, 1e-3)),
            "route": float(i["route_completion"]), "steps": int(i["steps"]) if "steps" in i else None}


def load_model(arm, torch):
    from bev_policy.model import BEVScalarPolicy
    if arm in REPO:
        ck = torch.load(REPO[arm], map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
    elif arm in ("ov_split", "safe_student"):
        ck = torch.load("bev_rl/adaptive/runs/ov_split/best.pt" if arm == "ov_split" else "bev_rl/adaptive/runs/sparse_overlay_split_safe_teacher/best.pt", map_location="cpu", weights_only=True)
        inner = BEVScalarPolicy(1, 6, 0); inner.load_state_dict(ck["model_state"]); inner.eval(); net = lambda b, s_, c: inner(b, s_)   # ignores caution
    else:
        run = "ctrl_nocaut" if arm == "ctrl_nocaut" else "caut_a"
        ck = torch.load(f"bev_rl/adaptive/runs/{run}/best.pt", map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
    return net, np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def run_cell(arm, w, a, env, net, mu, sd, torch, part):
    from bev_rl.adaptive.caution import keep_prob, c_auto, episode
    from bev_rl.adaptive.env import SCALE
    rows = json.loads(part.read_text()) if part.exists() else []; done = {r["seed"] for r in rows}
    p = keep_prob(*w, env.table); c = c_auto(*w, env.table) if arm == "caut_auto" else 0.0
    for s in range(a.seed0, a.seed0 + a.n):
        if s in done: continue
        if arm in REPO:                                                         # raw GitHub BEV, no overlay, all 6 scalars (as the repo trained it)
            o, _ = env.reset(options={"scenario_seed": s, "weather": w}); env.drng = np.random.default_rng(s * 7 + 1)
            while True:
                bev = env.asm.build_bev_grid([x for x in env.truth_objects() if env.drng.random() < p])
                sc = ((o[:6] * SCALE - mu) / sd)[None].astype(np.float32)
                with torch.no_grad():
                    act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
                o, r, te, tr, i = env.step(act)
                if te or tr: break
        else:                                                                   # overlay BEV + KEEP mask + caution input, via caution.episode
            env.drng = np.random.default_rng(s * 7 + 1)
            *_, i = episode(env, net, mu, sd, torch, s, w, c)
        rows.append(row(s, i)); print(arm, wname(w), rows[-1], flush=True)
        if len(rows) % 20 == 0: part.write_text(json.dumps(rows))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--arms", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--out", default="bev_rl/adaptive/results/n200_all_weather"); ap.add_argument("--no-reuse", action="store_true"); ap.add_argument("--horizon", type=int, default=800, help="episode step limit (default 800 = 80 s)")
    ap.add_argument("--only-weather", nargs="+", default=None, choices=["clear", "fog_1.0", "rain_0.6", "rain_1.0"], help="restrict to these weather cells")
    a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from bev_rl.adaptive.caution import CautionEnv
    for arm in a.arms:
        todo = [w for w in WEATHER if not (OUT / f"{arm}__{wname(w)}.json").exists() and (a.only_weather is None or wname(w) in a.only_weather or (w[0] == "clear" and "clear" in a.only_weather))]
        net = env = None
        for w in todo:
            cell = OUT / f"{arm}__{wname(w)}.json"
            if w[0] == "clear" and not a.no_reuse and arm in CLEAR_FILE and (FINAL / f"{CLEAR_FILE[arm]}.json").exists() and a.n == 200 and a.seed0 == 5000:
                src = json.loads((FINAL / f"{CLEAR_FILE[arm]}.json").read_text())["episodes"]
                cell.write_text(json.dumps(summarise(arm, w, a, src, "reused from results/final_200seed (same env settings; clear weather => no dropout)"), indent=1)); print("reused", arm, "clear"); continue
            if arm == "caut_auto" and w[0] == "clear" and (OUT / f"caut_c0__{wname(w)}.json").exists() and not a.no_reuse:
                src = json.loads((OUT / f"caut_c0__{wname(w)}.json").read_text())["episodes"]
                cell.write_text(json.dumps(summarise(arm, w, a, src, "copied from caut_c0@clear: c=0 in clear weather and same network => identical by construction"), indent=1)); print("copied caut_auto clear"); continue
            if net is None:
                net, mu, sd = load_model(arm, torch)
                env = CautionEnv(range(a.seed0, a.seed0 + a.n), WEATHER, dropout=True, traffic_density=a.density, horizon=a.horizon, sample_seed=7)
            rows = run_cell(arm, w, a, env, net, mu, sd, torch, OUT / f"{arm}__{wname(w)}.partial.json")
            cell.write_text(json.dumps(summarise(arm, w, a, rows), indent=1)); (OUT / f"{arm}__{wname(w)}.partial.json").unlink(missing_ok=True)
            print("SUMMARY", arm, wname(w), {k: v for k, v in summarise(arm, w, a, rows).items() if k != "episodes"}, flush=True)
        if env is not None: env.close()


if __name__ == "__main__":
    main()
