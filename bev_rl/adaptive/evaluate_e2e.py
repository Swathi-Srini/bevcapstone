"""END-TO-END evaluation: the policy's grid comes from the cameras (weather filter -> stereo depth + YOLO -> footprints -> grid), see e2e_pipeline.py.

One episode takes ~15-25 minutes (about 1.6 s per step on this CPU-only machine), so the work is a PRIORITISED QUEUE of (seed block, cell, seed) tasks that several workers pull from;
every finished episode is saved on its own, so partial results are always usable and the queue can be stopped or extended at any time.
Same scenario seeds, density 0.05, horizon 800 and caution rule as the 200-episode simulator-list tests, so the two can be compared episode by episode.

    python -m bev_rl.adaptive.evaluate_e2e --worker 1 --out bev_rl/adaptive/results/e2e --seeds 5000-5059 --block 10
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# priority of cells inside one block of seeds: the central comparisons first
CELLS = [("ov_split", "clear", 0.0), ("ov_split", "rain", 1.0), ("caut_auto", "clear", 0.0), ("caut_auto", "rain", 1.0),
         ("repo_bc_clear", "clear", 0.0), ("repo_bc_clear", "rain", 1.0), ("repo_bc_sparse", "clear", 0.0), ("repo_bc_sparse", "rain", 1.0),
         ("ov_split", "fog", 1.0), ("caut_auto", "fog", 1.0), ("ov_split", "rain", 0.6), ("caut_auto", "rain", 0.6),
         ("repo_bc_clear", "fog", 1.0), ("repo_bc_sparse", "fog", 1.0), ("repo_bc_clear", "rain", 0.6), ("repo_bc_sparse", "rain", 0.6)]


def tag(arm, kind, level): return f"{arm}__{kind}_{level}"


def tasks(lo, hi, block, cells=None):
    out = []
    for b0 in range(lo, hi + 1, block):
        for arm, kind, level in (cells or CELLS):
            for s in range(b0, min(b0 + block, hi + 1)):
                out.append((arm, kind, level, s))
    return out


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seeds", default="5000-5059"); ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--worker", type=int, default=1); ap.add_argument("--cells", nargs="*", default=None, help="arm:kind:level, in priority order (default: the built-in list)"); ap.add_argument("--horizon", type=int, default=800); ap.add_argument("--density", type=float, default=0.05)
    a = ap.parse_args(); lo, hi = map(int, a.seeds.split("-"))
    (a.out / "episodes").mkdir(parents=True, exist_ok=True); (a.out / "claims").mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import c_auto, keep_prob, prep
    from bev_rl.adaptive.e2e_pipeline import E2EEnv
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.evaluate_n200_all_weather import load_model, row
    env = E2EEnv([lo], [("clear", 0.0)], traffic_density=a.density, horizon=a.horizon, sample_seed=7)
    models = {}
    cells = [(x.split(":")[0], x.split(":")[1], float(x.split(":")[2])) for x in a.cells] if a.cells else None
    for arm, kind, level, seed in tasks(lo, hi, a.block, cells):
        name = f"{tag(arm, kind, level)}__s{seed}"
        res = a.out / "episodes" / f"{name}.json"
        if res.exists() or not claim(a.out / "claims" / name): continue
        if arm not in models: models[arm] = load_model(arm, torch)
        net, mu, sd = models[arm]; w = (kind, level); t0 = time.time()
        o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
        c = c_auto(kind, level, env.table) if arm == "caut_auto" else 0.0   # caution is supplied from the weather label, as in the simulator-list tests
        det_sum = truth_sum = 0; n = 0; env.t_pipe, env.n_pipe = 0.0, 0
        while True:
            raw = (o[:6] * SCALE).astype(np.float32)
            if arm.startswith("repo_bc"):
                bev = env.raw_bev(); sc = ((raw - mu) / sd)[None].astype(np.float32)
                with torch.no_grad(): act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
            else:
                bev = env.student_bev()
                with torch.no_grad(): act = net(*prep(torch, bev, raw, c, mu, sd)).squeeze(0).numpy()
            if n % 5 == 0:                                                                 # in-loop detection rate: camera detections vs vehicles truly inside the grid window
                truth_sum += len(env.truth_objects()); det_sum += len([x for x in env.last_objects if -10 <= x.x_right_m <= 10 and -2.5 <= x.y_forward_m <= 17.5]) if not arm.startswith("repo_bc") else 0
            o, r, te, tr, info = env.step(act); n += 1
            if te or tr: break
        rec = row(seed, info); rec.update({"arm": arm, "weather": f"{kind}:{level}", "wall_s": round(time.time() - t0, 1), "pipeline_s_per_step": round(env.t_pipe / max(1, env.n_pipe), 2),
                                           "sampled_truth_vehicles": truth_sum, "sampled_camera_detections": det_sum})
        res.write_text(json.dumps(rec)); print(json.dumps(rec), flush=True)
    env.md.close()


if __name__ == "__main__":
    main()
