"""Are the grids of the 200-episode tests the SAME as the grids the full camera pipeline produces?

The 200-episode tests built the policy's grid from the simulator's vehicle list (with random dropout in weather). The repository's own pipeline builds it from the cameras
(weather filter -> stereo depth + YOLO -> footprints -> grid). This script drives each episode EXACTLY as the 200-episode test did (same policy, same seeds, same dropout stream,
so the trajectory is the one that test produced) and, every --every steps, ALSO builds the camera grid from the same instant with the unmodified teammates' pipeline
(BEVStateAssembler.assemble). It then compares the two grids cell by cell.

Stored per sample: the three grids (simulator list without dropout = 'ideal', simulator list with dropout = what the 200-episode test fed the policy, camera pipeline),
number of vehicles truly in the window, number of camera objects in the window, exact-equality flags, and vehicle-cell mismatch counts.

    python -m bev_rl.adaptive.compare_grids --weather clear:0.0 --seeds 5000-5019 --every 10 --out bev_rl/adaptive/results/grid_comparison
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weather", required=True); ap.add_argument("--seeds", default="5000-5019"); ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--arm", default="ov_split")
    a = ap.parse_args(); kind, level = a.weather.split(":"); w = (kind, float(level)); lo, hi = map(int, a.seeds.split("-"))
    a.out.mkdir(parents=True, exist_ok=True); tag = f"{kind}_{level}"
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import c_auto, keep_prob, prep
    from bev_rl.adaptive.e2e_pipeline import E2EEnv
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.evaluate_n200_all_weather import load_model
    from bev_rl.adaptive.spec_bev import overlay_github_bev
    env = E2EEnv([lo], [("clear", 0.0)], traffic_density=0.05, horizon=800, sample_seed=7)
    net, mu, sd = load_model(a.arm, torch)
    p = keep_prob(*w, env.table); c = c_auto(*w, env.table) if a.arm == "caut_auto" else 0.0
    jl = a.out / f"samples_{tag}.jsonl"; done = set()
    if jl.exists():
        for l in jl.open(): r = json.loads(l); done.add(r["seed"])
    for seed in range(lo, hi + 1):
        if seed in done: continue
        t0 = time.time()
        o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
        step, recs, G = 0, [], {"ideal": [], "sim_dropout": [], "camera": []}
        while True:
            truth = env.truth_objects()
            kept = [x for x in truth if env.drng.random() < p]                                    # exactly the draw of the 200-episode test
            bev_sim = overlay_github_bev(env.asm.build_bev_grid(kept), env.md)                    # what the 200-episode test fed the policy
            if step % a.every == 0 or (len(truth) > 0 and step % 3 == 0):                           # every --every steps, and every 3rd step whenever a vehicle is inside the window
                ideal = overlay_github_bev(env.asm.build_bev_grid(truth), env.md)
                cam = E2EEnv.student_bev(env)                                                     # the full camera pipeline at this instant
                occ = lambda g: (g == 1.0)                                                         # vehicle cells (overlay/route/boundary values are not 1.0)
                n_cam = len([x for x in env.last_objects if -10 <= x.x_right_m <= 10 and -2.5 <= x.y_forward_m <= 17.5])
                inter = int((occ(cam) & occ(ideal)).sum()); uni = int((occ(cam) | occ(ideal)).sum())
                recs.append({"seed": seed, "step": step, "n_truth_in_window": len(truth), "n_kept_by_dropout": len(kept), "n_camera_objects_in_window": n_cam,
                             "cam_equals_ideal": bool(np.array_equal(cam, ideal)), "cam_equals_sim_dropout": bool(np.array_equal(cam, bev_sim)), "sim_dropout_equals_ideal": bool(np.array_equal(bev_sim, ideal)),
                             "cells_vehicle_ideal": int(occ(ideal).sum()), "cells_vehicle_camera": int(occ(cam).sum()), "cells_only_in_ideal": int((occ(ideal) & ~occ(cam)).sum()),
                             "cells_only_in_camera": int((occ(cam) & ~occ(ideal)).sum()), "vehicle_cell_iou": (inter / uni) if uni else 1.0,
                             "cells_differing_total": int((cam != ideal).sum())})
                for k, g in (("ideal", ideal), ("sim_dropout", bev_sim), ("camera", cam)): G[k].append(np.rint(g * 10).astype(np.int8))
            raw = (o[:6] * SCALE).astype(np.float32)
            with torch.no_grad(): act = net(*prep(torch, bev_sim, raw, c, mu, sd)).squeeze(0).numpy()
            o, r, te, tr, info = env.step(act); step += 1
            if te or tr: break
        with jl.open("a") as f:
            for r_ in recs: f.write(json.dumps(r_) + "\n")
        np.savez_compressed(a.out / f"grids_{tag}_s{seed}.npz", **{k: np.stack(v) for k, v in G.items()})
        print(seed, "steps", step, "outcome", info.get("outcome"), "samples", len(recs), "wall_s", round(time.time() - t0), flush=True)
    env.md.close()


if __name__ == "__main__":
    main()
