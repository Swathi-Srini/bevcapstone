"""PILOT: drive the ORIGINAL proposed model (ov_split, unchanged checkpoint) from the grid built by the full camera pipeline + the new CameraGridBuilder,
and measure, per weather condition, how that grid compares with the simulator-derived ground-truth grid.

Per simulation step (10 Hz, every step goes through the pipeline):
    5 cameras -> GitHub weather filter -> front SGBM depth + YOLOv8n on 4 views -> BEVStateAssembler.objects_from_detections (unmodified)
    -> CameraGridBuilder (centre/footprint/duplicate corrections + HD-map layers) -> 64x64 grid -> policy.
    The simulator's vehicle list is used ONLY for measurement (never as an input to the policy or to the builder).

Measured per step against the simulator ground truth (vehicles inside the grid window):
    grid agreement (exact equality; cell overlap of vehicle cells), detection recall (truth vehicles with a matched object within 3 m), recall among
    vehicles inside some camera's field of view, position error of matched objects (forward / lateral / Euclidean), duplicate rate (extra objects on
    an already-matched vehicle), false-positive rate, detections dropped by the depth module (invalid depth).  Everything is computed twice: for the
    ORIGINAL assembler objects (surface points, bbox-inflated footprints) and for the CORRECTED objects, so the effect of the builder is visible.

    python -m bev_rl.adaptive.pilot_camera_builder --worker 1 --out bev_rl/adaptive/results/pilot_camera_builder --seeds 5000-5009
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WEATHERS = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
MATCH_RADIUS_M = 3.0
YAWS = {"front": 0.0, "left": -math.pi / 2, "right": math.pi / 2, "rear": math.pi}


def match(truth_xy, obj_xy, radius=MATCH_RADIUS_M):
    """Greedy one-to-one matching by distance. -> (pairs [(ti, oi, dx, dy)], n_dup, n_fp)."""
    pairs, used_t, used_o = [], set(), set()
    cand = sorted((math.hypot(t[0] - o[0], t[1] - o[1]), ti, oi) for ti, t in enumerate(truth_xy) for oi, o in enumerate(obj_xy))
    for d, ti, oi in cand:
        if d > radius: break
        if ti in used_t or oi in used_o: continue
        used_t.add(ti); used_o.add(oi); pairs.append((ti, oi, obj_xy[oi][0] - truth_xy[ti][0], obj_xy[oi][1] - truth_xy[ti][1]))
    n_dup = n_fp = 0
    for oi, o in enumerate(obj_xy):
        if oi in used_o: continue
        if any(math.hypot(t[0] - o[0], t[1] - o[1]) <= radius for t in truth_xy): n_dup += 1      # extra object on a vehicle that is already matched
        else: n_fp += 1                                                                                # object with no vehicle nearby
    return pairs, n_dup, n_fp


def in_view(x, y, half_fov, cam_range):
    if math.hypot(x, y) > cam_range: return False
    a = math.atan2(x, y)
    return any(abs((a - yaw + math.pi) % (2 * math.pi) - math.pi) <= half_fov for yaw in YAWS.values())


class PilotEnv:
    """Wraps E2EEnv: policy grid from camera pipeline + CameraGridBuilder; measures against the simulator truth."""

    def __init__(self, **kw):
        from bev_rl.adaptive.e2e_pipeline import E2EEnv
        from bev_state.camera_grid_builder import CameraGridBuilder
        self.env = E2EEnv([5000], [("clear", 0.0)], traffic_density=kw.get("density", 0.05), horizon=kw.get("horizon", 800), sample_seed=7)
        self.builder = CameraGridBuilder(self.env.asm)
        import bev_state.assembler as A
        self.A = A

    def grid_and_metrics(self):
        from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
        from weather.weather_utils import apply_weather
        from bev_rl.adaptive.spec_bev import overlay_github_bev
        env, A = self.env, self.A
        raw = {n: frame_from_sensor(env.md, n) for n in CAMERA_RIGS}
        kind, level = env.weather
        if kind == "clear" or level <= 0: frames = raw
        else:
            np.random.seed(20261114 + int(env.md.engine.global_seed) * 100003 + env.step_count)
            frames = {n: apply_weather(f, kind, level) for n, f in raw.items()}
            frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw["front_left_camera"], raw["front_right_camera"], kind, level)
        env.step_count += 1
        self.last_front = frames["front_left_camera"]                                  # weathered front frame, kept only so an evaluation can record a video
        depth = A.front_stereo_depth(frames["front_left_camera"], frames["front_right_camera"], env.asm.matcher)
        dets = {n: A.run_yolo(env.yolo, frames[n]) for n in A.YOLO_CAMERAS}
        raw_objs = env.asm.objects_from_detections(dets, depth)                        # ORIGINAL assembler objects
        corr_objs = self.builder.corrected_objects(raw_objs)                           # corrected objects
        grid = self.builder.build(raw_objs, env.md)                                     # grid fed to the policy
        # ---------------- measurement against the simulator (never fed back)
        truth = env.truth_objects()
        ideal = overlay_github_bev(env.asm.build_bev_grid(truth), env.md)
        raw_grid = overlay_github_bev(env.asm.build_bev_grid(raw_objs), env.md)         # grid the ORIGINAL builder would have produced
        T = [(t.x_right_m, t.y_forward_m) for t in truth]
        half_fov, cam_rng = math.radians(self.A.CAM_FOV / 2.0), env.asm.config.camera_range_m
        vis = [in_view(x, y, half_fov, cam_rng) for x, y in T]
        n_det = sum(len(v) for v in dets.values())
        rec = {"n_truth": len(T), "n_truth_in_view": sum(vis), "n_det": n_det, "n_obj_raw": len(raw_objs), "n_obj_corr": len(corr_objs),
               "n_dropped_invalid_depth": n_det - len(raw_objs),
               "methods": {m: sum(1 for o in raw_objs if o.depth_method == m) for m in {o.depth_method for o in raw_objs}}}
        for tag, objs, g in (("raw", raw_objs, raw_grid), ("corr", corr_objs, grid)):
            O = [(o.x_right_m, o.y_forward_m) for o in objs if -10.0 <= o.x_right_m <= 10.0 and -2.5 <= o.y_forward_m <= 17.5]      # only objects that can appear in the grid window
            pairs, n_dup, n_fp = match(T, O)
            rec[f"{tag}_matched"] = len(pairs); rec[f"{tag}_matched_in_view"] = sum(1 for ti, *_ in pairs if vis[ti]); rec[f"{tag}_dup"] = n_dup; rec[f"{tag}_fp"] = n_fp
            rec[f"{tag}_dx"] = [round(p[2], 3) for p in pairs]; rec[f"{tag}_dy"] = [round(p[3], 3) for p in pairs]
            occ_c, occ_i = (g == 1.0), (ideal == 1.0); inter, uni = int((occ_c & occ_i).sum()), int((occ_c | occ_i).sum())
            rec[f"{tag}_grid_equal"] = bool(np.array_equal(g, ideal)); rec[f"{tag}_cells_differ"] = int((g != ideal).sum())
            rec[f"{tag}_veh_cells_inter"], rec[f"{tag}_veh_cells_union"] = inter, uni
        return grid, rec, (ideal, raw_grid)


def run(a):
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import prep
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.evaluate_n200_all_weather import load_model, row
    lo, hi = map(int, a.seeds.split("-"))
    (a.out / "episodes").mkdir(parents=True, exist_ok=True); (a.out / "claims").mkdir(exist_ok=True); (a.out / "grids").mkdir(exist_ok=True)
    pe = PilotEnv(density=a.density, horizon=a.horizon); env = pe.env
    net, mu, sd = load_model("ov_split", torch)                                       # the ORIGINAL proposed-model checkpoint, unchanged
    for seed in range(lo, hi + 1):
        for kind, level in WEATHERS:
            name = f"{kind}_{level}__s{seed}"; res = a.out / "episodes" / f"{name}.json"
            if res.exists(): continue
            try: fd = os.open(a.out / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            except FileExistsError: continue
            t0 = time.time(); w = (kind, level)
            o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
            steps, G = [], {"ideal": [], "original_builder": [], "new_builder": []}; step = 0
            while True:
                grid, rec, (ideal, raw_grid) = pe.grid_and_metrics()
                steps.append(rec)
                if step % 10 == 0 or (rec["n_truth"] > 0 and step % 3 == 0):
                    for k, g in (("ideal", ideal), ("original_builder", raw_grid), ("new_builder", grid)): G[k].append(np.rint(g * 10).astype(np.int8))
                raw = (o[:6] * SCALE).astype(np.float32)
                with torch.no_grad(): act = net(*prep(torch, grid, raw, 0.0, mu, sd)).squeeze(0).numpy()
                o, r, te, tr, info = env.step(act); step += 1
                if te or tr: break
            out = row(seed, info); out.update({"weather": f"{kind}:{level}", "wall_s": round(time.time() - t0), "steps_log": steps})
            np.savez_compressed(a.out / "grids" / f"{name}.npz", **{k: np.stack(v) for k, v in G.items()})
            res.write_text(json.dumps(out)); print(json.dumps({k: v for k, v in out.items() if k != "steps_log"}), flush=True)
    env.md.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seeds", default="5000-5009"); ap.add_argument("--worker", type=int, default=1)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--horizon", type=int, default=800)
    run(ap.parse_args())
