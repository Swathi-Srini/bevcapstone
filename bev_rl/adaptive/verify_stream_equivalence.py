"""EQUIVALENCE HARNESS (diagnostic only; runs inside the one-worker timing check, never in the full pass).  NEW FILE.

Question: is the frame / feature / object / grid set produced by the retention + stream code BIT-FOR-BIT the one the live student receives?
Method: at a diagnostic step t, for each of the 11 conditions
    (1) MY path   : measure_retention_camera.process_condition(full=True)      on the frozen capture `raw`
    (2) LIVE path : the genuine camera_dagger.make_env(...).student_bev()      (full path, light=False) with env.weather / env.step_count set to (condition, t),
                    observed through PASS-THROUGH recorders only (no behaviour change): a recorder that replaces env.sensor and stores the exact front-left frame handed to sensor.update,
                    and wrappers around assembler.front_stereo_depth, assembler.run_yolo and env.asm.objects_from_detections that store hashes / a repr snapshot and return the original result unchanged.
    Compared with exact equality: front-left frame (np.array_equal + SHA-256), front-right frame, the four YOLO input frames, the depth map, the 7-feature vector (byte-for-byte),
    the raw object list (repr snapshot, taken before the builder runs), and the final grid (np.array_equal).
Further checks: SHA-256 of the raw capture before and after all conditions; global RNG state restored after my path; NEGATIVE CONTROL: the 'light' (2-call) path frame must DIFFER from the live frame
(it is the non-equivalent path that the user's review caught), so the comparison is shown to be able to fail.
The harness patches module/instance attributes only inside a try/finally and restores them; no file is modified.  Any mismatch -> ok=False -> the caller aborts and reports.
"""
import hashlib
import time

import numpy as np


class _Recorder:
    """Stands in for env.sensor: remembers the exact object passed to update() and returns 0.0 (the value is unused here)."""

    def __init__(self): self.frame = None

    def update(self, f):
        self.frame = f
        return 0.0


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def check_step(ctx, t, raw, gseed):
    """-> report dict {t, ok, mismatches[], conditions{}, ...}.  `raw` = the frozen capture (read-only arrays) of step t."""
    from bev_rl.adaptive.measure_retention_camera import CONDITIONS, FL, FR, Stages, cname, hash_arr, preserve_rng, process_condition, weathered
    env, A = ctx.env, ctx.A; null = Stages(); mism, per = [], {}; t_start = time.perf_counter()
    h_before = {n: hash_arr(v) for n, v in raw.items()}
    rng0 = np.random.get_state()
    saved = {"weather": env.weather, "step_count": env.step_count, "sensor": env.sensor, "depth": A.front_stereo_depth, "yolo": A.run_yolo}
    had_inst_obj = "objects_from_detections" in env.asm.__dict__; orig_obj = env.asm.objects_from_detections
    rec = _Recorder(); log = {"depth": None, "yolo": [], "objs": None}

    def depth_w(l, r, m):
        d = saved["depth"](l, r, m); log["depth"] = (_h(l), _h(r), _h(d)); return d

    def yolo_w(model, img):
        log["yolo"].append(_h(img)); return saved["yolo"](model, img)

    def obj_w(*args, **kw):
        o = orig_obj(*args, **kw); log["objs"] = [repr(x) for x in o]; return o

    try:
        A.front_stereo_depth, A.run_yolo, env.asm.objects_from_detections, env.sensor = depth_w, yolo_w, obj_w, rec
        for kind, level in CONDITIONS:
            name = cname(kind, level); bad = []
            with preserve_rng():                                                                   # the same context manager the collector uses around every measurement block
                mine = process_condition(ctx, raw, kind, level, t, gseed, True, st=null)
                rng_touched = not _rng_equal(np.random.get_state(), rng0)                          # weather re-seeds the global RNG inside the block (expected True for fog/rain)
            rng_restored = _rng_equal(np.random.get_state(), rng0)                                 # ... and the block must leave it exactly as it found it
            log.update(depth=None, yolo=[], objs=None); env.weather = (kind, level); env.step_count = t
            grid_live = env.student_bev(); np.random.set_state(rng0)
            live_fl = rec.frame
            if not _same(live_fl, mine["frames"][FL]): bad.append("front_left frame")
            if live_fl is not None and _h(live_fl) != _h(mine["frames"][FL]): bad.append("front_left frame sha256")
            if log["depth"] is None or log["depth"] != (_h(mine["frames"][FL]), _h(mine["frames"][FR]), _h(mine["depth"])): bad.append("stereo inputs / depth map")
            if log["yolo"] != [_h(mine["frames"][n]) for n in A.YOLO_CAMERAS]: bad.append("YOLO input frames (4 views, in order)")
            if ctx.features(live_fl).tobytes() != mine["f"].tobytes(): bad.append("7-feature vector")
            if log["objs"] != mine["objs_repr"]: bad.append("raw objects")
            if not _same(grid_live, mine["grid"]): bad.append("final grid")
            if not rng_restored: bad.append("RNG restore")
            if (kind, level) != ("clear", 0.0) and not _same(mine["frames"][FL], weathered(ctx, raw, kind, level, gseed, t)[FL]) : bad.append("my path not reproducible (second call differs)")
            np.random.set_state(rng0)
            ctrl = None
            if kind != "clear":                                                                        # NEGATIVE CONTROL: the 2-call 'light' path (camera_dagger L52-57) must give a DIFFERENT front-left frame
                np.random.seed(20261114 + int(gseed) * 100003 + t); ctx.apply_weather(raw[FL], kind, level); light_fl, _ = ctx.apply_sync(raw[FL], raw[FR], kind, level)
                ctrl = not _same(light_fl, live_fl); np.random.set_state(rng0)
            per[name] = {"ok": not bad, "bad": bad, "negative_control_light_path_differs": ctrl, "rng_touched_inside_block": rng_touched, "n_raw_objects": len(mine["objs_repr"])}
            mism += [f"{name}: {b}" for b in bad]
            del mine, grid_live
    finally:
        A.front_stereo_depth, A.run_yolo, env.sensor, env.weather, env.step_count = saved["depth"], saved["yolo"], saved["sensor"], saved["weather"], saved["step_count"]
        if had_inst_obj: env.asm.objects_from_detections = orig_obj
        else: env.asm.__dict__.pop("objects_from_detections", None)
        np.random.set_state(rng0)
    if {n: hash_arr(v) for n, v in raw.items()} != h_before: mism.append("raw capture changed during the 11 conditions")
    ctrl_vals = [v["negative_control_light_path_differs"] for v in per.values() if v["negative_control_light_path_differs"] is not None]
    return {"t": t, "ok": not mism, "mismatches": mism, "conditions": per, "raw_sha256_unchanged": "raw capture changed during the 11 conditions" not in mism,
            "negative_control": {"conditions_tested": len(ctrl_vals), "light_path_differs_in": int(sum(ctrl_vals)), "discriminating": bool(any(ctrl_vals))}, "seconds": round(time.perf_counter() - t_start, 1)}


def _rng_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
