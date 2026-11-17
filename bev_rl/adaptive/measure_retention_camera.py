"""CAMERA-PIPELINE RETENTION MEASUREMENT + SENSOR FEATURE STREAMS (paired, same-frame) for the Live Caution camera model.  NEW FILE (rewrite of the never-run first draft).
No existing project file is modified; existing modules are only IMPORTED (camera_dagger, pilot_camera_builder, visibility_sensor.features, the YOLO+weather module, the assembler, the frozen builder).

Approved methodology (2026-11-15, revised by the user's decisions of the same day; see PREREGISTRATION_model_c_live_caution.md)
    Teacher A ('safe_g4.0') drives CLEAR episodes: density 0.05, horizon 600, ACTION NOISE 0.0 (weather augmentation of the live pipeline is unchanged: fresh noise per step from the live seed formula).
    Step t = number of env.step calls so far.  t = 0 is NEVER captured (the camera rig is mounted after the reset render; precautionary skip).  Capture happens at t = 1 .. steps-1.
    The weather noise seed is exactly the live one:  np.random.seed(20261114 + global_seed*100003 + t).

    A. RETENTION frames (training seeds 9000-9029; reserve 9030-9039 only with --allow-reserve): t % 5 == 0 and t >= 5.  The five raw frames are captured ONCE, frozen (read-only arrays + SHA-256 before and after),
       and replayed through ALL 11 conditions with the FULL live pipeline: weather filter -> stereo depth -> YOLO (4 views) -> objects_from_detections -> CameraGridBuilder (corrected_objects + build).
         retention frame     >= 1 truth vehicle in the window x[-10,10] y[-2.5,17.5], at most 40 per episode
         vehicle-free frame  0 truth vehicles, at most 5 per episode (ANALYSIS ONLY: valid-depth share, YOLO confidence, false-positive objects)
       The sensor features of every condition of these frames are stored (clear and weather on the SAME frames).  No stride-5 'feature-only' records exist any more (they are redundant).
    B. SENSOR STREAMS (scope S-B-stride2).  streams/s{seed}_stream.npz holds RAW image features only (never predictions):
         held-out seeds 9040-9049 : clear at EVERY step t >= 1; the 10 weather conditions only at EVEN steps t = 2, 4, ... (WEATHER_STREAM_STRIDE = 2), from the final weather-processed front-left image
                                    (full live weather block: 5 apply_weather calls + synchronized stereo call, identical to the live image at those steps).  Odd-step weather cells are marked
                                    by_design_unsampled (not missing, not zero).  Between sampled steps nothing is claimed about the live smoothed output.
         training seeds           : every step t >= 1, CLEAR only
    Nothing here is an input to driving: the simulator's truth vehicles are used for measurement only.

Collection modes
    collect   --worker 1 --out <dir> --seeds "9000-9029;9040-9049"       (one worker; workers 2-3 additionally need logs/retention_camera_multiworker.approved, a file only the user creates)
    status    --out <dir>
    validate  --out <dir> [--expect "..."]                                (exit 0 only if every check passes; the queue creates the done flag only then)
    timing check: --diagnostics (only into a folder named retention_camera_timing_check, one TRAINING seed): equivalence harness against the genuine live student_bev at t in {5,50,200,400}
                  (verify_stream_equivalence.py), stream benchmark (timing only, never saved as data), memory sampler CSV, per-stage timing report.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

SCHEMA = "retention_camera/2"
CONDITIONS = [("clear", 0.0), ("fog", 0.3), ("fog", 0.45), ("fog", 0.6), ("fog", 0.8), ("fog", 1.0),
              ("rain", 0.2), ("rain", 0.3), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
PROBES = {("fog", 0.45), ("rain", 0.3)}                       # never enter the table or the sensor fit
TRAIN_SEEDS, RESERVE_SEEDS, HELDOUT_SEEDS = range(9000, 9030), range(9030, 9040), range(9040, 9050)
GROUP_SIZE = 6                                                  # five groups of six consecutive training seeds
FRAME_STRIDE, MAX_RETENTION_FRAMES, MAX_VEHICLE_FREE_FRAMES = 5, 40, 5
DENSITY, HORIZON, TEACHER, ACTION_NOISE, MAX_WORKERS = 0.05, 600, "safe_g4.0", 0.0, 3
X_LO, X_HI, Y_LO, Y_HI = -10.0, 10.0, -2.5, 17.5
NOISE_BASE, NOISE_MULT = 20261114, 100003                       # identical to the live weather-noise seed formula (camera_dagger / e2e_pipeline)
FL, FR = "front_left_camera", "front_right_camera"
WEATHER_STREAM_STRIDE = 2            # held-out weather streams: the full live weather block runs only at even t (t = 2, 4, ...); clear features stay at EVERY step (scope S-B-stride2, user decision 2026-11-15)
DIAG_STEPS = (5, 50, 200, 400)
FEATURE_NAMES = ["log_lapvar", "grad", "contrast", "dark", "bright", "sat", "edge"]
DEFAULT_SEEDS = "9000-9029;9040-9049"
TIMING_DIR_NAME = "retention_camera_timing_check"
ROOT = Path("bev_rl/adaptive/results/retention_camera")
MULTIWORKER_FLAG = Path("logs/retention_camera_multiworker.approved")
PREREG = "bev_rl/adaptive/PREREGISTRATION_model_c_live_caution.md"
SOURCE_FILES = ["bev_state/camera_grid_builder.py", "bev_state/assembler.py", "bev_state/__init__.py", "weights/yolov8n.pt",
                "YOLO+weather module/manual_drive_stereo_yolo_weather.py", "YOLO+weather module/weather/weather_utils.py", "YOLO+weather module/yolo/yolo_utils.py",
                "bev_policy/runtime.py", "bev_rl/adaptive/visibility_sensor.py", "bev_rl/adaptive/pilot_camera_builder.py", "bev_rl/adaptive/camera_dagger.py",
                "bev_rl/adaptive/teacher_eval.py", "bev_rl/adaptive/safe_teacher.py", "bev_rl/adaptive/e2e_pipeline.py", "bev_rl/adaptive/env.py", "bev_rl/adaptive/caution.py",
                "bev_rl/adaptive/spec_bev.py", "bev_rl/adaptive/results/camera_dagger_final/FROZEN_MANIFEST.json", "bev_rl/adaptive/runs/ov_split/best.pt", "bev_rl/adaptive/runs/caut_a/best.pt",
                "bev_rl/adaptive/measure_retention_camera.py", "bev_rl/adaptive/retention_camera_summary.py", "bev_rl/adaptive/verify_stream_equivalence.py",
                "bev_rl/adaptive/queue_retention_camera.sh", PREREG]


class EquivalenceMismatch(RuntimeError):
    pass


class ImmutabilityError(RuntimeError):
    pass


def cname(kind, level):
    return f"{kind}:{float(level)}"


COND_NAMES = [cname(*c) for c in CONDITIONS]


def partition(seed):
    return "train" if seed in TRAIN_SEEDS else "reserve" if seed in RESERVE_SEEDS else "heldout" if seed in HELDOUT_SEEDS else None


def group_of(seed):
    return (seed - TRAIN_SEEDS.start) // GROUP_SIZE if seed in TRAIN_SEEDS else None


def parse_seeds(text):
    out = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if not part: continue
        lo, hi = part.split("-") if "-" in part else (part, part)
        out += list(range(int(lo), int(hi) + 1))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def write_atomic(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}"); tmp.write_text(json.dumps(obj, separators=(",", ":"))); os.replace(tmp, path)


def write_npz_atomic(path, **arrays):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


@contextmanager
def preserve_rng():
    """Global numpy RNG is re-seeded by the live weather formula at every measurement; save/restore it so the clear-weather driving never depends on the measurements."""
    s = np.random.get_state()
    try: yield
    finally: np.random.set_state(s)


# ----------------------------------------------------------------------------------------------------------------------------------------- instrumentation
class Stages:
    """Wall-clock per pipeline stage + the current stage label (read by the memory sampler) + the first stage in which an exception occurred."""

    def __init__(self):
        self.t, self.n, self.cur, self.fail = defaultdict(float), defaultdict(int), "idle", None

    def reset(self):
        self.t.clear(); self.n.clear(); self.fail = None

    @contextmanager
    def __call__(self, name):
        prev, t0 = self.cur, time.perf_counter(); self.cur = name
        try: yield
        except BaseException:
            if self.fail is None: self.fail = name
            raise
        finally:
            self.t[name] += time.perf_counter() - t0; self.n[name] += 1; self.cur = prev


class MemSampler(threading.Thread):
    """In-process sampler: this process' working set / private bytes and the SYSTEM available memory + page-file use, tagged with the current stage."""

    def __init__(self, stages, interval):
        super().__init__(daemon=True)
        import psutil
        self.ps, self.proc, self.st, self.interval, self.stop_evt = psutil, psutil.Process(), stages, float(interval), threading.Event()
        self.rows, self.lock = [], threading.Lock(); self.restart()

    def restart(self):
        with self.lock:
            self.rows = []; self.peak_ws = self.peak_priv = 0; self.min_avail = float("inf"); self.by_stage = {}; self.peak_swap = 0.0

    def sample(self):
        mi, vm, sw = self.proc.memory_info(), self.ps.virtual_memory(), self.ps.swap_memory()
        rss, priv = mi.rss / 2**20, getattr(mi, "private", mi.rss) / 2**20; avail = vm.available / 2**20; stage = self.st.cur
        with self.lock:
            self.peak_ws, self.peak_priv, self.min_avail, self.peak_swap = max(self.peak_ws, rss), max(self.peak_priv, priv), min(self.min_avail, avail), max(self.peak_swap, sw.percent)
            d = self.by_stage.setdefault(stage, {"peak_working_set_mb": 0.0, "min_system_available_mb": float("inf")})
            d["peak_working_set_mb"] = max(d["peak_working_set_mb"], rss); d["min_system_available_mb"] = min(d["min_system_available_mb"], avail)
            self.rows.append((round(time.time(), 2), stage, round(rss, 1), round(priv, 1), round(avail, 1), sw.percent))

    def run(self):
        while not self.stop_evt.wait(self.interval): self.sample()

    def summary(self):
        self.sample()
        mi = self.proc.memory_info()
        with self.lock:
            return {"peak_working_set_mb": round(self.peak_ws, 1), "peak_private_mb": round(self.peak_priv, 1), "os_lifetime_peak_working_set_mb": round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1),
                    "min_system_available_mb": round(self.min_avail, 1), "peak_swap_percent": self.peak_swap, "samples": len(self.rows), "interval_s": self.interval,
                    "by_stage": {k: {a: round(b, 1) for a, b in v.items()} for k, v in self.by_stage.items()}}

    def write_csv(self, path):
        with self.lock, open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["unix_time", "stage", "working_set_mb", "private_mb", "system_available_mb", "swap_percent"]); w.writerows(self.rows)


class Ctx:
    """The live objects of one worker: env (camera_dagger.make_env = the genuine BuilderEnv), builder, assembler module, YOLO+weather functions, instrumentation."""

    def __init__(self, interval=2.0):
        from bev_rl.adaptive.camera_dagger import make_env
        import bev_state.assembler as A
        from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
        from weather.weather_utils import apply_weather
        from bev_rl.adaptive.visibility_sensor import features
        from bev_rl.adaptive.pilot_camera_builder import match
        self.make_env, self.A, self.CAMERA_RIGS = make_env, A, CAMERA_RIGS
        self.apply_sync, self.frame_from_sensor, self.apply_weather, self.features, self.match = apply_synchronized_stereo_weather, frame_from_sensor, apply_weather, features, match
        self.st = Stages(); self.mem = MemSampler(self.st, interval); self.env = None; self.t = 0; self.cond = None; self.equiv = []; self.bench = []
        self.new_env()

    def new_env(self):
        if self.env is not None:
            try: self.env.md.close()
            except Exception: pass
        self.env = self.make_env(DENSITY, HORIZON, 7)               # camera_dagger.make_env: E2EEnv subclass with the live student_bev and CameraGridBuilder v2
        self.builder = self.env.builder

    def capture(self, names):
        """Fresh capture of the named cameras; arrays are frozen (read-only) so nothing downstream can modify them in place."""
        raw = {n: self.frame_from_sensor(self.env.md, n) for n in names}
        for a in raw.values(): a.flags.writeable = False
        return raw


def weathered(ctx, raw, kind, level, global_seed, t):
    """EXACTLY the live frame set (camera_dagger.BuilderEnv.student_bev, full path): clear -> the raw dict itself; else seed, apply_weather on all 5 cameras in CAMERA_RIGS order, then the synchronized pair overwrites
    front_left / front_right.  `raw` is never modified (new arrays are returned)."""
    if kind == "clear" or level <= 0: return raw
    np.random.seed(NOISE_BASE + int(global_seed) * NOISE_MULT + t)
    frames = {n: ctx.apply_weather(f, kind, level) for n, f in raw.items()}
    frames[FL], frames[FR] = ctx.apply_sync(raw[FL], raw[FR], kind, level)
    return frames


def process_condition(ctx, raw, kind, level, t, gseed, full, st=None, truth_xy=None):
    """One condition of one frame.  full=False: weather + features only (sensor stream).  full=True: live pipeline (stereo depth, YOLO on 4 views, objects_from_detections, corrected_objects, grid)."""
    st = st or ctx.st; A, env = ctx.A, ctx.env; res = {}
    t0 = time.perf_counter()
    with st("weather"): frames = weathered(ctx, raw, kind, level, gseed, t)
    with st("features"): res["f"] = ctx.features(frames[FL])
    res["frames"], res["t_weather_s"] = frames, time.perf_counter() - t0
    if not full: return res
    t0 = time.perf_counter()
    with st("depth"): depth = A.front_stereo_depth(frames[FL], frames[FR], env.asm.matcher)
    with st("yolo"): dets = {n: A.run_yolo(env.yolo, frames[n]) for n in A.YOLO_CAMERAS}
    with st("objects"): raw_objs = env.asm.objects_from_detections(dets, depth)
    res["objs_repr"] = [repr(o) for o in raw_objs]                                # snapshot BEFORE the builder runs (the builder must not be able to alter what is compared)
    with st("corrected"): corr = ctx.builder.corrected_objects(raw_objs)
    with st("grid"): grid = ctx.builder.build(raw_objs, env.md)
    res.update({"depth": depth, "dets": dets, "raw_objs": raw_objs, "corr": corr, "grid": grid, "t_pipeline_s": time.perf_counter() - t0})
    return res


# ----------------------------------------------------------------------------------------------------------------------------------------------- provenance
def versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ["numpy", "opencv-python", "opencv-python-headless", "torch", "ultralytics", "metadrive", "metadrive-simulator", "panda3d", "psutil", "scipy"]:
        try: out[pkg] = metadata.version(pkg)
        except Exception: pass
    return out


def source_hashes():
    missing = [f for f in SOURCE_FILES if not Path(f).is_file()]
    if missing: raise SystemExit(f"REQUIRED PROVENANCE FILES MISSING (explicit error, nothing started): {missing}")
    return {f: sha256(f) for f in SOURCE_FILES}


def provenance_doc(ctx, a, hashes):
    import psutil
    from manual_drive_stereo_yolo_weather import CAM_W, CAM_H, CAM_FOV, CAMERA_RIGS, FOCAL_LENGTH_PX, STEREO_BASELINE_M
    m = ctx.env.asm.matcher; sg = {}
    for k in ["BlockSize", "NumDisparities", "MinDisparity", "P1", "P2", "Disp12MaxDiff", "PreFilterCap", "UniquenessRatio", "SpeckleWindowSize", "SpeckleRange", "Mode"]:
        try: sg[k] = getattr(m, "get" + k)()
        except Exception as e: sg[k] = f"unavailable: {e}"
    return {"schema": SCHEMA, "created": time.strftime("%F %T"), "purpose": "timing_check" if a.diagnostics else "full_pass", "sha256": hashes,
            "versions": versions(), "host": {"name": socket.gethostname(), "cpu_logical": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "processor": platform.processor()},
            "camera": {"width": CAM_W, "height": CAM_H, "fov_deg": CAM_FOV, "focal_px": FOCAL_LENGTH_PX, "baseline_m": STEREO_BASELINE_M, "rigs": {k: [list(v[0]), list(v[1])] for k, v in CAMERA_RIGS.items()}},
            "stereo_sgbm": sg, "yolo": {"conf": 0.4, "weights": "weights/yolov8n.pt", "cameras": list(ctx.A.YOLO_CAMERAS)},
            "weather": {"conditions": COND_NAMES, "probes": sorted(cname(*c) for c in PROBES), "noise_seed_formula": f"{NOISE_BASE} + global_seed*{NOISE_MULT} + t", "t0_skipped": True, "smoothing_start_t": 1},
            "teacher": {"name": TEACHER, "density": DENSITY, "horizon": HORIZON, "action_noise": ACTION_NOISE},
            "sampling": {"frame_stride": FRAME_STRIDE, "max_retention_frames": MAX_RETENTION_FRAMES, "max_vehicle_free_frames": MAX_VEHICLE_FREE_FRAMES, "window": [X_LO, X_HI, Y_LO, Y_HI],
                         "match_radius_m": 3.0, "stream_scope": "S-B-stride2", "weather_stream_stride": WEATHER_STREAM_STRIDE},
            "partitions": {"train": [9000, 9029], "reserve": [9030, 9039], "heldout": [9040, 9049]}, "group_size": GROUP_SIZE,
            "bootstrap": {"resamples": 2000, "seed": 20261115, "min_valid_resamples": 1900}}


def ensure_provenance(out, ctx, a):
    """Written ONCE (atomic hard-link create); every worker re-validates every source hash against it and refuses to run on any mismatch."""
    hashes, p = source_hashes(), out / "PROVENANCE.json"
    if not p.exists():
        tmp = out / f"PROVENANCE.tmp{os.getpid()}"; tmp.write_text(json.dumps(provenance_doc(ctx, a, hashes), indent=1))
        try: os.link(tmp, p)
        except FileExistsError: pass
        finally: tmp.unlink(missing_ok=True)
    old = json.loads(p.read_text()); bad = [f for f, h in old["sha256"].items() if hashes.get(f) != h]
    if bad: raise SystemExit(f"source files changed since PROVENANCE.json was written: {bad}. Refusing to run.")
    return old


# ------------------------------------------------------------------------------------------------------------------------------------------- claims / failures
def read_claim(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def make_claim(path, worker):
    path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: return False
    with os.fdopen(fd, "w") as f: json.dump({"pid": os.getpid(), "worker": worker, "host": socket.gethostname(), "started": time.strftime("%F %T"), "status": "running"}, f)
    return True


def set_claim_status(path, status, **extra):
    c = read_claim(path) or {}; c.update(status=status, **extra); write_atomic(path, c)


def n_attempts(out, seed):
    return len(list((out / "failures").glob(f"s{seed}.attempt*.json")))


def write_failure(out, seed, ctx, attempt, exc):
    (out / "failures").mkdir(exist_ok=True)
    write_atomic(out / "failures" / f"s{seed}.attempt{attempt}.json",
                 {"seed": seed, "attempt": attempt, "t": ctx.t, "condition": ctx.cond, "stage": ctx.st.fail or ctx.st.cur, "exception": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "time": time.strftime("%F %T"), "pid": os.getpid()})


# ----------------------------------------------------------------------------------------------------------------------------------------------------- episode
def run_episode(ctx, seed, part, a):
    from bev_rl.adaptive.camera_dagger import teacher_driver
    env, st, mem = ctx.env, ctx.st, ctx.mem
    st.reset(); mem.restart(); ctx.equiv, ctx.bench = [], []; ctx.t, ctx.cond = 0, None
    t_ep = time.perf_counter()
    env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); env.drng = np.random.default_rng(seed * 7 + 1)
    act_fn = teacher_driver(TEACHER, env, 0.0)                                   # c = 0.0 (clear), no action noise anywhere in this loop
    gseed = int(env.md.engine.global_seed)
    held, wx = part == "heldout", [c for c in CONDITIONS if c != ("clear", 0.0)]
    K = len(CONDITIONS) if held else 1
    s_t, s_f, s_u = [], [], []; retention, vfree, ret_feats, grids = [], [], [], {}; t_cond = defaultdict(lambda: {"weather_s": 0.0, "pipeline_s": 0.0, "n_calls": 0}); t_frame = []
    t = 0
    while True:
        if t >= 1:
            ctx.t = t
            sample = part in ("train", "reserve") and t % FRAME_STRIDE == 0 and t >= FRAME_STRIDE
            diag = a.diagnostics and t in DIAG_STEPS
            bench = a.diagnostics and not held and t <= a.stream_benchmark_steps
            wstream = held and t % WEATHER_STREAM_STRIDE == 0                      # held-out weather stream step (even t)
            all5 = wstream or sample or diag or bench
            with preserve_rng():
                with st("capture"): raw = ctx.capture(ctx.CAMERA_RIGS.keys() if all5 else [FL])
                row = np.full((K, len(FEATURE_NAMES)), np.nan)
                with st("features"): row[0] = ctx.features(raw[FL])               # clear = the raw front-left frame itself, as in the live clear path
                unsamp = np.zeros(K, bool)
                if held and not wstream: unsamp[1:] = True                          # odd step: weather cells are unsampled BY DESIGN (row stays NaN)
                if wstream:
                    for k, (kind, level) in enumerate(wx, 1):                      # even steps, 10 weather conditions: full live weather block, features of the final front-left image
                        ctx.cond = cname(kind, level)
                        with st("stream_weather"): frames = weathered(ctx, raw, kind, level, gseed, t)
                        with st("stream_features"): row[k] = ctx.features(frames[FL])
                        del frames
                s_t.append(t); s_f.append(row); s_u.append(unsamp)
                if bench:                                                          # diagnostic timing of the held-out style stream on a training episode; NEVER saved as data
                    tb = {}; t0 = time.perf_counter()
                    for kind, level in wx:
                        ctx.cond = cname(kind, level); t1 = time.perf_counter()
                        with st("bench_weather"): fr_ = weathered(ctx, raw, kind, level, gseed, t)
                        with st("bench_features"): ctx.features(fr_[FL])
                        tb[cname(kind, level)] = round(time.perf_counter() - t1, 4); del fr_
                    ctx.bench.append({"t": t, "total_s": round(time.perf_counter() - t0, 4), "by_condition": tb})
                if sample:
                    T = [(x.x_right_m, x.y_forward_m) for x in env.truth_objects()]
                    mode = "retention" if (T and len(retention) < MAX_RETENTION_FRAMES) else "vehicle_free" if (not T and len(vfree) < MAX_VEHICLE_FREE_FRAMES) else None
                    if mode:
                        t_fr = time.perf_counter()
                        with st("hash"): h0 = {n: hash_arr(v) for n, v in raw.items()}
                        rec = {"t": t, "n_truth": len(T), "truth_xy": [[round(x, 2), round(y, 2)] for x, y in T], "raw_sha256": h0, "cond": {}}; fr = {}
                        for kind, level in CONDITIONS:
                            name = cname(kind, level); ctx.cond = name
                            r = process_condition(ctx, raw, kind, level, t, gseed, True)
                            fr[name] = [float(v) for v in r["f"]]
                            with st("match"):
                                O = [(q.x_right_m, q.y_forward_m) for q in r["corr"] if X_LO <= q.x_right_m <= X_HI and Y_LO <= q.y_forward_m <= Y_HI]
                                pairs, n_dup, n_fp = ctx.match(T, O); n_det = sum(len(v) for v in r["dets"].values())
                            cell = {"matched": len(pairs), "matched_idx": sorted(p[0] for p in pairs), "dup": n_dup, "fp": n_fp, "n_det": n_det, "n_raw": len(r["raw_objs"]), "n_corr": len(r["corr"]),
                                    "n_dropped_invalid_depth": n_det - len(r["raw_objs"]), "grid_occupied_cells": int(np.sum(np.asarray(r["grid"]) == 1.0))}
                            if mode == "vehicle_free":
                                confs = [float(d.get("confidence", 0.0)) for v in r["dets"].values() for d in v]
                                cell.update({"valid_depth_share": round(float((np.asarray(r["depth"]) > 0).mean()), 5), "conf_mean": round(float(np.mean(confs)), 4) if confs else None,
                                             "conf_max": round(max(confs), 4) if confs else None})
                            rec["cond"][name] = cell; grids[f"t{t}|{name}"] = np.asarray(r["grid"], np.float32)
                            tc = t_cond[name]; tc["weather_s"] += r["t_weather_s"]; tc["pipeline_s"] += r["t_pipeline_s"]; tc["n_calls"] += 1
                            del r
                        with st("hash"): h1 = {n: hash_arr(v) for n, v in raw.items()}
                        if h1 != h0: raise ImmutabilityError(f"raw capture at t={t} changed while the 11 conditions were processed")
                        (retention if mode == "retention" else vfree).append(rec); ret_feats.append({"t": t, "kind": mode, "f": fr}); t_frame.append(round(time.perf_counter() - t_fr, 2))
                        ctx.mem.sample()
                if diag:
                    from bev_rl.adaptive.verify_stream_equivalence import check_step
                    with st("verify"): rep = check_step(ctx, t, raw, gseed)
                    ctx.equiv.append(rep)
                    if not rep["ok"]: raise EquivalenceMismatch(f"live-path equivalence FAILED at t={t}: {rep['mismatches'][:5]}")
            ctx.cond = None
        with st("sim"):
            act = np.clip(np.asarray(act_fn(env), np.float32), -1, 1); _, _, te, tr, info = env.step(act)
        t += 1
        if te or tr: break
    steps = t
    # ----------------------------------------------------------------------- stream + grids + episode record (the episode json is written LAST = completion marker)
    st_arr = np.asarray(s_f, np.float64).reshape(len(s_t), K, len(FEATURE_NAMES)); valid = np.isfinite(st_arr).all(-1); unsampled = np.asarray(s_u, bool).reshape(len(s_t), K)
    expected = list(range(1, steps)); w_exp = [x for x in expected if (not held) or x % WEATHER_STREAM_STRIDE == 0]
    bad_cells = ~(valid | unsampled)                                             # a cell is acceptable only if finite or unsampled by design
    missing = sorted(set(expected) - set(s_t)) + [int(x) for x in np.asarray(s_t)[bad_cells.any(-1)]]
    return {"schema": SCHEMA, "seed": seed, "partition": part, "group": group_of(seed), "status": "complete", "global_seed": gseed, "outcome": str(info.get("outcome")), "steps": steps,            "wall_s": round(time.perf_counter() - t_ep, 1),
            "config": {"teacher": TEACHER, "density": DENSITY, "horizon": HORIZON, "action_noise": ACTION_NOISE, "frame_stride": FRAME_STRIDE, "stream_scope": "S-B-stride2", "weather_stream_stride": WEATHER_STREAM_STRIDE, "t0_skipped": True, "smoothing_start_t": 1,
                       "max_retention_frames": MAX_RETENTION_FRAMES, "max_vehicle_free_frames": MAX_VEHICLE_FREE_FRAMES, "conditions": COND_NAMES},
            "time_s": {k: round(v, 2) for k, v in st.t.items()}, "stage_calls": dict(st.n),
            "time_by_condition": {k: {"weather_s": round(v["weather_s"], 2), "pipeline_s": round(v["pipeline_s"], 2), "n_calls": v["n_calls"]} for k, v in t_cond.items()},
            "retention_frame_wall_s": t_frame, "memory": mem.summary(),
            "frames": {"retention": [r["t"] for r in retention], "vehicle_free": [r["t"] for r in vfree], "skipped_by_design": [0], "failed": []},
            "retention": retention, "vehicle_free": vfree, "retention_features": ret_feats,
            "stream": {"file": f"streams/s{seed}_stream.npz", "scope": "all_conditions" if held else "clear_only", "n_expected": len(expected), "n_valid": int((valid | unsampled).all(-1).sum()), "missing": missing,
                       "weather_stride": WEATHER_STREAM_STRIDE if held else None, "n_weather_expected": len(w_exp) if held else 0,
                       "n_weather_valid": int(valid[:, 1:].all(-1).sum()) if held else 0, "n_weather_unsampled_by_design": int(unsampled.any(-1).sum()), "skipped_by_design": [0], "conditions": COND_NAMES if held else [COND_NAMES[0]]}}, st_arr, np.asarray(s_t, np.int32), valid, unsampled, grids


def save_episode(out, seed, ep, st_arr, s_t, valid, unsampled, grids, attempt):
    for sub in ("episodes", "streams", "grids"): (out / sub).mkdir(exist_ok=True)
    held = ep["partition"] == "heldout"; sp, gp = out / "streams" / f"s{seed}_stream.npz", out / "grids" / f"s{seed}.npz"
    for p in (sp, gp):                                                  # never overwrite: an orphan from a crashed attempt is moved aside, not deleted
        if p.exists(): os.replace(p, p.with_name(p.name + f".orphan_attempt{attempt}"))
    write_npz_atomic(sp, t=s_t, cond_names=np.array(ep["stream"]["conditions"]), feature_names=np.array(FEATURE_NAMES), feat=st_arr, valid=valid, by_design_unsampled=unsampled,
                    weather_stride=np.int32(WEATHER_STREAM_STRIDE if held else 1))
    ep["stream"]["sha256"] = sha256(sp)
    if not held:
        write_npz_atomic(gp, **grids); ep["grids"] = {"file": f"grids/s{seed}.npz", "n": len(grids), "sha256": sha256(gp)}
    ep["attempt"] = attempt
    write_atomic(out / "episodes" / f"s{seed}.json", ep)


def diag_outputs(out, ctx, ep, seeds_done):
    d = out / "diagnostics"; d.mkdir(exist_ok=True)
    sched = list(DIAG_STEPS); done = [r["t"] for r in ctx.equiv]; unreached = [s for s in sched if s not in done]
    n_mis = sum(len(r["mismatches"]) for r in ctx.equiv); eq_ok = len(done) >= 2 and n_mis == 0 and all(r["ok"] for r in ctx.equiv)
    write_atomic(d / "equivalence_report.json", {"scheduled": sched, "completed": done, "unreached": unreached, "total_mismatches": n_mis, "pass_rule": ">=2 completed steps, zero mismatches at every completed step",
                                                   "pass": eq_ok, "steps": ctx.equiv})
    if ctx.bench:
        tot = [b["total_s"] for b in ctx.bench]; per_cond = defaultdict(list)
        for b in ctx.bench:
            for k, v in b["by_condition"].items(): per_cond[k].append(v)
        n_steps = (ep or {}).get("steps"); mean = float(np.mean(tot))
        write_atomic(d / "stream_benchmark.json", {"DIAGNOSTIC_ONLY": "timing of the held-out-style all-condition stream on a TRAINING episode; no features saved; never a data product", "n_steps_timed": len(tot), "mean_step_s": mean,
                                                    "median_step_s": float(np.median(tot)), "p90_step_s": float(np.percentile(tot, 90)), "mean_by_condition_s": {k: float(np.mean(v)) for k, v in per_cond.items()},
                                                    "episode_steps": n_steps, "extrapolated_stream_s_per_heldout_episode": None if n_steps is None else mean * (n_steps - 1), "per_step": ctx.bench})
    ctx.mem.write_csv(d / "memory_samples.csv")
    if ep:
        rf = ep["retention_frame_wall_s"]
        write_atomic(d / "timing_report.json", {"seed": ep["seed"], "steps": ep["steps"], "wall_s": ep["wall_s"], "stage_total_s": ep["time_s"], "stage_calls": ep["stage_calls"], "time_by_condition": ep["time_by_condition"],
                                                 "retention_frame_wall_s": {"n": len(rf), "mean": float(np.mean(rf)) if rf else None, "max": max(rf) if rf else None}, "memory": ep["memory"],
                                                 "verdict": {"episode_complete": ep["status"] == "complete", "equivalence_pass": eq_ok, "equivalence_completed_steps": done, "equivalence_unreached_steps": unreached,
                                                             "failure_records": len(list((out / "failures").glob("*.json"))) if (out / "failures").exists() else 0}})


# ------------------------------------------------------------------------------------------------------------------------------------------------------ collect
def collect(a):
    if a.worker not in range(1, MAX_WORKERS + 1): raise SystemExit(f"--worker must be 1..{MAX_WORKERS}")
    if a.worker > 1 and not MULTIWORKER_FLAG.exists(): raise SystemExit(f"worker {a.worker}: more than one worker needs the user-created flag file {MULTIWORKER_FLAG} (not present). Refused.")
    seeds = parse_seeds(a.seeds)
    for s in seeds:
        p = partition(s)
        if p is None: raise SystemExit(f"seed {s} is outside the approved ranges 9000-9029 / 9030-9039 (reserve) / 9040-9049")
        if p == "reserve" and not a.allow_reserve: raise SystemExit(f"seed {s} is a RESERVE seed: refused without --allow-reserve (only under the approved stop rule)")
    out = a.out
    if a.diagnostics:
        if out.name != TIMING_DIR_NAME: raise SystemExit(f"--diagnostics is only allowed with --out .../{TIMING_DIR_NAME}")
        if a.worker != 1 or any(partition(s) != "train" for s in seeds) or len(seeds) != 1: raise SystemExit("--diagnostics: exactly one TRAINING seed and worker 1")
    elif out.name == TIMING_DIR_NAME: raise SystemExit("the timing-check folder only accepts --diagnostics runs")
    if not Path("bev_state").is_dir(): raise SystemExit("run from the repository root (D:/bev_Fresh/bevcapstone)")
    from bev_rl.adaptive.camera_dagger import verify_frozen
    verify_frozen()                                                                 # builder / assembler / YOLO weights / OV checkpoints / old sensor files unchanged (read-only check)
    import torch
    torch.set_num_threads(2)
    for sub in ("episodes", "claims", "failures", "streams", "grids", "summary"): (out / sub).mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a.mem_interval); ensure_provenance(out, ctx, a); ctx.mem.start()
    exit_code = 0
    try:
        for seed in seeds:
            res, cl = out / "episodes" / f"s{seed}.json", out / "claims" / f"s{seed}.json"
            if res.exists(): continue                                                # completed data is never overwritten
            if cl.exists():
                c = read_claim(cl) or {}
                if a.retry_failed and c.get("status") == "failed":                  # only a claim its own worker marked failed may be reclaimed; running / orphan claims are NEVER released
                    os.replace(cl, out / "claims" / f"s{seed}.failed_claim_attempt{n_attempts(out, seed)}.json")
                else: continue
            if not make_claim(cl, a.worker): continue
            attempt = n_attempts(out, seed) + 1
            try:
                ep, st_arr, s_t, valid, unsampled, grids = run_episode(ctx, seed, partition(seed), a)
                save_episode(out, seed, ep, st_arr, s_t, valid, unsampled, grids, attempt); set_claim_status(cl, "complete")
                print(json.dumps({k: ep[k] for k in ("seed", "partition", "group", "steps", "outcome", "wall_s", "time_s")} | {"memory": {k: ep["memory"][k] for k in ("peak_working_set_mb", "min_system_available_mb")},
                                                                                                                            "retention_frames": len(ep["retention"]), "vehicle_free_frames": len(ep["vehicle_free"])}), flush=True)
                if a.diagnostics: diag_outputs(out, ctx, ep, [seed])
            except BaseException as e:                                               # noqa: BLE001 - every failure is recorded with its exact reason, never skipped silently
                write_failure(out, seed, ctx, attempt, e); set_claim_status(cl, "failed", attempt=attempt)
                print(f"FAILED seed {seed} attempt {attempt}: {type(e).__name__}: {e}", flush=True)
                if a.diagnostics: diag_outputs(out, ctx, None, [])
                if isinstance(e, (EquivalenceMismatch, KeyboardInterrupt, SystemExit)): exit_code = 3; break
                ctx.new_env()
    finally:
        ctx.mem.stop_evt.set()
        try: ctx.env.md.close()
        except Exception: pass
    sys.exit(exit_code)


# ------------------------------------------------------------------------------------------------------------------------------------------ status / validate
def status(a):
    import psutil
    out, seeds = a.out, parse_seeds(a.seeds); rows = []
    for s in seeds:
        ep, cl = out / "episodes" / f"s{s}.json", out / "claims" / f"s{s}.json"; c = read_claim(cl) if cl.exists() else None; nf = n_attempts(out, s)
        if ep.exists(): st = "complete"
        elif c and c.get("status") == "failed": st = f"FAILED (attempts {nf})"
        elif c: st = "running" if psutil.pid_exists(int(c.get("pid", -1))) else "ORPHAN CLAIM (pid not alive: needs the user's decision, never auto-released)"
        else: st = "pending"
        rows.append((s, partition(s), st, c and c.get("worker")))
    for r in rows: print("seed %d  %-8s %-70s worker %s" % r)
    print("complete %d / %d" % (sum(r[2] == "complete" for r in rows), len(rows)))


def validate_out(out, expect):
    checks = []

    def chk(name, ok, detail=""): checks.append({"check": name, "pass": bool(ok), "detail": detail})
    out = Path(out); prov = out / "PROVENANCE.json"
    chk("provenance exists", prov.exists())
    if prov.exists():
        try: old = json.loads(prov.read_text()); bad = [f for f, h in old["sha256"].items() if not Path(f).is_file() or sha256(f) != h]; chk("source hashes still match PROVENANCE.json", not bad, str(bad))
        except Exception as e: chk("provenance readable", False, str(e))
    n_ok = 0
    for s in expect:
        p = partition(s); ep = out / "episodes" / f"s{s}.json"
        if not ep.exists(): chk(f"s{s} episode present", False); continue
        e = json.loads(ep.read_text()); bad = []
        if e.get("schema") != SCHEMA or e.get("status") != "complete": bad.append("schema/status")
        if e["partition"] != p or e["config"]["action_noise"] != 0.0 or e["config"]["teacher"] != TEACHER or e["config"]["density"] != DENSITY or e["config"]["horizon"] != HORIZON: bad.append("partition/config")
        sp = out / "streams" / f"s{s}_stream.npz"
        if not sp.exists() or sha256(sp) != e["stream"].get("sha256"): bad.append("stream file/hash")
        else:
            z = np.load(sp, allow_pickle=False); K = 11 if p == "heldout" else 1; N = e["steps"] - 1
            uns = z["by_design_unsampled"]; tt = z["t"]; want_uns = np.zeros((N, K), bool)
            if p == "heldout": want_uns[(tt % WEATHER_STREAM_STRIDE) != 0, 1:] = True            # weather cells exist exactly at even t; clear at every t
            if z["feat"].shape != (N, K, 7) or not np.array_equal(tt, np.arange(1, e["steps"])) or e["stream"]["missing"]: bad.append("stream shape/t/missing")
            elif not np.array_equal(uns, want_uns) or not bool(np.array_equal(z["valid"], ~want_uns)): bad.append("stream sampling mask (valid must equal ~unsampled; unsampled = odd-t weather cells)")
            elif p == "heldout" and int(z["weather_stride"]) != WEATHER_STREAM_STRIDE: bad.append("weather_stride")
            if list(z["cond_names"]) != e["stream"]["conditions"]: bad.append("stream conditions")
        if p == "heldout" and (e["retention"] or e["vehicle_free"]): bad.append("held-out episode carries retention records")
        if p != "heldout":
            ts = [r["t"] for r in e["retention"]] + [r["t"] for r in e["vehicle_free"]]
            if len(e["retention"]) > MAX_RETENTION_FRAMES or len(e["vehicle_free"]) > MAX_VEHICLE_FREE_FRAMES or any(t % 5 or t < 5 for t in ts): bad.append("retention frame caps / stride / t>=5")
            if any(r["n_truth"] < 1 or set(r["cond"]) != set(COND_NAMES) for r in e["retention"]) or any(r["n_truth"] != 0 or set(r["cond"]) != set(COND_NAMES) for r in e["vehicle_free"]): bad.append("frame class / 11 conditions")
            gp = out / "grids" / f"s{s}.npz"
            if not gp.exists() or sha256(gp) != e.get("grids", {}).get("sha256") or e["grids"]["n"] != 11 * len(ts): bad.append("grids")
        chk(f"s{s} ({p}) episode valid", not bad, ", ".join(bad)); n_ok += not bad
        cl = read_claim(out / "claims" / f"s{s}.json")
        chk(f"s{s} claim not left running", cl is None or cl.get("status") == "complete", str(cl))
    chk("no seed outside the expected set has an episode", set(int(f.stem[1:]) for f in (out / "episodes").glob("s*.json")) <= set(expect) | set(RESERVE_SEEDS), "")
    chk("all expected episodes valid", n_ok == len(expect), f"{n_ok}/{len(expect)}")
    ok = all(c["pass"] for c in checks); rep = {"time": time.strftime("%F %T"), "expected_seeds": expect, "all_pass": ok, "checks": checks}
    (out / "summary").mkdir(exist_ok=True); write_atomic(out / "summary" / "validation_report.json", rep)
    return rep


def validate(a):
    rep = validate_out(a.out, parse_seeds(a.expect))
    for c in rep["checks"]:
        if not c["pass"]: print("FAIL", c["check"], c["detail"])
    print("VALID" if rep["all_pass"] else "INVALID", f"({sum(c['pass'] for c in rep['checks'])}/{len(rep['checks'])} checks)")
    sys.exit(0 if rep["all_pass"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "status", "validate"]); ap.add_argument("--out", type=Path, default=ROOT); ap.add_argument("--worker", type=int, default=1)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS); ap.add_argument("--expect", default=DEFAULT_SEEDS); ap.add_argument("--allow-reserve", action="store_true"); ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--diagnostics", action="store_true"); ap.add_argument("--stream-benchmark-steps", type=int, default=0); ap.add_argument("--mem-interval", type=float, default=2.0)
    a = ap.parse_args(); {"collect": collect, "status": status, "validate": validate}[a.cmd](a)
