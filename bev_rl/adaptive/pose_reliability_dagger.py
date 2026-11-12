"""DAgger under random ego-pose (HD-map localisation) error, with optional RELIABILITY CONDITIONING.

Student = overlay-BEV + 3 scalars (speed, progress, goal), same as ov_split, with one extra input c = pose-error level in [0,1] (BEVScalarPolicy weather branch).
Per training episode: level L ~ 30% exactly 0 / 70% U(0,1); persistent pose error lateral = +-L*1.5 m, heading = +-L*0.15 rad is applied to the MAP OVERLAY
(spec_bev.POSE_ERROR). The TEACHER (stock IDM, or SafeTeacher with --safe-teacher) always acts on the TRUE state, so it labels what the car should do.
  --condition      : c = L is fed to the student (it is told how unreliable its map alignment is)  [proposed method]
  (default)        : c = 0 always (pure noise augmentation)                                       [control]
No detector dropout, no caution-teacher, clear weather: the only degradation is pose error.
TWO-CUE VARIANT (--cam-cue): with one cue (the map) a shifted map is unrecoverable, so conditioning cannot help. --cam-cue also gives the student NOISY lateral-offset and
heading-error scalars (an on-board lane-detector stand-in, independent of the map error): per step Gaussian noise sd = Lcam * (0.6 m, 0.06 rad), Lcam ~ 30% 0 / 70% U(0,1)
per episode, drawn independently of the map level L. With --condition the student is told BOTH levels c = [L_map, L_cam] (weather_dim 2). Without it c = [0, 0].
Selection: validation seeds 3100-3109 at density 0.05, mean success over pose-error levels {0, 0.33, 0.67} (with c = true level if --condition).
    python -m bev_rl.adaptive.pose_reliability_dagger --name pose_aug_conditioned --condition
"""
import argparse, json, time
from pathlib import Path
import numpy as np

LAT_MAX, HEAD_MAX = 1.5, 0.15
CAM_LAT, CAM_HEAD = 0.6, 0.06           # lane-detector noise sd at Lcam = 1
KEEP_MAP_ONLY = np.array([1, 1, 0, 0, 0, 1], np.float32)
KEEP_TWO_CUE = np.array([1, 1, 1, 1, 0, 1], np.float32)     # + lateral offset (2) and heading error (3) from the camera stand-in
VAL_SEEDS = list(range(3100, 3110)); VAL_LEVELS = [0.0, 1 / 3, 2 / 3]; DENSITIES = [0.0, 0.05, 0.1]


def prep2(torch, bev, raw, cvec, mu, sd, keep):
    b = torch.from_numpy(bev[None, None].astype(np.float32)); sc = torch.from_numpy((((raw - mu) / sd) * keep)[None].astype(np.float32))
    return b, sc, torch.tensor([cvec], dtype=torch.float32)


def episode(env, net, mu, sd, torch, seed, level, cond, beta, noise, rng, collect, safe, lane_change, cam_cue=False, lcam=0.0):
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive import spec_bev
    from bev_rl.adaptive.env import SCALE
    sx, st = rng.choice([-1.0, 1.0], 2)
    o, _ = env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)})
    spec_bev.POSE_ERROR = (float(sx * level * LAT_MAX), 0.0, float(st * level * HEAD_MAX))
    keep = KEEP_TWO_CUE if cam_cue else KEEP_MAP_ONLY
    cvec = ([float(level), float(lcam)] if cam_cue else [float(level)]) if cond else ([0.0, 0.0] if cam_cue else [0.0])
    pol = IDMPolicy(env.md.agent, env.md.engine.global_seed); pol.enable_lane_change = lane_change
    teach = pol
    if safe:
        from bev_rl.adaptive.safe_teacher import SafeTeacher
        pol.enable_lane_change = False; teach = SafeTeacher(pol)
    B, S, C, Y = [], [], [], []
    try:
        while True:
            bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)
            if cam_cue: raw[2] += rng.normal(0, lcam * CAM_LAT); raw[3] += rng.normal(0, lcam * CAM_HEAD)       # lane-detector stand-in (noisy lateral/heading)
            ea = np.clip(np.asarray(teach.act(env) if safe else pol.act(), np.float32)[:2], -1, 1)
            if collect: B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); C.append(cvec); Y.append(ea)
            if net is not None and not (collect and rng.random() < beta):
                with torch.no_grad(): a = net(*prep2(torch, bev, raw, cvec, mu, sd, keep)).squeeze(0).numpy()
            else: a = ea
            if noise > 0: a = np.clip(a + rng.normal(0, noise, 2), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(a)
            if te or tr: return B, S, C, Y, info
    finally:
        spec_bev.POSE_ERROR = None


def train2(net, B, S, C, Y, mu, sd, torch, epochs, lr, rng, keep):
    opt = torch.optim.Adam(net.parameters(), lr)
    Bt = torch.from_numpy(np.stack(B)).float().div(10).unsqueeze(1); St = torch.from_numpy((((np.stack(S) - mu) / sd) * keep).astype(np.float32))
    Ct = torch.tensor(np.array(C, np.float32)); Yt = torch.from_numpy(np.stack(Y)); idx = np.arange(len(Bt)); net.train()
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; loss = ((net(Bt[j], St[j], Ct[j]) - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return float(loss)


def validate(env, net, mu, sd, torch, cond, rng, n, cam_cue=False):
    res = []
    for L in VAL_LEVELS:
        for s in VAL_SEEDS[:n]:
            *_, i = episode(env, net, mu, sd, torch, s, L, cond, 0.0, 0.0, rng, False, False, True, cam_cue, 0.33 if cam_cue else 0.0)
            res.append((i["outcome"] == "success", i["outcome"] in ("crash", "off_road")))
    a = np.array(res, float); return float(a[:, 0].mean()), float(a[:, 1].mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--eps-per-iter", type=int, default=45); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val-n", type=int, default=10)
    ap.add_argument("--condition", action="store_true"); ap.add_argument("--safe-teacher", action="store_true"); ap.add_argument("--cam-cue", action="store_true", help="two-cue variant: add noisy lateral/heading scalars (lane-detector stand-in)")
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.caution import CautionEnv
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True); (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    wd = 2 if a.cam_cue else 1; net = BEVScalarPolicy(1, 6, wd); B, S, C, Y = [], [], [], []; mu = sd = None; best, log, t0 = (-1.0, 1.0), [], time.time()
    for it in range(a.iters + 1):
        beta = 1.0 if it == 0 else 0.5 ** it; stats = []
        for d in DENSITIES:
            env = CautionEnv(range(2000, 3000), [("clear", 0.0)], dropout=False, traffic_density=d, horizon=600, sample_seed=a.seed * 10 + it)
            for k in range(max(1, a.eps_per_iter // len(DENSITIES))):
                L = 0.0 if rng.random() < 0.3 else float(rng.random())
                Lc = (0.0 if rng.random() < 0.3 else float(rng.random())) if a.cam_cue else 0.0
                b, s, cc, y, info = episode(env, None if it == 0 else net, mu, sd, torch, int(rng.choice(range(2000, 3000))), L, a.condition, beta, a.noise, rng, True, a.safe_teacher, True, a.cam_cue, Lc)
                B += b; S += s; C += cc; Y += y; stats.append(info["outcome"])
            env.close()
        if mu is None:
            Sa = np.stack(S); mu, sd = Sa.mean(0).astype(np.float32), np.maximum(Sa.std(0), 1e-6).astype(np.float32)
        loss = train2(net, B, S, C, Y, mu, sd, torch, a.epochs if it else 3 * a.epochs, 1e-3, rng, KEEP_TWO_CUE if a.cam_cue else KEEP_MAP_ONLY)
        venv = CautionEnv(VAL_SEEDS, [("clear", 0.0)], dropout=False, traffic_density=0.05, horizon=600); v = validate(venv, net, mu, sd, torch, a.condition, np.random.default_rng(99), a.val_n, a.cam_cue); venv.close()
        rec = {"iter": it, "beta": beta, "samples": len(B), "train_loss": loss, "rollout_outcomes": {o: stats.count(o) for o in set(stats)}, "val_success": v[0], "val_fail": v[1], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save({"model_state": net.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist(), "iter": it, "val": v, "condition": a.condition, "cam_cue": a.cam_cue}, out / f"iter_{it}.pt")
        if (v[0], -v[1]) > (best[0], -best[1]):
            best = v; torch.save(torch.load(out / f"iter_{it}.pt", weights_only=True), out / "best.pt"); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))

if __name__ == "__main__":
    main()
