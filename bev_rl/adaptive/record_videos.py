"""Record a video of every model driving the SAME scenario (paired seeds). Policy-side tool; read-only on checkpoints.

Each frame (860x560): LEFT = MetaDrive top-down view (ego heading up, other vehicles as boxes); RIGHT = the 64x64 BEV the policy actually receives, colour-coded
(unknown grey, free black, route green, road boundary yellow, ego blue, vehicle red; vehicles removed by the weather dropout are missing there), + model / seed / weather / step / speed / min gap / action bars,
and a final outcome banner. 10 Hz simulation written at 10 fps (real time). File: <out>/<model>__seed<seed>__<weather>.mp4 ; index: <out>/INDEX.md

    python -m bev_rl.adaptive.record_videos --out paper/videos --seeds 1012 1020 1039 1040 --weather clear:0.0
Environment = the evaluation environment (CautionEnv, density 0.05, horizon 800, detector dropout from the measured retention table for fog/rain; NO weather physics).
Rule-based references see ground truth (their BEV panel is shown only for comparison).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2
import numpy as np

LEARNED = ["repo_bc_sparse", "repo_bc_clear", "base_a", "ov_split", "ctrl_nocaut", "caut_c0", "caut_auto", "safe_student"]
RULE = ["stock_idm", "safe_teacher"]
ABL = ["arch_bev_only", "arch_scalar_only", "ov_both", "ov_bev_only"]       # earlier DAgger ablations (input configuration inferred from the model catalogue)
ABL_CFG = {"arch_bev_only": (False, [0, 0, 0, 0, 0, 0], False), "arch_scalar_only": (False, [1, 1, 1, 1, 1, 1], True),
           "ov_both": (True, [1, 1, 1, 1, 1, 1], False), "ov_bev_only": (True, [0, 0, 0, 0, 0, 0], False)}            # (overlay, scalar mask, blank BEV)
PAL = {-1.0: (95, 95, 95), 0.0: (22, 22, 22), 0.5: (70, 200, 100), 0.8: (235, 205, 45), 0.9: (70, 160, 255), 1.0: (235, 60, 60)}
OUTCOME_COL = {"success": (60, 170, 80), "crash": (40, 40, 220), "off_road": (30, 140, 240), "timeout": (160, 90, 60)}   # BGR


def bev_panel(bev, size=256):
    keys = np.array(list(PAL)); cols = np.array([PAL[k] for k in PAL], np.uint8)
    idx = np.abs(bev[..., None] - keys[None, None, :]).argmin(-1)
    img = cols[idx][..., ::-1].astype(np.uint8)                       # RGB palette -> BGR for cv2
    ego_rows = np.where(np.abs(bev - 0.9) < 0.05)[0]
    if len(ego_rows) and ego_rows.mean() < bev.shape[0] / 2: img = img[::-1]       # keep forward = up with the ego in the lower half
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def compose(top_rgb, bev, lines, act, W=860, H=560):
    f = np.full((H, W, 3), 245, np.uint8); f[:, :560] = cv2.cvtColor(top_rgb[:H, :560].astype(np.uint8), cv2.COLOR_RGB2BGR)
    f[24:24 + 256, 580:580 + 256] = bev_panel(bev)
    cv2.putText(f, "policy BEV input (forward = up)", (580, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(f, "green route | yellow boundary | blue ego | red vehicle | grey unknown", (580, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 90, 90), 1, cv2.LINE_AA)
    y = 322
    for k, t in enumerate(lines):
        cv2.putText(f, t, (580, y + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if k else 0.6, (20, 20, 20), 1 if k else 2, cv2.LINE_AA)
    for j, (name, v) in enumerate((("steer", act[0]), ("throttle(+)/brake(-)", act[1]))):
        yy = 470 + 38 * j; cv2.putText(f, name, (580, yy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.rectangle(f, (580, yy), (820, yy + 12), (200, 200, 200), 1); mid = 700
        cv2.rectangle(f, (mid, yy + 1), (int(mid + 119 * float(np.clip(v, -1, 1))), yy + 11), (40, 120, 220) if v >= 0 else (60, 60, 200), -1)
    return f


def build(arm, torch, env):
    """-> dict(kind, net, mu, sd, extra)"""
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.degraded_env import make_safe_expert
    from bev_rl.adaptive.evaluate_n200_all_weather import load_model
    if arm in ("ov_split", "safe_student", "ctrl_nocaut", "caut_c0", "caut_auto"):
        net, mu, sd = load_model(arm, torch); return {"kind": "cond" if arm != "ov_split" and arm != "safe_student" else "ov", "net": net, "mu": mu, "sd": sd}
    if arm in ("repo_bc_sparse", "repo_bc_clear"):
        net, mu, sd = load_model(arm, torch); return {"kind": "raw", "net": net, "mu": mu, "sd": sd}
    if arm == "base_a":
        ck = torch.load("bev_rl/adaptive/runs/base_a/best.pt", map_location="cpu", weights_only=True); net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
        return {"kind": "raw", "net": net, "mu": np.asarray(ck["scalar_mean"], np.float32), "sd": np.asarray(ck["scalar_std"], np.float32)}
    if arm in ABL:
        ck = torch.load(f"bev_rl/adaptive/runs/{arm}/best.pt", map_location="cpu", weights_only=True); net = BEVScalarPolicy(int(ck.get("in_channels", 1)), 6, int(ck.get("weather_dim", 0)))
        net.load_state_dict(ck["model_state"]); net.eval(); return {"kind": "abl", "net": net, "mu": np.asarray(ck["scalar_mean"], np.float32), "sd": np.asarray(ck["scalar_std"], np.float32), "cfg": ABL_CFG[arm]}
    return {"kind": arm}                                                # stock_idm / safe_teacher


def record(arm, seed, weather, out, env, torch, max_steps=None):
    from bev_rl.adaptive.caution import c_auto, keep_prob, prep
    from bev_rl.adaptive.degraded_env import make_safe_expert
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.spec_bev import overlay_github_bev
    P = build(arm, torch, env); kind = P["kind"]
    o, _ = env.reset(options={"scenario_seed": seed, "weather": weather}); env.drng = np.random.default_rng(seed * 7 + 1)
    pol = None
    if kind == "stock_idm":
        from metadrive.policy.idm_policy import IDMPolicy
        pol = IDMPolicy(env.md.agent, env.md.engine.global_seed)
    elif kind == "safe_teacher": pol = make_safe_expert(env)
    c = c_auto(*weather, env.table) if arm == "caut_auto" else 0.0; p = keep_prob(*weather, env.table)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (860, 560)); t, info, speeds = 0, None, []
    while True:
        raw = (o[:6] * SCALE).astype(np.float32)
        if kind in ("ov", "cond"):
            bev = env.student_bev(); b, s, cc = prep(torch, bev, raw, c, P["mu"], P["sd"])
            with torch.no_grad(): act = P["net"](b, s, cc).squeeze(0).numpy()
        elif kind in ("raw", "abl"):
            bev = env.asm.build_bev_grid([x for x in env.truth_objects() if env.drng.random() < p]); sc = ((raw - P["mu"]) / P["sd"]).astype(np.float32)
            if kind == "abl":
                ov, mask, blank = P["cfg"]
                if ov: bev = overlay_github_bev(bev, env.md)
                if blank: bev = np.zeros_like(bev)
                sc = sc * np.asarray(mask, np.float32)
            with torch.no_grad(): act = P["net"](torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc[None])).squeeze(0).numpy()
        else:
            act = np.clip(np.asarray(pol.act(env) if kind == "safe_teacher" else pol.act(), np.float32)[:2], -1, 1); bev = env.student_bev()
        o, r, te, tr, info = env.step(act); t += 1; speeds.append(info["mean_speed"])
        top = env.md.render(mode="topdown", window=False, screen_size=(560, 560), film_size=(1200, 1200), scaling=7, target_agent_heading_up=True, num_stack=0)
        spd = float(env.md.agent.speed); gap = env.min_gap_m()
        wtxt = f"seed {seed}   weather {weather[0]} {weather[1]}" + (f"   c={c:.2f}" if arm == "caut_auto" else "")
        lines = [arm, wtxt, f"t = {t / 10:5.1f} s   step {t}", f"speed {spd:4.1f} m/s   gap ahead {gap:5.1f} m",
                 "rule-based: sees ground truth" if kind in ("stock_idm", "safe_teacher") else "learned policy: sees only the BEV at right"]
        vw.write(compose(top, bev, lines, act))
        if te or tr or (max_steps and t >= max_steps): break
    outcome = info["outcome"] or ("max_steps" if max_steps else "?")
    last = compose(top, bev, lines, act)
    for _ in range(18):                                                  # 1.8 s outcome banner
        fr = last.copy(); cv2.rectangle(fr, (0, 230), (560, 330), (255, 255, 255), -1); cv2.rectangle(fr, (0, 230), (560, 330), OUTCOME_COL.get(outcome, (90, 90, 90)), 4)
        cv2.putText(fr, outcome.upper().replace("_", "-"), (40, 298), cv2.FONT_HERSHEY_SIMPLEX, 1.6, OUTCOME_COL.get(outcome, (90, 90, 90)), 4, cv2.LINE_AA); vw.write(fr)
    vw.release()
    return {"arm": arm, "seed": seed, "weather": f"{weather[0]}:{weather[1]}", "outcome": outcome, "steps": t, "mean_speed": float(np.mean(speeds)), "file": out.name}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/videos"); ap.add_argument("--seeds", nargs="+", type=int, default=[1012, 1020, 1039, 1040])
    ap.add_argument("--arms", nargs="+", default=LEARNED + RULE + ABL); ap.add_argument("--weather", nargs="+", default=["clear:0.0"]); ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    from bev_rl.adaptive.degraded_env import DegradedEnv
    ws = [(x.split(":")[0], float(x.split(":")[1])) for x in a.weather]
    env = DegradedEnv(a.seeds, ws, mode="dropout", traffic_density=0.05, horizon=800, sample_seed=7); idx = out / "index.jsonl"
    for w in ws:
        for arm in a.arms:
            for s in a.seeds:
                f = out / f"{arm}__seed{s}__{w[0]}{'' if w[0] == 'clear' else w[1]}.mp4"
                if a.skip_existing and f.exists(): continue
                try:
                    rec = record(arm, s, w, f, env, torch, a.max_steps); print(json.dumps(rec), flush=True)
                    with open(idx, "a") as fh: fh.write(json.dumps(rec) + "\n")
                except Exception as e: print("FAILED", arm, s, w, repr(e), flush=True)
    env.close()


if __name__ == "__main__":
    main()
