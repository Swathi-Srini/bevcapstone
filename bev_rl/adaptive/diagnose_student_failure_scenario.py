"""Why does a learned policy fail a scenario the teacher passes? Runs the student on one seed (density 0.05, clear, no dropout) and, at every step,
also queries the stock IDM on the STUDENT'S state (label only, not executed). Prints outcome, the step where the student's action first deviates from the
teacher's by > thr, and a trace around the failure: ego lane/lateral offset/speed, nearest vehicle, student vs teacher action.
    python -m bev_rl.adaptive.diagnose_student_failure_scenario --arm ov_split --seed 1019
"""
import argparse
import numpy as np

ARMS = {"ov_split": ("bev_rl/adaptive/runs/ov_split/best.pt", True, [0, 1, 5]), "base_a": ("bev_rl/adaptive/runs/base_a/best.pt", False, None),
        "repo_bc_sparse": ("bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", False, None)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="ov_split", choices=ARMS); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--thr", type=float, default=0.3); ap.add_argument("--tail", type=int, default=40)
    a = ap.parse_args()
    import torch
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import spec_bev
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.env import SCALE
    path, overlay, keep = ARMS[a.arm]
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
    mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
    mask = np.ones(6, np.float32) if keep is None else np.array([1.0 if i in keep else 0.0 for i in range(6)], np.float32)
    env = CautionEnv([a.seed], [("clear", 0.0)], dropout=False, traffic_density=a.density, horizon=800, sample_seed=7)
    o, _ = env.reset(options={"scenario_seed": a.seed, "weather": ("clear", 0.0)})
    ag = env.md.agent; idm = IDMPolicy(ag, env.md.engine.global_seed); rows = []; first_dev = None
    while True:
        bev = env.asm.build_bev_grid(env.truth_objects())
        if overlay: bev = spec_bev.overlay_github_bev(bev, env.md)
        raw = (o[:6] * SCALE).astype(np.float32); sc = (((raw - mu) / sd) * mask)[None].astype(np.float32)
        with torch.no_grad():
            act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
        te = np.clip(np.asarray(idm.act(), np.float32)[:2], -1, 1)
        th = float(ag.heading_theta); fw = np.array([np.cos(th), np.sin(th)]); rt = np.array([np.sin(th), -np.cos(th)]); near = None
        for ob in env.md.engine.get_objects(lambda x: hasattr(x, "speed")).values():
            if ob is ag: continue
            d = np.asarray(ob.position, float) - np.asarray(ag.position, float)
            if near is None or np.hypot(*d) < near[0]: near = (round(float(np.hypot(*d)), 1), round(float(d @ rt), 1), round(float(d @ fw), 1), round(float(ob.speed), 1))
        try: lane = tuple(ag.lane.index)[:2]; lat = round(float(ag.lane.local_coordinates(ag.position)[1]), 2)
        except Exception: lane, lat = None, None
        dev = float(np.abs(act - te).max())
        if first_dev is None and dev > a.thr: first_dev = env.t
        rows.append((env.t, round(float(ag.speed), 1), lane, lat, near, [round(float(x), 2) for x in act], [round(float(x), 2) for x in te], round(dev, 2)))
        o, r, tm, tr, info = env.step(act)
        if tm or tr: break
    print(f"{a.arm} seed {a.seed}: {info['outcome']} at step {env.t}, route {info['route_completion']:.2f}; first |student-teacher| > {a.thr} at step {first_dev}")
    print("  t | speed | lane | lat | nearest(dist,x_right,y_fwd,speed) | student[steer,acc] | teacher[steer,acc] | maxdev")
    for r in rows[-a.tail::2]: print("  ", r)

if __name__ == "__main__":
    main()
