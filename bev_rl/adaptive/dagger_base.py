"""Strong base policy by DAgger (policy side only).

Learner = repo CNN+MLP (bev_policy.model.BEVScalarPolicy) on the repo input contract: 1-channel GitHub BEV (unmodified
build_bev_grid, ideal-detector objects) + 6-D scalar. Expert = MetaDrive IDM queried at the learner's own states
(IDMPolicy(agent).act()). Model selection uses VALIDATION seeds 3100-3109 only; the reserved seeds 1000-1009 are touched
only by the final evaluation script.

    python -m bev_rl.adaptive.dagger_base --name base_a --iters 6
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

DENSITIES = [0.0, 0.05, 0.1, 0.2, 0.3]
CFG = {"safe": False, "use_bev": True, "mask": np.ones(6, np.float32), "sigma": np.zeros(6, np.float32), "overlay": False}      # input ablation: hidden inputs are replaced by constants


def prep(torch, bev, raw, mu, sd):
    b = torch.from_numpy(bev[None, None] if CFG["use_bev"] else np.zeros_like(bev)[None, None])
    z = (((raw - mu) / sd) * CFG["mask"])[None].astype(np.float32)
    return b, torch.from_numpy(z)

VAL_SEEDS = list(range(3100, 3110))


def rollout(env, net, mu, sd, seed, beta, noise, rng, torch, SCALE, expert_only=False):
    from metadrive.policy.idm_policy import IDMPolicy
    o, _ = env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)})
    pol = IDMPolicy(env.md.agent, env.md.engine.global_seed)
    st = None
    if CFG["safe"]:       # collision-aware privileged teacher (safe_teacher.py): IDM, lane change off, + predictive braking layer
        from bev_rl.adaptive.safe_teacher import SafeTeacher
        pol.enable_lane_change = False; st = SafeTeacher(pol)
    B, S, Y = [], [], []
    while True:
        bev = env.bev(); raw = (o[:6] * SCALE).astype(np.float32)
        raw = (raw + rng.normal(0, 1, 6) * CFG["sigma"]).astype(np.float32)      # localisation-noise study (approved item 2)
        ea = np.clip(np.asarray(st.act(env) if st else pol.act(), np.float32)[:2], -1, 1)
        B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); Y.append(ea)
        if expert_only or rng.random() < beta or net is None:
            a = ea
        else:
            with torch.no_grad():
                a = net(*prep(torch, bev, raw, mu, sd)).squeeze(0).numpy()
        a = np.clip(a + rng.normal(0, noise, 2), -1, 1).astype(np.float32)
        o, r, te, tr, info = env.step(a)
        if te or tr:
            break
    return B, S, Y, info


def train(net, B, S, Y, mu, sd, torch, epochs, lr, rng):
    opt = torch.optim.Adam(net.parameters(), lr)
    Bt = torch.from_numpy(np.stack(B)).float().div(10).unsqueeze(1)
    if not CFG["use_bev"]:
        Bt = torch.zeros_like(Bt)
    St = torch.from_numpy((((np.stack(S) - mu) / sd) * CFG["mask"]).astype(np.float32)); Yt = torch.from_numpy(np.stack(Y))
    idx = np.arange(len(Bt)); net.train()
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            loss = ((net(Bt[j], St[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return float(loss)


def evaluate(env_factory, net, mu, sd, seeds, dens, torch, SCALE):
    env = env_factory(dens)
    out = []
    nrng = np.random.default_rng(12345)
    for s in seeds:
        o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
        while True:
            with torch.no_grad():
                a = net(*prep(torch, env.bev(), (o[:6] * SCALE + nrng.normal(0, 1, 6) * CFG["sigma"]).astype(np.float32), mu, sd)).squeeze(0).numpy()
            o, r, te, tr, i = env.step(a)
            if te or tr:
                break
        out.append(i["outcome"])
    env.close()
    return {"success": out.count("success") / len(out), "crash": out.count("crash") / len(out), "off_road": out.count("off_road") / len(out)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--iters", type=int, default=6)
    p.add_argument("--eps-per-iter", type=int, default=40); p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bev", action="store_true", help="ablation: BEV replaced by zeros")
    p.add_argument("--scalar-keep", default="0,1,2,3,4,5", help="ablation: indices of the 6-D scalar that stay visible ('' = none)")
    p.add_argument("--overlay", action="store_true", help="approved item 1: fill GitHub's reserved route(0.5)/boundary(0.8) values from the HD map")
    p.add_argument("--safe-teacher", action="store_true", help="label with SafeTeacher (IDM + predictive braking layer) instead of stock IDM")
    p.add_argument("--densities", default=",".join(map(str, DENSITIES)), help="training traffic densities")
    p.add_argument("--val-densities", default="0.05,0.2")
    p.add_argument("--input-noise", dest="input_noise", default="0,0,0,0,0,0", help="std of Gaussian noise added to the 6 scalar inputs (raw units: m/s,-,m,rad,rad/m,m)")
    a = p.parse_args()
    CFG["overlay"] = a.overlay; CFG["safe"] = a.safe_teacher
    dens_train = [float(x) for x in a.densities.split(",")]; dens_val = [float(x) for x in a.val_densities.split(",")]
    CFG["sigma"] = np.array([float(x) for x in a.input_noise.split(",")], np.float32)
    CFG["use_bev"] = not a.no_bev
    CFG["mask"] = np.array([1.0 if str(i) in a.scalar_keep.split(",") else 0.0 for i in range(6)], np.float32)
    import torch, ultralytics  # noqa
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.env import AdaptiveEnv, SCALE
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    train_seeds = list(range(2000, 3000))
    envs = {}

    def factory(d):
        e = AdaptiveEnv(train_seeds, [("clear", 0.0)], horizon=600, traffic_density=d, sample_seed=a.seed)
        e.overlay = CFG["overlay"]
        return e
    net = BEVScalarPolicy(1, 6, 0)
    B, S, Y = [], [], []
    mu = sd = None
    best, log, t0 = (-1.0, 1.0), [], time.time()
    for it in range(a.iters + 1):
        beta = 1.0 if it == 0 else 0.5 ** it
        stats = []
        for d in dens_train:
            env = factory(d)
            for k in range(max(1, a.eps_per_iter // len(dens_train))):
                seed = int(rng.choice(train_seeds))
                b, s, y, info = rollout(env, None if it == 0 else net, mu, sd, seed, beta, a.noise, rng, torch, SCALE)
                B += b; S += s; Y += y; stats.append(info["outcome"])
            env.close()
        if mu is None:
            Sa = np.stack(S); mu, sd = Sa.mean(0).astype(np.float32), np.maximum(Sa.std(0), 1e-6).astype(np.float32)
        loss = train(net, B, S, Y, mu, sd, torch, a.epochs if it else 3 * a.epochs, 1e-3, rng)
        val = {f"d{d}": evaluate(factory, net, mu, sd, VAL_SEEDS, d, torch, SCALE) for d in dens_val}
        score = (np.mean([v["success"] for v in val.values()]), np.mean([v["crash"] + v["off_road"] for v in val.values()]))
        rec = {"iter": it, "beta": beta, "samples": len(B), "train_loss": loss, "rollout_outcomes": {o: stats.count(o) for o in set(stats)},
               "val": val, "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save({"model_state": net.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist(), "in_channels": 1, "iter": it, "val": val},
                   out / f"iter_{it}.pt")
        if (score[0], -score[1]) > (best[0], -best[1]):
            best = score; torch.save(torch.load(out / f"iter_{it}.pt", weights_only=True), out / "best.pt")
            (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
