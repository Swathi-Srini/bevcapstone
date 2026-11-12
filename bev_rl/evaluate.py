"""Policy-agnostic closed-loop evaluation (BC and PPO share exactly this code path).

    python -m bev_rl.evaluate --policy bc  --checkpoint bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt \
        --seeds 1000-1029 --out bev_rl/results/eval_bc
    python -m bev_rl.evaluate --policy ppo --checkpoint <ppo.zip> --bc-stats <bc .pt> --seeds 1000-1029 --out ...

Deterministic actions (policy mean), clear weather, zero traffic. Per-step traces are
saved so every number can be recomputed independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from bev_rl.energy import trace_energy

METRIC_KEYS = [
    "success", "route_completion", "episode_reward", "travel_time_s", "mean_speed_mps", "distance_m",
    "mean_abs_lateral_m", "rms_lateral_m", "max_abs_lateral_m", "mean_abs_heading_rad",
    "mean_abs_accel_mps2", "rms_accel_mps2", "max_decel_mps2", "rms_jerk_mps3", "p95_abs_jerk_mps3",
    "mean_brake_cmd", "brake_fraction", "mean_abs_steer_cmd", "steer_rate_rms",
    "energy_wh", "wh_per_km",
]


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_episode(env, act_fn, scenario_seed: int, weather=None) -> tuple[dict, dict]:
    """Run one episode; return (summary metrics, per-step trace)."""
    opts = {"scenario_seed": scenario_seed}
    if weather is not None:
        opts["weather"] = weather
    obs, _ = env.reset(options=opts)
    dt = env.cfg.dt
    speeds = [env.tel.speed_mps]
    rows = {k: [] for k in ("reward", "lateral", "heading", "accel", "jerk", "steer", "long", "progress")}
    outcome, wall0 = None, time.time()
    while True:
        action = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        speeds.append(info["speed_mps"])
        rows["reward"].append(reward); rows["lateral"].append(info["lateral_m"])
        rows["heading"].append(info["heading_err_rad"]); rows["accel"].append(info["accel_mps2"])
        rows["jerk"].append(info["jerk_mps3"]); rows["steer"].append(float(action[0]))
        rows["long"].append(float(action[1])); rows["progress"].append(info["progress_m"])
        if terminated or truncated:
            outcome = info["outcome"]
            break
    a = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    steps = len(a["reward"])
    energy = trace_energy(np.asarray(speeds), dt, env.energy_params)
    # Independent cross-check of the in-env running energy against the offline recomputation.
    if abs(energy["traction_pos_j"] - info["energy_pos_j"]) > 1e-6 * max(1.0, energy["traction_pos_j"]):
        raise RuntimeError("in-env energy and offline trace energy disagree")
    brake = np.clip(-a["long"], 0.0, None)
    distance = energy["distance_m"]
    summary = {
        "scenario_seed": scenario_seed, "weather_kind": info["weather_kind"], "weather_level": info["weather_level"],
        "outcome": outcome, "steps": steps,
        "success": float(outcome == "success"),
        "route_completion": float(info["route_completion"]),
        "episode_reward": float(a["reward"].sum()),
        "travel_time_s": steps * dt,
        "mean_speed_mps": float(np.mean(speeds[1:])),
        "distance_m": distance,
        "mean_abs_lateral_m": float(np.mean(np.abs(a["lateral"]))),
        "rms_lateral_m": float(np.sqrt(np.mean(a["lateral"] ** 2))),
        "max_abs_lateral_m": float(np.max(np.abs(a["lateral"]))),
        "mean_abs_heading_rad": float(np.mean(np.abs(a["heading"]))),
        "mean_abs_accel_mps2": float(np.mean(np.abs(a["accel"]))),
        "rms_accel_mps2": float(np.sqrt(np.mean(a["accel"] ** 2))),
        "max_decel_mps2": float(-np.min(a["accel"])),
        "rms_jerk_mps3": float(np.sqrt(np.mean(a["jerk"] ** 2))),
        "p95_abs_jerk_mps3": float(np.percentile(np.abs(a["jerk"]), 95)),
        "mean_brake_cmd": float(brake.mean()),
        "brake_fraction": float(np.mean(brake > 0.05)),
        "mean_abs_steer_cmd": float(np.mean(np.abs(a["steer"]))),
        "steer_rate_rms": float(np.sqrt(np.mean(np.diff(a["steer"]) ** 2))) if steps > 1 else 0.0,
        "energy_wh": energy["traction_pos_wh"],
        "wh_per_km": energy["wh_per_km"] if distance >= 50.0 else float("nan"),   # undefined on very short runs
        "n_unique_bev_so_far": int(info["n_unique_bev"]), "bev_frames_so_far": int(info["bev_frames"]),
        "wall_s": time.time() - wall0,
    }
    trace = {**{k: v for k, v in a.items()}, "speed": np.asarray(speeds)}
    return summary, trace


def bc_act_fn(ckpt: dict):
    import torch
    from bev_policy.model import BEVScalarPolicy
    model = BEVScalarPolicy(int(ckpt["in_channels"]), 6, int(ckpt.get("weather_dim", 0)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def act(obs):
        w = torch.from_numpy(obs["weather"][None]) if "weather" in obs and model.weather_dim else None
        with torch.no_grad():
            return model(torch.from_numpy(obs["bev"][None]), torch.from_numpy(obs["scalar"][None]), w).squeeze(0).numpy()
    return act


def ppo_act_fn(model):
    def act(obs):
        return model.predict(obs, deterministic=True)[0]
    return act


def evaluate_policy(env, act_fn, seeds, out_dir: Path, tag: str, conditions=(None,)) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, seed in [(c, sd) for c in conditions for sd in seeds]:
        summary, trace = run_episode(env, act_fn, seed, cond)
        name = f"{tag}_{summary['weather_kind']}{summary['weather_level']}_seed_{seed}"
        np.savez_compressed(out_dir / f"{name}.npz", **trace)
        rows.append(summary)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, flush=True)
    with (out_dir / f"{tag}_episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("bc", "ppo"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bc-stats", type=Path, default=None,
                   help="BC checkpoint supplying scalar normalisation (the observation contract).")
    p.add_argument("--seeds", default="1000-1029")
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--yolo-weights", default="weights/yolov8n.pt")
    p.add_argument("--conditions", default="clear:0", help="Evaluation weather grid, e.g. 'clear:0,fog:0.3,rain:0.3'.")
    p.add_argument("--weather-input", choices=("explicit", "zeroed"), default="explicit")
    p.add_argument("--cam-scale", type=float, default=0.5)
    args = p.parse_args()

    from bev_rl.factory import make_v2_env
    from bev_rl.policy import load_bc_checkpoint
    if args.bc_stats is None:
        if args.policy != "bc":
            raise SystemExit("--bc-stats (the BC checkpoint whose scalar normalisation the PPO policy was trained with) is required for ppo.")
        args.bc_stats = args.checkpoint   # a BC checkpoint carries its own scalar normalisation
    stats = load_bc_checkpoint(args.bc_stats)
    seeds = parse_seeds(args.seeds)
    conds = [(c.split(":")[0], float(c.split(":")[1])) for c in args.conditions.split(",")]
    env = make_v2_env(seeds, stats["scalar_mean"], stats["scalar_std"], weather_conditions=conds,
                      weather_input=args.weather_input, horizon=args.horizon, sequential=True, cam_scale=args.cam_scale)
    try:
        if args.policy == "bc":
            act = bc_act_fn(load_bc_checkpoint(args.checkpoint))
        else:
            from stable_baselines3 import PPO
            from bev_rl.policy import BEVActorCritic  # noqa: F401  (needed for unpickling)
            act = ppo_act_fn(PPO.load(args.checkpoint, device="cpu"))
        rows = evaluate_policy(env, act, seeds, args.out, args.tag or args.policy, conds)
    finally:
        env.close()
    per = {}
    for kind, lvl in conds:
        rs = [r for r in rows if r["weather_kind"] == kind and r["weather_level"] == lvl]
        per[f"{kind}:{lvl}"] = {"n": len(rs), "mean": {k: float(np.nanmean([r[k] for r in rs])) for k in METRIC_KEYS},
                                 "outcomes": {o: sum(r["outcome"] == o for r in rs) for o in {r["outcome"] for r in rs}}}
    (args.out / f"{args.tag or args.policy}_summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "seeds": seeds, "weather_input": args.weather_input,
                    "n": len(rows), "per_condition": per}, indent=2))
    print(json.dumps(per, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
