"""Build the LaTeX tables of the report from the saved result files (companion of make_report_figures.py).

Run from the repository root:
    venv\\Scripts\\python.exe report\\make_report_tables.py
Output: report/tables/*.tex (included by the sections with \\input).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402  (reuses the naming, the data loading and the Wilson interval)

ROOT, TAB, RES = F.ROOT, F.TAB, F.RES
NAME, FINAL, wilson = F.NAME, F.FINAL, F.wilson


def tex(s):
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


def write(name, lines):
    (TAB / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table", name)


PRETTY_POLICY = {"DAgger, map-overlay BEV (ov_split)": "Headline model (map-overlay view)",
                 "DAgger, GitHub BEV (base_a)": "Dataset-aggregation student, unmodified view",
                 "Repo BC (sparse traffic)": "Baseline behaviour cloning, sparse-traffic demonstrations",
                 "Repo BC (clear, no traffic)": "Baseline behaviour cloning, no-traffic demonstrations"}


def table_headline():
    order = ["safe_teacher", "base_ovsplit", "ctrl_nocaut", "stock_idm", "repo_bc_clear", "repo_bc_sparse"] + (["image_only"] if "image_only" in FINAL else [])
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule",
             "Model & Success & 95\\% interval & Collision & Left road & Timeout & Speed (m/s) & Energy (Wh per km)\\\\\\midrule"]
    for a in order:
        d = FINAL[a]
        lo, hi = wilson(round(d["success"] * d["n"]), d["n"])
        lines.append(f"{tex(NAME[a].replace(chr(10), ' '))} & {d['success']:.3f} & [{lo:.3f}, {hi:.3f}] & {d['crash']:.3f} & {d['off_road']:.3f} & {d['timeout']:.3f} & {d['mean_speed']:.2f} & {d['wh_per_km']:.1f}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("headline_200_episodes.tex", lines)


def table_dropout():
    rows = list(csv.DictReader(open(ROOT / "report" / "data" / "sparse_weather_robustness_detector_dropout_table.csv", encoding="utf-8")))
    lines = ["\\begin{tabular}{@{}llrrr@{}}\\toprule", "Model & Weather & Share of vehicles kept & Success (40 episodes) & 95\\% interval\\\\\\midrule"]
    last = None
    for r in rows:
        name = PRETTY_POLICY.get(r["policy"], r["policy"])
        lines.append(f"{tex(name) if name != last else ''} & {r['weather']} & {r['detector keep prob']} & {r['success']} & {tex(r['success 95% CI'])}\\\\")
        last = name
    lines.append("\\bottomrule\\end{tabular}")
    write("weather_detector_dropout_40_episodes.tex", lines)


def table_ablation():
    names = {"ov_split": "Overlay grid, three vehicle-state inputs (proposed model)", "base_a": "Unmodified grid, six vehicle-state inputs", "ov_both": "Overlay grid, six vehicle-state inputs",
             "ov_bev_only": "Overlay grid, no vehicle-state inputs", "arch_scalar_only": "No grid, six vehicle-state inputs"}
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule", "Input design & Traffic & Success & 95\\% interval & Collision & Left road & Energy (Wh per km)\\\\\\midrule"]
    for run in ["ov_split", "base_a", "ov_both", "ov_bev_only", "arch_scalar_only"]:
        for d in json.loads((RES / f"final_{run}_best.json").read_text()):
            lo, hi = d["success_ci95"]
            lines.append(f"{names[run]} & {d['density']} & {d['success']:.2f} & [{lo:.2f}, {hi:.2f}] & {d['crash']:.2f} & {d['off_road']:.2f} & {d['wh_per_km']:.0f}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("input_design_ablation_10_episodes.tex", lines)


def table_caution():
    names = {"ctrl": "Control: caution input fixed to 0", "caut_c0": "Caution-conditioned network, caution 0",
             "caut_auto": "Caution-conditioned network, caution from weather", "base_ovsplit": "Headline model"}
    lines = ["\\begin{tabular}{@{}llrrrr@{}}\\toprule", "Model & Weather & Success & Timeout & Speed (m/s) & Energy (Wh per km)\\\\\\midrule"]
    for a in ["base_ovsplit", "ctrl", "caut_c0", "caut_auto"]:
        cells = json.loads((RES / "caution" / f"{a}.json").read_text())["cells"]
        for w, v in cells.items():
            lines.append(f"{names[a]} & {w.replace(':', ' ')} & {v['success']:.3f} & {v['timeout']:.3f} & {v['mean_speed']:.2f} & {v['wh_per_km']:.1f}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("caution_study_40_episodes.tex", lines)


def table_pose():
    rows = list(csv.DictReader(open(ROOT / "report" / "data" / "localisation_error_sensitivity_table.csv", encoding="utf-8")))
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Model & Lateral error (m) & Heading error (rad) & Success (40 episodes) & 95\\% interval & Left road\\\\\\midrule"]
    for r in rows:
        lines.append(f"{tex(PRETTY_POLICY.get(r['policy'], r['policy']))} & {r['lateral error (m)']} & {r['heading error (rad)']} & {r['success']} & {tex(r['success 95% CI'])} & {r['off-road']}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("localisation_error_40_episodes.tex", lines)


def table_rl():
    """Reinforcement-learning runs: validation success at the start and at the selected checkpoint."""
    runs = [("rl_plain", "Base reward"), ("rl_energy", "Base reward + energy term"), ("rl_safe", "Base reward + safe speed + scaled crash penalty"),
            ("rl_head", "Base reward + headway term"), ("rl_full", "All reward terms")]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}\\toprule", "Reward combination & Validation at step 0 & Validation at step 150\\,000 & Best validation & Energy (Wh per km) at step 150\\,000\\\\\\midrule"]
    for run, label in runs:
        log = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / run / "log.json").read_text())
        best = max(r["val_success"] for r in log)
        lines.append(f"{label} & {log[0]['val_success']:.3f} & {log[-1]['val_success']:.3f} & {best:.3f} & {log[-1]['wh_per_km']:.0f}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("reinforcement_learning_reward_sweep.tex", lines)


def table_ppo_versus_start():
    """Paired comparison of reward-version-3 PPO against its own starting point and the headline model, calibrated perception, 200 episodes."""
    import math

    def eps(m, w):
        d = json.loads((RES / "degraded_eval" / "cal_d0.05" / f"{m}__{w}.json").read_text())
        return {e["seed"]: e["outcome"] == "success" for e in d["episodes"]}, d

    def mcnemar(b, c):
        n = b + c
        return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}\\toprule", "Weather & Start (degraded-perception student) & After PPO & Difference & Only start succeeded / only PPO succeeded & Exact McNemar $p$\\\\\\midrule"]
    for w, lab in [("clear_0.0", "Clear"), ("fog_1.0", "Fog 1.0"), ("rain_0.6", "Rain 0.6"), ("rain_1.0", "Rain 1.0")]:
        a, da = eps("deg_base", w)
        b, db = eps("ppo_plain", w)
        x = sum(a[s] and not b[s] for s in a)
        y = sum(b[s] and not a[s] for s in a)
        p = mcnemar(x, y)
        ps = "$<0.0001$" if p < 0.0001 else f"{p:.4f}"
        lines.append(f"{lab} & {da['success']:.3f} & {db['success']:.3f} & {db['success'] - da['success']:+.3f} & {x} / {y} & {ps}\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("ppo_versus_its_start_calibrated_perception.tex", lines)


def table_headline_failures():
    """The headline model's failed episodes (200-episode block) and what every other model did on the same seeds."""
    order = ["base_ovsplit", "safe_teacher", "stock_idm", "ctrl_nocaut", "repo_bc_clear", "repo_bc_sparse"]
    ep = {a: {e["seed"]: e for e in FINAL[a]["episodes"]} for a in order}
    short = {"success": "success", "crash": "collision", "off_road": "left road", "timeout": "timeout"}
    seeds = sorted(s for s, e in ep["base_ovsplit"].items() if e["outcome"] != "success")
    lines = ["\\begin{tabular}{@{}rlllllll@{}}\\toprule",
             "Seed & Headline model (step of end) & Collision-aware teacher & Stock driver model & Caution control & Baseline, no traffic & Baseline, sparse traffic\\\\\\midrule"]
    for s in seeds:
        h = ep["base_ovsplit"][s]
        cells = [f"{short[ep[a][s]['outcome']]}" for a in order[1:]]
        lines.append(f"{s} & {short[h['outcome']]} ({h['steps']}) & " + " & ".join(cells) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    write("headline_model_failed_episodes.tex", lines)


def image_only_numbers():
    """LaTeX macros with the image-only model's 200-episode numbers (used in the text of both documents)."""
    d = FINAL.get("image_only")
    macros = {"ImOnSuccess": "not yet available", "ImOnLow": "--", "ImOnHigh": "--", "ImOnCollision": "--", "ImOnOffroad": "--", "ImOnTimeout": "--",
              "ImOnSpeed": "--", "ImOnEnergy": "--"}
    if d is not None:
        lo, hi = wilson(round(d["success"] * d["n"]), d["n"])
        macros = {"ImOnSuccess": f"{d['success']:.3f}", "ImOnLow": f"{lo:.3f}", "ImOnHigh": f"{hi:.3f}", "ImOnCollision": f"{d['crash']:.3f}",
                  "ImOnOffroad": f"{d['off_road']:.3f}", "ImOnTimeout": f"{d['timeout']:.3f}", "ImOnSpeed": f"{d['mean_speed']:.2f}", "ImOnEnergy": f"{d['wh_per_km']:.1f}"}
    write("image_only_numbers.tex", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()])


if __name__ == "__main__":
    image_only_numbers()
    table_headline_failures()
    table_headline()
    table_dropout()
    table_ablation()
    table_caution()
    table_pose()
    table_rl()
    table_ppo_versus_start()
