"""Table: caution-aware model with the 800-step limit (80 s) against the 1600-step limit (160 s), plus the proposed model at 800 steps for reference.

Reads bev_rl/adaptive/results/n200_all_weather (800 steps) and bev_rl/adaptive/results/n200_horizon1600 (1600 steps; fog 1.0, rain 0.6, rain 1.0).
Clear weather had no caution-aware timeouts at 800 steps, so its 1600-step result is identical by construction and is not re-run.
Writes report/tables/caution_longer_time_limit.tex
Run from the repository root:   venv\\Scripts\\python.exe report\\make_caution_longer_time_limit.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402
from make_caution_comparison import mcnemar   # noqa: E402

A800 = F.RES / "n200_all_weather"
A1600 = F.RES / "n200_horizon1600"
CELLS = [("fog_1.0", "Fog (intensity 1.0)"), ("rain_0.6", "Rain (intensity 0.6)"), ("rain_1.0", "Rain (intensity 1.0)")]


def ld(p):
    return json.loads(p.read_text())


def main():
    lines = ["\\begin{tabular}{@{}lllrrrrrr@{}}\\toprule",
             "Weather & Model & Step limit & Success & Collision & Left road & Timeout & Speed (m/s) & Energy (Wh per km)\\\\\\midrule"]
    out = {}
    for c, lab in CELLS:
        ov = ld(A800 / f"ov_split__{c}.json")
        c8 = ld(A800 / f"caut_auto__{c}.json")
        c16 = ld(A1600 / f"caut_auto__{c}.json")
        rows = [("Proposed Bird's-Eye-View Driving Model", "800 (80 s)", ov), ("Caution-Study Caution-Aware Driving Model", "800 (80 s)", c8),
                ("Caution-Study Caution-Aware Driving Model", "1600 (160 s)", c16)]
        for i, (m, lim, d) in enumerate(rows):
            lines.append(f"{lab if i == 0 else ''} & {m} & {lim} & {d['success']:.3f} & {d['crash']:.3f} & {d['off_road']:.3f} & {d['timeout']:.3f} & {d['mean_speed']:.2f} & {d['wh_per_km']:.1f}\\\\")
        lines.append("\\midrule")
        sa = {e["seed"]: e["outcome"] == "success" for e in c16["episodes"]}
        sb = {e["seed"]: e["outcome"] == "success" for e in ov["episodes"]}
        x, y, p = mcnemar(sa, sb)
        out[c] = (c16["success"], ov["success"], x, y, p)
        print(c, "caut@1600", c16["success"], "timeout", c16["timeout"], "| ov@800", ov["success"], "| only caut", x, "only ov", y, "p", round(p, 4))
    lines[-1] = "\\bottomrule\\end{tabular}"
    (F.TAB / "caution_longer_time_limit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pl = ["\\begin{tabular}{@{}lrrrr@{}}\\toprule", "Weather & Difference in success (caution-aware at 1600 minus proposed at 800) & Only caution-aware succeeded & Only proposed succeeded & Exact McNemar $p$\\\\\\midrule"]
    for c, lab in CELLS:
        s, o, x, y, p = out[c]
        pl.append(f"{lab} & {s - o:+.3f} & {x} & {y} & {F.pfmt(p)}\\\\")
    pl.append("\\bottomrule\\end{tabular}")
    (F.TAB / "caution_longer_time_limit_paired.tex").write_text("\n".join(pl) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
