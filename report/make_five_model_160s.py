"""Tables: all five models with the 1600-step (160 s) limit, 200 episodes per cell, and paired tests of the caution-aware model.

Method of the 160 s numbers: a longer limit changes only episodes that timed out at 800 steps, so those were re-run at 1600 steps and every other episode keeps its 800-step result
(proposed model, clear: 1 episode, seed 5151; image-only model, clear: 13 episodes; the two baselines, the caution-aware model in clear weather and every model in fog/rain
except the image-only one had no timeouts; caution-aware fog/rain were re-run completely earlier, results/n200_horizon1600).
Reads bev_rl/adaptive/results/n200_horizon1600/five_model_160s_summary.json (+ the episode files).  Writes tables/five_model_160s.tex and tables/caution_vs_proposed_160s_paired.tex.
Run from the repository root:   python report/make_five_model_160s.py
"""
import json
from math import comb, sqrt
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "bev_rl" / "adaptive" / "results"
T = Path(__file__).resolve().parent / "tables"
S = json.loads((R / "n200_horizon1600" / "five_model_160s_summary.json").read_text())
W = [("clear_0.0", "Clear"), ("fog_1.0", "Fog (intensity 1.0)"), ("rain_0.6", "Rain (intensity 0.6)"), ("rain_1.0", "Rain (intensity 1.0)")]
M = [("repo_bc_clear", "Baseline Driving Model Trained Without Traffic"), ("repo_bc_sparse", "Baseline Driving Model Trained With Traffic"), ("ov_split", "Proposed Bird's-Eye-View Driving Model"),
     ("caut_auto", "Caution-Study Caution-Aware Driving Model"), ("vcam", "Image-Only Driving Model (front camera)")]
BS = "\\"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    x = sum(1 for k in a if a[k] and not b[k]); y = sum(1 for k in a if b[k] and not a[k]); n = x + y
    return x, y, (min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0)


def ok(f): return {e["seed"]: e["outcome"] == "success" for e in json.loads(Path(f).read_text())["episodes"]}


def write(path, lines): path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


EOL = BS * 2
L = [BS + "begin{tabular}{@{}llrrrrrrr@{}}" + BS + "toprule",
     "Weather & Model & Success & 95" + BS + "% interval & Collision & Left road & Timeout & Speed (m/s) & Energy (Wh per km)" + EOL + BS + "midrule"]
for wi, (w, wl) in enumerate(W):
    for i, (k, mn) in enumerate(M):
        d = S[f"{k}|{w}"]; lo, hi = wilson(round(d["success"] * 200), 200)
        L.append(f"{wl if i == 0 else ''} & {mn} & {d['success']:.3f} & [{lo:.3f}, {hi:.3f}] & {d['crash']:.3f} & {d['off']:.3f} & {d['timeout']:.3f} & {d['speed']:.2f} & {d['wh']:.1f}" + EOL)
    if wi < 3: L.append(BS + "midrule")
L.append(BS + "bottomrule" + BS + "end{tabular}"); write(T / "five_model_160s.tex", L)

P = [BS + "begin{tabular}{@{}lllrrrr@{}}" + BS + "toprule",
     "Weather & Comparison & Difference in success & Only first succeeded & Only second succeeded & Exact McNemar $p$" + EOL + BS + "midrule"]
for wi, (w, wl) in enumerate(W):
    ov = ok(R / f"n200_all_weather/ov_split__{w}.json")
    if w == "clear_0.0": ov[5151] = True                                                   # re-run at 1600 steps: success
    ca = ok(R / (f"n200_horizon1600/caut_auto__{w}.json" if w != "clear_0.0" else f"n200_all_weather/caut_auto__{w}.json"))
    c1 = ok(R / f"sparse_always_cautious/h1600/caut_c1__{w}.json")
    for i, (lab, a, b) in enumerate([("caution-aware minus proposed", ca, ov), ("always-cautious control minus caution-aware", c1, ca)]):
        x, y, p = mcnemar(a, b); dif = (x - y) / 200
        P.append(f"{wl if i == 0 else ''} & {lab} & {dif:+.3f} & {x} & {y} & {p:.3f}" + EOL)
    if wi < 3: P.append(BS + "midrule")
P.append(BS + "bottomrule" + BS + "end{tabular}"); write(T / "caution_vs_proposed_160s_paired.tex", P)
print("\n".join(P))
