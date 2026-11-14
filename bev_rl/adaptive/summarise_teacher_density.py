"""Per-density teacher table from results/teacher_density/*.json (50 validation seeds 3100-3149, horizon 160 s)."""
import glob, json, numpy as np
rows = {}
for f in glob.glob("bev_rl/adaptive/results/teacher_density/*.json"):
    if f.endswith(".partial.json"): continue
    d = json.load(open(f)); rows[(d["density"], d["teacher"], d["weather"])] = d
order = [("stock_idm", "clear:0.0"), ("safe_g4.0", "clear:0.0"), ("cs0_g4.0", "fog:1.0"), ("cs0_g4.0", "rain:0.6"), ("cs0_g4.0", "rain:1.0")]
print("| density | teacher | weather | n | success@160s | success@80s | crash | off-road | timeout | speed m/s | completion s | Wh/km | episodes with a predicted conflict |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for dens in (0.05, 0.1, 0.2, 0.3):
    for t, w in order:
        d = rows.get((dens, t, w))
        if not d: continue
        E = d["episodes"]; n = len(E); s80 = sum(e["outcome"] == "success" and e["steps"] <= 800 for e in E) / n
        conf = [e for e in E if e["min_ttc"] is not None and e["min_ttc"] < 2.4]; cf = f"{len(conf)/n:.2f}" if t != "stock_idm" else "n/a"
        ct = f"{d['completion_time_s']:.1f}" if d["completion_time_s"] else "-"
        print(f"| {dens} | {t} | {w} | {n} | {d['success']:.2f} | {s80:.2f} | {d['crash']:.2f} | {d['off_road']:.2f} | {d['timeout']:.2f} | {d['mean_speed']:.2f} | {ct} | {d['wh_per_km']:.1f} | {cf} |")
print("\nSuccess split by whether a conflict occurred (safe teachers only): success among episodes WITH conflict / WITHOUT conflict")
for dens in (0.05, 0.1, 0.2, 0.3):
    for t, w in order[1:]:
        d = rows.get((dens, t, w))
        if not d: continue
        a = [e for e in d["episodes"] if e["min_ttc"] is not None and e["min_ttc"] < 2.4]; b = [e for e in d["episodes"] if not (e["min_ttc"] is not None and e["min_ttc"] < 2.4)]
        f = lambda x: f"{sum(e['outcome']=='success' for e in x)}/{len(x)}"
        print(f"  d={dens} {t} {w}: with conflict {f(a)} | without {f(b)}")
