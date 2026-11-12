"""n=40 ALL-MODELS table: the six presented models (from n40_table1_full_results.csv) + base_a (earlier DAgger reference, GitHub BEV, no overlay; weather cells exist at n=40)
+ the two teachers (clear only: teachers see ground truth, so detector dropout does not apply). Read-only on results; writes paper/n40_story/tables/n40_table7_all_models.{md,csv}.
Run from repo root:  python paper/make_n40_all_models_table.py   (numpy + stdlib only)"""
import csv, json
from pathlib import Path
import numpy as np

R = Path("bev_rl/adaptive/results"); T = Path("paper/n40_story/tables")
WX = ["clear:0.0", "fog:1.0", "rain:0.6", "rain:1.0"]; WL = {"clear:0.0": "clear", "fog:1.0": "fog 1.0", "rain:0.6": "rain 0.6", "rain:1.0": "rain 1.0"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def cell(name, harness, wx, eps, wh_key):
    n = len(eps); oc = [e["outcome"] for e in eps]; k = oc.count("success"); lo, hi = wilson(k, n)
    whs = [e[wh_key] for e in eps if wh_key and e.get(wh_key) is not None]
    return [name, harness, wx, n, f"{k/n:.3f}", f"[{lo:.2f}, {hi:.2f}]", f"{oc.count('crash')/n:.3f}", f"{oc.count('off_road')/n:.3f}", f"{oc.count('timeout')/n:.3f}",
            f"{np.mean([e['speed'] for e in eps]):.2f}", f"{np.mean(whs):.1f}" if whs else "not recorded", f"{np.mean([e['route'] for e in eps]):.3f}"]


rows = list(csv.reader(open(T / "n40_table1_full_results.csv", encoding="utf-8")))
header, body = rows[0], rows[1:]
extra = []
d = json.loads((R / "sparse_traffic_weather_robustness_base_a_d0.05.json").read_text())
for w in WX: extra.append(cell("DAgger, GitHub BEV (base_a)", "A", WL[w], [e for e in d["episodes"] if e["weather"] == w], None))
for key, label in (("stock", "Stock IDM (teacher, rule-based)"), ("no_lane_change", "IDM, lane change off (rule-based)")):
    e = json.loads((R / "expert_ceiling_lane_change_ablation_d0.05.json").read_text())[key]
    extra.append(cell(label, "ceiling", "clear only", [{"outcome": x["outcome"], "speed": x["speed"], "route": x["route"], "wh": x["wh_km"]} for x in e["episodes"]], "wh"))
e = json.loads((R / "safe_teacher_ceiling_d0.05_soft2.4_hard1.3.json").read_text())["safe_teacher"]
extra.append(cell("SafeTeacher (rule-based, TTC layer)", "ceiling", "clear only", [{"outcome": x["outcome"], "speed": x["speed"], "route": x["route"], "wh": x["wh_km"]} for x in e["episodes"]], "wh"))
allrows = body + extra
cap = ("n=40 (seeds 1010-1049, density 0.05, horizon 800): every model with weather cells, plus the rule-based teachers (clear only; they see ground truth). "
       "Harness A = robustness script (no energy), B = caution script, ceiling = teacher-ceiling script. Weather = detector dropout only.")
with open(T / "n40_table7_all_models.csv", "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows([header] + allrows)
(T / "n40_table7_all_models.md").write_text(f"**{cap}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in allrows) + "\n", encoding="utf-8")
print(len(allrows), "rows")
print((T / "n40_table7_all_models.md").read_text(encoding="utf-8"))
