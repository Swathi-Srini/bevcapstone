"""Plot paired clean/fog/rain closed-loop BEV-policy evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="JSON result files emitted by bev_policy.evaluate.")
    parser.add_argument("--output", type=Path, default=Path("bev_policy/outputs/weather/summary.png"))
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    labels = [f"{item['config']['weather']}\n(level {item['config']['weather_level']:.2f})" for item in reports]
    summaries = [item["summary"] for item in reports]
    panels = (
        ("Success rate", "success_rate", 1.0),
        ("Off-road rate", "off_road_rate", 1.0),
        ("Mean speed (km/h)", "mean_speed_mps", None),
        ("Longitudinal control-change proxy", "mean_abs_longitudinal_delta", None),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colours = ("#3977b7", "#b96f2f", "#4c9a5e")
    for axis, (title, key, maximum) in zip(axes.flat, panels):
        values = [summary[key] * 3.6 if key == "mean_speed_mps" else summary[key] for summary in summaries]
        bars = axis.bar(labels, values, color=[colours[index % len(colours)] for index in range(len(values))])
        axis.set_title(title)
        if maximum is not None:
            axis.set_ylim(0.0, maximum)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                      ha="center", va="bottom", fontsize=9)
    fig.suptitle("Paired synthetic-weather perception evaluation", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
