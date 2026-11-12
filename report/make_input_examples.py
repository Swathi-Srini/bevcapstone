"""Render one real example of the two bird's-eye-view inputs used by the models (figure fig_birds_eye_view_input_examples).

Takes the evaluation scenario of seed 5169 (clear weather, sparse traffic), drives the rule-based driver until at least one traffic vehicle is
inside the grid window (or 200 steps), then draws
    (a) the teammates' unmodified grid       (values: -1 unknown, 0 free, 0.9 ego, 1.0 detected vehicle),
    (b) the same grid with the map overlay   (adds: 0.5 route centre line, 0.8 road boundary),
exactly as the models receive them. Run from the repository root:
    venv\\Scripts\\python.exe report\\make_input_examples.py
"""
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
import sys
sys.path.insert(0, str(ROOT))   # so that bev_state and bev_rl import when the script is started from any folder


def main():
    import torch, ultralytics  # noqa  (import order matters on Windows)
    import bev_state  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv
    from bev_rl.adaptive.spec_bev import overlay_github_bev

    env = AdaptiveEnv([5169], [("clear", 0.0)], expert=True, sample_seed=0, horizon=800, traffic_density=0.05)
    found = False
    for seed in (5169, 5006, 5060, 5119):          # the first scenario in which a vehicle comes into the grid window
        env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)})
        for t in range(700):
            _, _, te, tr, _ = env.step([0.0, 0.0])
            if t > 20 and len(env.truth_objects()) > 0:
                found = True
                break
            if te or tr:
                break
        if found:
            print("example taken from seed", seed, "step", t)
            break
    plain = env.asm.build_bev_grid(env.truth_objects())
    overlay = overlay_github_bev(plain, env.md)
    env.close()

    # value -> colour (same meaning everywhere in the report)
    levels = [-1.0, 0.0, 0.5, 0.8, 0.9, 1.0]
    colours = ["#b9b8b2", "#161616", "#1baf7a", "#eda100", "#2a78d6", "#e34948"]
    names = ["unknown (not visible)", "free", "route centre line (map)", "road boundary (map)", "ego car", "detected vehicle"]

    def to_index(g):
        idx = np.zeros(g.shape, int)
        for k, v in enumerate(levels):
            idx[np.isclose(g, v, atol=0.05)] = k
        return idx

    cmap = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(-0.5, len(levels) + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, g, title in ((axes[0], plain, "(a) Unmodified grid\n(teammates' encoding)"), (axes[1], overlay, "(b) Grid with map overlay\n(proposed model input)")):
        ax.imshow(to_index(g), cmap=cmap, norm=norm, interpolation="nearest", extent=(-10, 10, -2.5, 17.5), origin="upper")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Lateral position (metres, right positive)")
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-2.5, 5, 12.5, 17.5])
    axes[0].set_ylabel("Distance ahead of the car (metres)")
    axes[1].set_yticklabels([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colours, names)], loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("The two grid encodings given to the models, drawn from scenario 5169 at step 238", fontsize=9, x=0.01, ha="left", y=1.02)
    fig.savefig(OUT / "fig_birds_eye_view_input_examples.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_birds_eye_view_input_examples.png", dpi=170, bbox_inches="tight")
    print("saved", "occupied cells:", int((np.isclose(plain, 1.0, atol=0.05)).sum()), "vehicles in window:", len(env.truth_objects()) if False else "n/a")


if __name__ == "__main__":
    main()
