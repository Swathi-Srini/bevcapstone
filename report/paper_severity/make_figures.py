"""Paper figures for the severity-adaptive system.  Read-only on results; writes PNG+PDF into report/paper_severity/figures/.
Run from D:\\bev_Fresh\\bevcapstone:  venv\\Scripts\\python.exe -I report\\paper_severity\\make_figures.py [--episodes DIR ...]
Colour convention follows the architecture diagram: A = grey, C (new, adaptive) = orange, fog = blue, rain = purple.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/paper_severity/figures"; OUT.mkdir(parents=True, exist_ok=True)
EST = ROOT / "bev_rl/adaptive/results/severity_estimator/eval_heldout.json"
RAW = OUT / "raw"
GREY, ORG, ORG_F, BLUE, PURP = "#666666", "#d79b00", "#ffe6cc", "#6c8ebf", "#9673a6"
CONDS = ["clear:0.0", "fog:1.0", "rain:1.0"]; NAMES = {"clear:0.0": "Clear", "fog:1.0": "Fog 1.0", "rain:1.0": "Rain 1.0"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif", "axes.titlesize": 8.5})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig); print("wrote", name)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(dirs):
    R = {}
    for d in dirs:
        for f in Path(d).glob("episodes/*.json"):
            e = json.loads(f.read_text()); R[(e["arm"], e["weather"], e["seed"])] = e
    return R


def fig_calibration():
    pc = json.loads(EST.read_text())["per_condition"]
    fig, ax = plt.subplots(figsize=(3.4, 3.1)); ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", label="ideal $\\hat c=c$")
    for kind, col, mk in (("fog", BLUE, "o"), ("rain", PURP, "s")):
        pts = sorted((v["c_true"], v["c_hat_mean"], v["probe"]) for k, v in pc.items() if k.startswith(kind))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.0, alpha=0.7)
        for x, y, probe in pts: ax.scatter(x, y, s=28, marker=mk, facecolor="white" if probe else col, edgecolor=col, lw=1.2, zorder=3)
        ax.scatter([], [], marker=mk, color=col, label=kind)
    ax.scatter([0], [pc["clear:0.0"]["c_hat_mean"]], s=28, marker="^", color=GREY, zorder=3, label="clear")
    ax.scatter([], [], marker="o", facecolor="white", edgecolor="k", label="never-fitted probe level")
    ax.set_xlabel("true severity $c$"); ax.set_ylabel("mean estimated caution $\\hat c$"); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left"); save(fig, "fig_estimator_calibration")


def fig_weather_frames():
    items = [("clear_f60", "Clear"), ("rain06_f60", "Rain 0.6"), ("rain1_f60", "Rain 1.0"), ("fog1_f60", "Fog 1.0")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.85))
    for ax, (f, t) in zip(axs, items):
        im = cv2.cvtColor(cv2.imread(str(RAW / f"{f}.png")), cv2.COLOR_BGR2RGB); ax.imshow(im); ax.set_title(t); ax.axis("off")
    fig.tight_layout(pad=0.3); save(fig, "fig_weather_frames")


def fig_speed_energy(R):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3)); x = np.arange(3)
    def arr(arm, c, key): return np.array([R[(arm, c, s)][key] for s in sorted(s for (a, w, s) in R if a == arm and w == c and ("C" if arm == "A" else "A", c, s) in R)])
    for i, (key, ttl, yl) in enumerate([("mean_speed", "(a) Mean speed", "m/s"), ("wh_per_km", "(b) Traction energy, both-success pairs", "Wh/km"), ("success", "(c) Success rate", "")]):
        for j, (arm, col, lab) in enumerate([("A", GREY, "A non-adaptive"), ("C", ORG, "C adaptive")]):
            vals, errs = [], []
            for c in CONDS:
                seeds = sorted(s for (a, w, s) in R if a == "A" and w == c and ("C", c, s) in R)
                if key == "success":
                    k = sum(R[(arm, c, s)]["outcome"] == "success" for s in seeds); lo, hi = wilson(k, len(seeds)); vals.append(k / len(seeds)); errs.append([k / len(seeds) - lo, hi - k / len(seeds)])
                else:
                    use = seeds if key == "mean_speed" else [s for s in seeds if R[("A", c, s)]["outcome"] == "success" and R[("C", c, s)]["outcome"] == "success"]
                    v = np.array([R[(arm, c, s)][key] for s in use]); vals.append(v.mean()); se = 1.96 * v.std(ddof=1) / math.sqrt(len(v)); errs.append([se, se])
            ax[i].bar(x + (j - 0.5) * 0.38, vals, 0.36, yerr=np.array(errs).T, color=ORG_F if arm == "C" else "#eeeeee", edgecolor=col, lw=1.2, capsize=2, label=lab, error_kw={"lw": 0.8})
        ax[i].set_xticks(x); ax[i].set_xticklabels([NAMES[c] for c in CONDS]); ax[i].set_title(ttl); ax[i].set_ylabel(yl)
    ax[2].set_ylim(0, 1.08); ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.text(0.5, -0.04, "Error bars: 95% interval (Wilson for success; mean $\\pm$1.96 SE for speed and energy). $n$ = paired seeds per condition.", ha="center", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_speed_energy_v2")


def fig_speed_time(R):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    for j, c in enumerate(CONDS):
        for arm, col in (("A", GREY), ("C", ORG)):
            eps = [e for (a, w, s), e in R.items() if a == arm and w == c]; L = max(len(e["trace"]["speed"]) for e in eps)
            M = np.full((len(eps), L), np.nan)
            for i, e in enumerate(eps): v = e["trace"]["speed"]; M[i, :len(v)] = v
            alive = (~np.isnan(M)).sum(0); ok = alive >= max(5, len(eps) // 2); t = np.arange(L) * 0.1
            q1, med, q3 = (np.nanpercentile(M, q, axis=0) for q in (25, 50, 75))
            axs[0, j].fill_between(t[ok], q1[ok], q3[ok], color=col, alpha=0.18, lw=0); axs[0, j].plot(t[ok], med[ok], color=col, lw=1.3, label=("A non-adaptive" if arm == "A" else "C adaptive"))
            if arm == "C":
                H = np.full((len(eps), L), np.nan)
                for i, e in enumerate(eps): v = e["trace"]["c_hat"]; H[i, :len(v)] = v
                axs[1, j].plot(t[ok], np.nanmedian(H, axis=0)[ok], color=ORG, lw=1.3)
        axs[1, j].axhline(R[("C", c, next(s for (a, w, s) in R if a == "C" and w == c))]["c_true"], color="#999999", ls="--", lw=0.8)
        axs[0, j].set_title(NAMES[c]); axs[1, j].set_xlabel("time (s)"); axs[0, j].set_ylim(0, 12); axs[1, j].set_ylim(-0.03, 1.1)
        if j == 0: axs[0, j].set_ylabel("speed (m/s)"); axs[1, j].set_ylabel("$\\hat c$ (C)")
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.text(0.5, -0.02, "Median and inter-quartile band over paired seeds, shown while at least half the episodes are still running. Dashed: true caution.", ha="center", fontsize=6.5)
    fig.tight_layout(); save(fig, "fig_speed_time")


def fig_capture():
    """Depth / detection / BEV stills captured by bev_rl/adaptive/capture_pipeline_stills.py (same scene, same raw frames, only the weather differs)."""
    cap = OUT / "capture"
    if not (cap / "stats.json").exists(): print("capture not found, skipping capture figures"); return
    st = json.loads((cap / "stats.json").read_text()); cols = [("clear_0.0", "Clear"), ("fog_0.6", "Fog 0.6"), ("fog_1.0", "Fog 1.0"), ("rain_0.6", "Rain 0.6"), ("rain_1.0", "Rain 1.0")]
    # 1) five camera views (raw, clear)
    names = [("front_left_camera", "Front-left"), ("front_right_camera", "Front-right"), ("left_camera", "Left"), ("right_camera", "Right"), ("rear_camera", "Rear")]
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.45))
    for ax, (n, t) in zip(axs, names):
        f = cap / f"cam_{n}.png"
        if f.exists(): ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=7.5); ax.axis("off")
    fig.tight_layout(pad=0.3); save(fig, "fig_camera_views")
    # 2) weathered front-left frame with detections, and the SGM depth map
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 3.0))
    cmap = plt.get_cmap("turbo_r").copy(); cmap.set_bad("black")
    for j, (tag, t) in enumerate(cols):
        im = cv2.cvtColor(cv2.imread(str(cap / f"{tag}_front_left.png")), cv2.COLOR_BGR2RGB); axs[0, j].imshow(im)
        dets = json.loads((cap / f"{tag}_dets.json").read_text()); dl = dets.get("front_left_camera", next(iter(dets.values()), []))
        for d in dl: axs[0, j].add_patch(plt.Rectangle((d["xmin"], d["ymin"]), d["xmax"] - d["xmin"], d["ymax"] - d["ymin"], fill=False, ec="#d79b00", lw=1.4))
        axs[0, j].set_title(t, fontsize=8); axs[0, j].axis("off")
        d = np.load(cap / f"{tag}_depth.npy"); axs[1, j].imshow(np.where(d > 0, d, np.nan), cmap=cmap, vmin=0, vmax=40); axs[1, j].axis("off")
        axs[1, j].text(0.5, -0.03, f"valid depth {100 * st['conditions'][tag]['valid_depth_frac_lower_half']:.0f}%", transform=axs[1, j].transAxes, ha="center", va="top", fontsize=6.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 40)); cb = fig.colorbar(sm, ax=axs[1, :], fraction=0.02, pad=0.01); cb.set_label("depth (m)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.93, bottom=0.08, wspace=0.03, hspace=0.18); save(fig, "fig_depth_weather")
    # 3) BEV grids: simulator truth and the camera-derived grid per weather
    from matplotlib.colors import ListedColormap, BoundaryNorm
    bc = ListedColormap(["#ffffff", "#9ecae1", "#8c8c8c", "#d62728"]); bn = BoundaryNorm([-0.01, 0.25, 0.65, 0.9, 1.01], bc.N)
    items = [("truth_bev.npy", "Simulator truth (reference)")] + [(f"{tag}_bev.npy", t) for tag, t in cols]
    fig, axs = plt.subplots(1, 6, figsize=(7.2, 1.7))
    for ax, (f, t) in zip(axs, items):
        g = np.load(cap / f); g = g[0] if g.ndim == 3 else g; ax.imshow(g, cmap=bc, norm=bn, interpolation="nearest"); ax.plot([32], [56], marker="^", color="#2ca02c", ms=4); ax.set_title(t, fontsize=7); ax.axis("off")
    fig.tight_layout(pad=0.3); save(fig, "fig_bev_weather")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", nargs="+", default=[str(ROOT / "bev_rl/adaptive/results/severity_eval_20seed_snapshot")]); a = ap.parse_args()
    R = load(a.episodes); fig_calibration(); fig_weather_frames(); fig_speed_energy(R); fig_speed_time(R); fig_capture()
