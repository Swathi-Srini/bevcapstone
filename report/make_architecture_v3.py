"""Generates report/Architecture_Severity_System_v3_vertical.drawio (5 pages) in the colour language of Architecture_Live_Caution_Branch_v2.drawio.
Colours: green=sensors/env, light green=perception/network, purple=weather, yellow=BEV, blue=HD map/action, red=scalar, ORANGE=new in this work, dashed=training only.
Page 1 = overall (Fig. 1, few details).  Pages 2-5 = module expansions (estimator, perception->BEV, policy network, training loop)."""
from html import escape
from pathlib import Path
C = {"green": ("#d5e8d4", "#82b366"), "lgreen": ("#d5f5e3", "#82b366"), "blue": ("#dae8fc", "#6c8ebf"), "purple": ("#e1d5e7", "#9673a6"),
     "yellow": ("#fff2cc", "#d6b656"), "red": ("#f8cecc", "#b85450"), "orange": ("#ffe6cc", "#d79b00"), "grey": ("#f5f5f5", "#666666")}


class Page:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 2; self.R = {}; self.E = []

    def _t(self, s): return escape(s, quote=True).replace("\n", "&lt;br&gt;") if False else s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def box(self, id, text, x, y, w, h, col, new=False, dashed=False, fs=12):
        self.R[id] = dict(t=text, x=x, y=y, w=w, h=h, col=col, new=new, dashed=dashed, fs=fs, kind="box")
        f, s = C[col]; st = f"rounded=1;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};fontSize={fs};" + ("strokeWidth=2;" if new else "") + ("dashed=1;" if dashed else "")
        self.cells.append(f'<mxCell id="{id}" value="{self._t(text)}" style="{st}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def cont(self, id, text, x, y, w, h, fs=12):
        self.R[id] = dict(t="<b>" + text + "</b>", x=x, y=y, w=w, h=h, col=None, new=False, dashed=False, fs=fs, kind="cont")
        self.cells.append(f'<mxCell id="{id}" value="{self._t("<b>"+text+"</b>")}" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#333333;verticalAlign=top;fontSize={fs};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def text(self, id, text, x, y, w, h, fs=11):
        self.R[id] = dict(t=text, x=x, y=y, w=w, h=h, col=None, new=False, dashed=False, fs=fs, kind="text")
        self.cells.append(f'<mxCell id="{id}" value="{self._t(text)}" style="text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize={fs};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def edge(self, src, tgt, label="", orange=False, dashed=False, pts=None, extra=""):
        st = "edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;fontSize=10;" + ("strokeColor=#d79b00;strokeWidth=2;" if orange else "") + ("dashed=1;" if dashed else "") + extra
        g = '<mxGeometry relative="1" as="geometry">' + ("<Array as=\"points\">" + "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts) + "</Array>" if pts else "") + "</mxGeometry>"
        self.E.append(dict(s=src, t=tgt, label=label, orange=orange, dashed=dashed, pts=pts, extra=extra))
        self.n += 1
        self.cells.append(f'<mxCell id="e{self.n}" value="{self._t(label)}" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}">{g}</mxCell>')

    def xml(self):
        return (f'<diagram name="{escape(self.name)}" id="{abs(hash(self.name)) % 10**8}"><mxGraphModel dx="900" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) + "</root></mxGraphModel></diagram>")


def p1():
    p = Page("1 Overall system (Fig. 1)", 660, 1350)
    p.text("t", "<b>Overall system.</b> Orange = new in this work. Dashed orange = training only; dashed grey = environment feedback loop. Model A = the same network without the orange caution path.", 20, 6, 620, 34, 11)
    p.cont("c_sens", "Sensor Layer", 20, 48, 600, 104)
    for i, (n, nm) in enumerate([("c1", "Front-right"), ("c2", "Front-left"), ("c3", "Left"), ("c4", "Right"), ("c5", "Rear")]):
        p.box(n, f"{nm} RGB camera<br>1200×900, 60° FOV", 30 + 118 * i, 84, 110, 56, "green", fs=10)
    p.box("wx", "<b>Controlled weather intervention</b><br>synthetic fog / rain image corruption on all camera inputs<br>visual degradation only: no friction / dynamics change", 120, 190, 420, 66, "purple")
    p.cont("c_perc", "Perception Modules", 20, 290, 390, 262)
    p.box("sgm", "Stereo SGM  Z = f·B / d_px<br>depth map H×W", 40, 326, 350, 50, "lgreen"); p.box("yf", "YOLOv8 front: detections (bboxes)", 40, 388, 350, 40, "lgreen")
    p.box("ylr", "YOLOv8 left and right: detections (bboxes)", 40, 440, 350, 40, "lgreen"); p.box("yr", "YOLOv8 rear: detections (bboxes)", 40, 492, 350, 40, "lgreen")
    p.box("est", "<b>Severity estimator (NEW)</b><br>image-based<br>front-left weather-degraded frame<br>→ 7 image features<br>→ kind classifier + fog / rain level regressors<br>→ EMA filter", 440, 290, 180, 150, "orange", new=True, fs=11)
    p.box("caut", "<b>Caution ĉ ∈ [0, 1]</b><br>clear 0, fog = level, rain = level<br>(no weather label given to the policy)", 440, 480, 180, 72, "orange", new=True, fs=11)
    p.box("hd", "<b>HD map</b><br>route waypoints<br>lane boundaries", 20, 592, 150, 62, "blue"); p.box("bev", "<b>BEV grid builder</b><br>64×64 float32<br>0 free · 0.5 centreline · 0.8 boundary · 1.0 obstacle<br>ego at row 56, column 32", 190, 584, 220, 110, "yellow", fs=11)
    p.box("sc", "<b>Scalar state (6D)</b><br>[v, s, d, θ_err, κ_ahead, d_goal]<br>HD map + GPS + IMU", 20, 682, 150, 78, "red", fs=10)
    p.cont("c_pol", "Policy network (one network)", 20, 780, 600, 232)
    p.box("mlp", "<b>MLP branch (scalar)</b><br>FC 6→64→32<br>32D", 40, 820, 180, 70, "lgreen"); p.box("cnn", "<b>CNN branch (BEV)</b><br>Conv 5×5 → 3×3 → 3×3<br>→ pooling → FC → 128D", 240, 820, 160, 70, "lgreen", fs=11)
    p.box("cb", "<b>Caution branch (NEW)</b><br>FC 1→16 + ReLU<br>16D", 420, 820, 180, 70, "orange", new=True)
    p.box("fus", "<b>Fusion concatenation</b> [128, 32, 16] = 176D<br>(Model A: [128, 32] = 160D)", 40, 920, 330, 62, "lgreen"); p.box("head", "<b>Action head</b><br>FC 176→128→2, ReLU + tanh", 390, 920, 210, 62, "lgreen")
    p.box("act", "<b>Action output (2D)</b>: [steering, longitudinal] ∈ [−1, 1]<br>positive: throttle, negative: brake", 130, 1050, 400, 56, "blue")
    p.box("env", "<b>MetaDrive environment</b>: next state s(t+1)<br>energy and outcome logging", 130, 1140, 400, 54, "green")
    p.box("tch", "<b>Training only (DAgger)</b>: privileged teacher, true caution c<br>target speed 30(1 − c/3) km/h, following gap 10 + 15c m<br>relabels the states the student visits", 20, 1236, 600, 78, "orange", new=True, dashed=True, fs=11)
    for a, b in [("c_sens", "wx"), ("wx", "c_perc"), ("c_perc", "bev"), ("hd", "bev"), ("bev", "cnn"), ("sc", "mlp"), ("mlp", "fus"), ("cnn", "fus"), ("fus", "head"), ("head", "act"), ("act", "env")]: p.edge(a, b)
    p.edge("wx", "est", orange=True); p.edge("est", "caut", orange=True); p.edge("caut", "cb", "ĉ (student input)", orange=True); p.edge("cb", "fus", orange=True)
    p.edge("tch", "c_pol", "action labels", orange=True, dashed=True, pts=[(640, 1275), (640, 896)], extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
    p.edge("env", "c_sens", dashed=True, pts=[(10, 1167), (10, 100)], extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    return p


def p2():
    p = Page("2 Severity estimator (expansion)", 580, 790)
    p.text("t", "<b>Image-based severity estimator</b> (new). Inputs only the weather-processed front-left frame.", 20, 8, 540, 30)
    p.box("f", "Front-left frame<br>(weather-processed, 1200×900)", 140, 50, 300, 50, "purple")
    p.box("ft", "<b>7 image features</b><br>log Laplacian variance · mean gradient · RMS contrast · dark channel · brightness · saturation · strong-edge share", 60, 130, 460, 70, "orange", new=True)
    p.box("k", "<b>Kind classifier</b><br>softmax: clear / fog / rain<br>balanced weights, L2", 20, 240, 170, 80, "orange", new=True); p.box("rf", "<b>Fog regressor</b><br>ridge, level ∈ [0, 1]", 205, 240, 170, 80, "orange", new=True); p.box("rr", "<b>Rain regressor</b><br>ridge, level ∈ [0, 1]", 390, 240, 170, 80, "orange", new=True)
    p.box("ema1", "<b>Temporal filter 1</b>: EMA of kind probabilities (α = 0.15) → hard kind = argmax", 40, 360, 500, 44, "orange", new=True)
    p.box("sel", "<b>Level of the selected kind</b> (clear → 0)", 40, 436, 500, 40, "orange", new=True)
    p.box("ema2", "<b>Temporal filter 2</b>: EMA of level (α = 0.15)", 40, 508, 500, 40, "orange", new=True)
    p.box("cau", "<b>Caution ĉ</b> = filtered level (identity map: clear 0, fog = level, rain = level)", 40, 580, 500, 44, "orange", new=True)
    p.box("pol", "→ caution branch of the policy (FC 1→16)", 140, 662, 300, 40, "lgreen")
    p.text("n", "Fitted on 25,408 frames; evaluated on held-out routes (seeds 9040–9042). Probe levels fog 0.45 / 0.6 and rain 0.3 / 0.45 were never fitted.", 20, 730, 540, 50, 10)
    for a, b in [("f", "ft"), ("ft", "k"), ("ft", "rf"), ("ft", "rr"), ("k", "ema1"), ("ema1", "sel"), ("rf", "sel"), ("rr", "sel"), ("sel", "ema2"), ("ema2", "cau"), ("cau", "pol")]: p.edge(a, b, orange=True)
    return p


def p3():
    p = Page("3 Perception to BEV (expansion)", 580, 680)
    p.text("t", "<b>Driving perception to BEV grid</b> (unchanged from the non-adaptive system).", 20, 8, 540, 30)
    p.cont("cs", "Weather-processed camera images", 20, 44, 540, 120)
    p.box("fl", "Front-left", 40, 80, 100, 60, "green"); p.box("fr", "Front-right", 155, 80, 100, 60, "green"); p.box("lr", "Left / right", 270, 80, 100, 60, "green"); p.box("rear", "Rear", 385, 80, 100, 60, "green")
    p.box("sgm", "<b>Stereo SGM</b> (front pair)<br>Z = f·B / d_px → depth map", 40, 200, 230, 60, "lgreen"); p.box("yolo", "<b>YOLOv8n</b> on four views<br>→ bounding boxes", 300, 200, 240, 60, "lgreen")
    p.box("fp", "<b>Object footprints</b> in ego frame (depth + boxes)", 90, 300, 400, 44, "lgreen")
    p.box("hd", "<b>HD map</b><br>route waypoints, lane boundaries", 40, 390, 200, 56, "blue"); p.box("bev", "<b>BEV grid builder</b> (frozen)<br>64×64: 0 free · 0.5 centreline · 0.8 boundary · 1.0 obstacle", 260, 380, 290, 76, "yellow")
    p.box("sc", "<b>Scalar state</b> [v, progress, d_goal]<br>no simulator position given to the student", 40, 490, 220, 76, "red", fs=11)
    p.box("out", "→ policy network (CNN branch, MLP branch)", 140, 600, 300, 40, "lgreen")
    for a, b in [("fl", "sgm"), ("fr", "sgm"), ("cs", "yolo"), ("sgm", "fp"), ("yolo", "fp"), ("fp", "bev"), ("hd", "bev"), ("bev", "out"), ("sc", "out")]: p.edge(a, b)
    return p


def p4():
    p = Page("4 Policy network (expansion)", 580, 550)
    p.text("t", "<b>Policy network.</b> Model A omits the orange branch (fusion 160D).", 20, 8, 540, 30)
    p.box("bev", "BEV 64×64", 20, 50, 160, 40, "yellow"); p.box("sc", "Scalars (v, progress, d_goal)", 200, 50, 180, 40, "red"); p.box("ch", "Caution ĉ", 400, 50, 160, 40, "orange", new=True)
    p.box("cnn", "<b>CNN branch</b><br>Conv 5×5 → Conv 3×3 → Conv 3×3<br>→ pooling → FC → 128D", 20, 130, 170, 100, "lgreen"); p.box("mlp", "<b>MLP branch</b><br>FC 6→64→32<br>(32D)", 205, 130, 170, 100, "lgreen"); p.box("cb", "<b>Caution branch (NEW)</b><br>FC 1→16 + ReLU<br>(16D)", 390, 130, 170, 100, "orange", new=True)
    p.box("fus", "<b>Fusion concatenation</b> [128, 32, 16] = 176D", 60, 280, 460, 50, "lgreen"); p.box("head", "<b>Action head</b> FC 176→128→2 (ReLU, tanh)", 60, 370, 460, 50, "lgreen")
    p.box("act", "<b>Action</b> [steering, longitudinal command] ∈ [−1, 1]<br>positive: throttle, negative: brake", 60, 460, 460, 56, "blue")
    for a, b in [("bev", "cnn"), ("sc", "mlp"), ("ch", "cb")]: p.edge(a, b, orange=(a == "ch"))
    for a, b in [("cnn", "fus"), ("mlp", "fus"), ("fus", "head"), ("head", "act")]: p.edge(a, b)
    p.edge("cb", "fus", orange=True)
    return p


def p5():
    p = Page("5 Training by DAgger (expansion)", 580, 620)
    p.text("t", "<b>Training only: camera-pipeline DAgger.</b> Not part of the deployed system.", 20, 8, 540, 30)
    p.box("run", "<b>Student drives</b> through the full camera pipeline<br>(β = 1, then 0.5^t mixing with teacher; action noise 0.1)", 60, 50, 460, 60, "green", dashed=True)
    p.box("st", "<b>States visited</b> (weather mix clear / fog / rain; true c known in training)", 60, 150, 460, 44, "lgreen", dashed=True)
    p.box("tch", "<b>Privileged teacher</b> (reads exact simulator state)<br>IDM + SafeTeacher + curve guard 4.0<br>target speed 30(1 − c/3) km/h · following gap 10 + 15c m", 60, 234, 460, 80, "orange", new=True, dashed=True)
    p.box("lab", "<b>Action labels</b> for the visited states → aggregate all rounds", 60, 354, 460, 44, "lgreen", dashed=True)
    p.box("fit", "<b>Retrain student</b> (Adam 1e-3, batch 128): 4 rounds × 12 episodes", 60, 438, 460, 44, "lgreen", dashed=True)
    p.box("B", "<b>Model B</b> = trained network with true c (reference)<br><b>Model C</b> = the same network run with the image-based ĉ", 60, 522, 460, 60, "orange", new=True)
    for a, b in [("run", "st"), ("st", "tch"), ("tch", "lab"), ("lab", "fit"), ("fit", "B")]: p.edge(a, b, orange=(a == "tch"), dashed=True)
    p.edge("fit", "run", "next round", dashed=True, pts=[(545, 460), (545, 80)], extra="exitX=1;exitY=0.5;entryX=1;entryY=0.5;")
    return p


import re, textwrap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def _lines(t, w, fs):
    out = []
    for ln in t.split("<br>"):
        bold = "<b>" in ln
        plain = re.sub(r"</?b>|</?i>", "", ln)
        wrap = max(8, int((w - 10) / (fs * 0.56)))
        for k in (textwrap.wrap(plain, wrap) or [""]): out.append((k, bold))
    return out


def render(p, outdir):
    sc = 1 / 96; fig = plt.figure(figsize=(p.w * sc, p.h * sc), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, p.w); ax.set_ylim(p.h, 0); ax.axis("off")
    for id, r in p.R.items():
        if r["kind"] == "cont":
            ax.add_patch(Rectangle((r["x"], r["y"]), r["w"], r["h"], fill=False, ec="#333333", lw=1.0)); ax.text(r["x"] + 8, r["y"] + 6, re.sub(r"</?b>", "", r["t"]), ha="left", va="top", fontsize=r["fs"] * 0.75, fontweight="bold")
        elif r["kind"] == "text":
            ax.text(r["x"], r["y"], "\n".join(textwrap.wrap(re.sub(r"</?[bi]>", "", r["t"]), int(r["w"] / (r["fs"] * 0.56)))), ha="left", va="top", fontsize=r["fs"] * 0.75)
    for id, r in p.R.items():
        if r["kind"] != "box": continue
        f, s_ = C[r["col"]]; ax.add_patch(FancyBboxPatch((r["x"], r["y"]), r["w"], r["h"], boxstyle="round,pad=0,rounding_size=7", fc=f, ec=s_, lw=2.0 if r["new"] else 1.0, ls="--" if r["dashed"] else "-", zorder=2))
        L = _lines(r["t"], r["w"], r["fs"]); lh = r["fs"] * 1.3; y0 = r["y"] + r["h"] / 2 - lh * (len(L) - 1) / 2
        for i, (k, b) in enumerate(L): ax.text(r["x"] + r["w"] / 2, y0 + i * lh, k, ha="center", va="center", fontsize=r["fs"] * 0.75, fontweight="bold" if (b and i == 0) else "normal", zorder=3)

    def anc(r, fx, fy): return (r["x"] + fx * r["w"], r["y"] + fy * r["h"])
    for e in p.E:
        a, b = p.R[e["s"]], p.R[e["t"]]; ex = dict(kv.split("=") for kv in e["extra"].split(";") if "=" in kv); col = "#d79b00" if e["orange"] else "#333333"
        if e["pts"]:
            P = [anc(a, float(ex.get("exitX", .5)), float(ex.get("exitY", 1)))] + list(e["pts"]) + [anc(b, float(ex.get("entryX", .5)), float(ex.get("entryY", 0)))]
        elif b["y"] >= a["y"] + a["h"] - 1:
            x1, y1 = anc(a, .5, 1); x2, y2 = anc(b, .5, 0); ov = (max(a["x"], b["x"]), min(a["x"] + a["w"], b["x"] + b["w"]))
            if ov[0] < ov[1]: x1 = x2 = (ov[0] + ov[1]) / 2
            ym = (y1 + y2) / 2; P = [(x1, y1), (x1, ym), (x2, ym), (x2, y2)]
        else:
            x1, y1 = anc(a, 1, .5); x2, y2 = anc(b, 0, .5); P = [(x1, y1), (x2, y2)]
        xs, ys = zip(*P); ax.plot(xs, ys, color=col, lw=2.0 if e["orange"] else 1.0, ls="--" if e["dashed"] else "-", zorder=1, solid_capstyle="butt")
        (xa, ya), (xb, yb) = P[-2], P[-1]; ax.annotate("", xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=9), zorder=1)
        if e["label"]:
            sg = sorted(zip(P[:-1], P[1:]), key=lambda s: -abs(s[0][0] - s[1][0]) - abs(s[0][1] - s[1][1]))[0]; mx, my = (sg[0][0] + sg[1][0]) / 2, (sg[0][1] + sg[1][1]) / 2
            ax.text(mx + 4, my, e["label"], fontsize=7.5, color=col if e["orange"] else "#333333", ha="left", va="center", rotation=90 if sg[0][0] == sg[1][0] else 0, bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
    return fig


FIGNAMES = ["fig_architecture", "fig_arch_estimator", "fig_arch_perception", "fig_arch_policy", "fig_arch_training"]

out = Path(__file__).parent / "Architecture_Severity_System_v3_vertical.drawio"
PAGES = [f() for f in (p1, p2, p3, p4, p5)]
out.write_text('<mxfile host="app.diagrams.net">' + "".join(p.xml() for p in PAGES) + "</mxfile>", encoding="utf-8"); print("wrote", out)
fd = Path(__file__).parent / "paper_severity" / "figures"; fd.mkdir(exist_ok=True)
for p, n in zip(PAGES, FIGNAMES):
    fg = render(p, fd); fg.savefig(fd / f"{n}.png", dpi=200); fg.savefig(fd / f"{n}.pdf"); plt.close(fg); print("wrote", n)
