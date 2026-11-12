"""Generate Appendix A (inventory of every file created or changed) and the model manifest table of the report.

Code files are described by the first sentence of their module docstring (or the first comment line of a shell script), so the inventory
cannot drift from the code. Data and result folders are described by a hand-written table in this script.

Run from the repository root:
    venv\\Scripts\\python.exe report\\make_file_inventory.py
Output: report/sections/appendix_a_file_inventory.tex
"""
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "technical" / "appendix_a_file_inventory.tex"
GITHUB = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/main"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.splitlines())


def esc(text: str) -> str:
    text = text.replace("\\", " ")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def first_sentence(text: str, limit: int = 230) -> str:
    text = " ".join(text.split())
    m = re.search(r"(?<=[a-z0-9\)])\.(\s|$)", text)
    if m and m.start() < limit:
        text = text[: m.start() + 1]
    return text[:limit] + ("..." if len(text) > limit else "")


def describe(path: Path) -> str:
    try:
        if path.suffix == ".py":
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
            return first_sentence(doc) if doc else "(no description in the file)"
        if path.suffix == ".sh":
            for line in path.read_text(encoding="utf-8").splitlines()[1:8]:
                if line.startswith("#"):
                    return first_sentence(line.lstrip("# ").strip())
            return "(shell script, no description)"
        if path.suffix == ".md":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    return first_sentence(line.lstrip("# ").strip())
        if path.suffix == ".tex":
            return "LaTeX source of a report section"
    except Exception as exc:  # pragma: no cover
        return f"(unreadable: {exc})"
    return ""


def code_table(title: str, files, note: str = "") -> str:
    rows = [f"\\subsection{{{title}}}", note, "{\\small\\begin{longtable}{@{}p{5.6cm}p{8.4cm}p{1.6cm}@{}}\\toprule File & What it does & On GitHub\\\\\\midrule\\endhead"]
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        rows.append(f"\\pth{{{rel}}} & {esc(describe(f))} & {'yes' if rel in GITHUB else 'local only'}\\\\")
    rows.append("\\bottomrule\\end{longtable}}")
    return "\n".join(rows)


DATA_ROWS = [
    ("bev_rl/adaptive/runs/<run>/", "Training workspace of every model run (one folder per run): best.pt (selected), iter_k.pt or step_k.pt (every round / checkpoint), log.json (training curve), best.json (selection record), config.json (settings). 27 run folders; selected files are copied to trained_models/."),
    ("bev_rl/adaptive/results/final_200seed/", "Per-episode results of the 200-episode clear-weather evaluation (six models, seeds 5000--5199)."),
    ("bev_rl/adaptive/results/n200_all_weather/", "Per-episode results of the 200-episode evaluation in four weather cells with the detector-dropout proxy (two baselines, headline model; the no-traffic baseline has the clear cell and a partial fog cell)."),
    ("bev_rl/adaptive/results/degraded_eval/", "Per-episode results under the calibrated degradation: cal_d0.05/ (four models x four weather cells, 200 episodes) and hurt_n100/ (headline model, 100 episodes)."),
    ("bev_rl/adaptive/results/caution/", "Caution-study results, 40 episodes per cell."),
    ("bev_rl/adaptive/results/depth_error_measurements.json, depth_error_model.json", "Corrected measurement of detector and stereo-depth degradation and the sampling model fitted to it. The *.SUSPECT_* files are the invalid first run, kept for the record."),
    ("bev_rl/adaptive/results/retention_table.json, perception_*.json", "Earlier measurement of detector retention (basis of the dropout proxy) and perception benchmarks."),
    ("bev_rl/adaptive/results/localisation_error_*.json, pose_two_cue_*.json", "Pose-error sensitivity results, 40 episodes per point."),
    ("bev_rl/adaptive/results/final_*_best.json", "10-episode evaluations of the five input designs (densities 0.05 and 0.2)."),
    ("bev_rl/adaptive/results/expert_ceiling_*, safe_teacher_ceiling_*", "Teacher ceiling results in sparse and dense traffic."),
    ("bev_rl/adaptive/results/C2/, probe/, idm/, scalar_ablation/", "Episode tables of the privileged-teacher reinforcement-learning attempts, of the stock driver model and of the early scalar-state ablation."),
    ("bev_rl/adaptive/data/idm_clear.npz", "Demonstrations of the stock driver model used for the first behaviour-cloning initialisation (ignored by git)."),
    ("bev_rl/adaptive/logs/, logs/", "Text logs and completion flags of training, evaluation and recording runs, including the night queue (night_status.txt)."),
    ("bev_rl/adaptive/checkpoints/", "Initialisation checkpoints of the early privileged-teacher reinforcement-learning attempts."),
    ("bev_rl/results/", "Results of the earliest policy-side work: behaviour-cloning baseline on seeds 1000--1009, reward audits, perception benchmarks."),
    ("trained_models/", "Final models in one folder (Table in this appendix), with MODEL_MANIFEST.json and training_records/."),
    ("report/figures, report/tables, report/data, report/videos", "Everything the report is built from; figures and tables are generated by the make_*.py scripts."),
    ("docs/project_history/", "Dated notes from 1--5 November: architecture state, decisions, experiment log, known limitations, preliminary literature review, research plan, environment log, perception audit."),
    ("archive/paper_earlier_drafts/", "Earlier paper material: IEEE-format draft, outline, 42 earlier figures, tables, literature PDFs, hand-off documents for the weather module, pre-registration, old videos' index files."),
    ("archive/removed_scripts_and_early_runs_2026-11-05/", "Scripts and runs that were superseded on 5 November (camera-view-version-2 collection and perception scripts, PPO training script, early runs C3, pilot_cond, smoke tests)."),
]


def main() -> None:
    parts = ["\\section{Inventory of every file created or changed}\\label{app:inventory}",
             "This appendix is generated by \\pth{report/make_file_inventory.py}. Descriptions are the first sentence of each file's own documentation, quoted as written in the code (so abbreviations such as BC, BEV, PPO or IDM appear there; Table~\\ref{tab:terms} explains them). The three modified teammate files, the policy-side code, the tests, the report and its tools, and the data folders are listed separately."]
    parts.append("\\subsection{Teammate files modified locally (uncommitted)}\\begin{itemize}\\item \\pth{bev_policy/model.py}: optional extra-input branch (\\code{weather_dim}). \\item \\pth{bev_policy/dataset.py}: optional weather array, unsigned 8-bit grids. \\item \\pth{bev_policy/train_bc.py}: options \\code{--weather-input}, \\code{--only-clear}, \\code{--val-count}.\\end{itemize}")
    parts.append(code_table("Policy-side code, first layer (\\texttt{bev\\_rl/})", sorted((ROOT / "bev_rl").glob("*.py"))))
    adaptive = sorted((ROOT / "bev_rl" / "adaptive").glob("*.py")) + sorted((ROOT / "bev_rl" / "adaptive").glob("*.sh"))
    parts.append(code_table("Policy-side code, experiments (\\texttt{bev\\_rl/adaptive/})", adaptive,
                            "Code names and what they correspond to are in Table~\\ref{tab:terms}; the order of the files is alphabetical."))
    parts.append(code_table("Policy-side tests", sorted((ROOT / "tests").glob("test_bev_rl_*.py"))))
    tools = sorted((ROOT / "report").glob("*.py")) + sorted((ROOT / "repository_tools").glob("*.py"))
    parts.append(code_table("Report and repository tools", tools))
    report_sources = sorted((ROOT / "report").glob("*.tex")) + sorted((ROOT / "report" / "shared").glob("*.tex")) + sorted((ROOT / "report" / "technical").glob("*.tex")) + sorted((ROOT / "report" / "guide").glob("*.tex"))
    parts.append(code_table("Report sources (LaTeX)", report_sources, "(This appendix itself is generated by the script above and is therefore listed as a section. The sixteen figure blocks in \\pth{report/figure_blocks/} are generated by \\pth{report/make_figure_blocks.py}.)"))
    parts.append("\\subsection{Data, results, logs and archives}")
    parts.append("{\\small\\begin{longtable}{@{}p{6.0cm}p{9.6cm}@{}}\\toprule Location & Content\\\\\\midrule\\endhead")
    for loc, desc in DATA_ROWS:
        parts.append(f"\\pth{{{loc}}} & {esc(desc)}\\\\")
    parts.append("\\bottomrule\\end{longtable}}")
    # manifest
    manifest = json.loads((ROOT / "trained_models" / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    parts.append("\\subsection{Trained models: new file name, previous file name, GitHub status and hash}")
    mapping = {r["new_file"]: r["previous_file"] for r in json.loads((ROOT / "trained_models" / "MODEL_NAME_MAPPING.json").read_text(encoding="utf-8"))["mapping"]}
    parts.append("{\\scriptsize\\begin{longtable}{@{}p{5.6cm}p{4.9cm}p{1.5cm}p{1.2cm}p{2.0cm}@{}}\\toprule New file name in trained\\_models/ & Previous file name & On GitHub & Size (kB) & SHA-256 (first 12 digits)\\\\\\midrule\\endhead")
    for m in manifest["models"]:
        gh = "yes" if m["original_path"] in GITHUB else "local only"
        parts.append(f"\\pth{{{m['file'].split('/')[-1]}}} & \\pth{{{mapping[m['file']].split('/')[-1]}}} & {gh} & {m['size_kilobytes']:.0f} & \\texttt{{{m['sha256'][:12]}}}\\\\")
    parts.append("\\bottomrule\\end{longtable}}")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote", OUT, "with", len(adaptive), "adaptive files")


if __name__ == "__main__":
    main()
