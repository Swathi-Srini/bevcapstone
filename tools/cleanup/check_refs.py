"""Read-only reference checker: finds file paths written in backticks in the documentation (README.md, docs/*.md, report/*.tex) and `python -m` modules in shell scripts
that do not exist in the repository. It never writes or changes anything.

    python tools/cleanup/check_refs.py            # run from the repository root; exit code 0 always (report only)
    python tools/cleanup/check_refs.py --strict   # exit code 1 if anything is unresolved
    python tools/cleanup/check_refs.py --root <folder>   # check another folder, e.g. an extracted copy of main

Known benign patterns (not reported): paths that are *previous names by design* (trained_models/MODEL_NAME_MAPPING.md), dated history notes (docs/project_history/), runtime outputs, git-ignored local files.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if "--root" in sys.argv:                      # optional: check another folder (for example an extracted copy of main)
    ROOT = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
SKIP_FILES = {"trained_models/MODEL_NAME_MAPPING.md"}
SKIP_PREFIX = ("docs/project_history/", ".claude/", "report/local_only/")
BASES = ("", "report", "docs", "bev_rl/adaptive", "bev_rl/adaptive/results", "trained_models", "YOLO+weather module")
IGNORED_LOCAL = re.compile(r"(^logs/|\.pth$|integrated_output/|captured_frames|^checkpoints/|/runs/|^runs/|results/camera_dagger_final/.*round)")


def tracked():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout
    if out.strip():
        return out.splitlines()
    # not a git checkout (for example an extracted copy): walk the folder instead
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "venv" not in p.parts]


def main():
    files = tracked()
    docs = [f for f in files if (f.endswith(".md") or (f.startswith("report/") and f.endswith(".tex"))) and f not in SKIP_FILES and not f.startswith(SKIP_PREFIX)]
    missing = []
    for f in docs:
        text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        for tok in set(re.findall(r"`([A-Za-z0-9_./+ \-\(\)]+\.[A-Za-z0-9]{1,5})`", text)):
            t = tok.strip()
            if t.split()[0] in ("powershell", "bash", "python", "pwsh"):      # a command line: check the file it names
                t = t.split()[-1]
            if "/" not in t or t.startswith(("http", "~", ".")) or "*" in t or "<" in t or IGNORED_LOCAL.search(t):
                continue
            if not any((ROOT / b / t).exists() for b in BASES):
                missing.append((f, t))
    for f in files:
        if f.endswith(".sh") and not f.startswith(SKIP_PREFIX):
            text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
            for m in set(re.findall(r"-m\s+([A-Za-z0-9_.]+)", text)):
                if m.startswith(("bev_", "yolo", "camera")) and not (ROOT / (m.replace(".", "/") + ".py")).exists():
                    missing.append((f, "python -m " + m))
    for f, t in sorted(set(missing)):
        print("MISSING  %-55s -> %s" % (f, t))
    print("\n%d documentation/script files checked, %d unresolved references" % (len(docs), len(set(missing))))
    if "--strict" in sys.argv and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
