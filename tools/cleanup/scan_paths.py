"""Read-only scan for machine-specific paths that make scripts non-portable.

Reports, per file: absolute `/d/bev_Fresh/bevcapstone` (Git-Bash style) or `D:\\bev_Fresh` paths, and hardcoded `./venv/Scripts/python.exe`. It only REPORTS. A later, separately
approved step (docs/CLEANUP_LOG.md, B1) will rewrite them, and only after all experiments have finished (several queue scripts are running or queued now).

    python tools/cleanup/scan_paths.py
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABS = re.compile(r"/d/bev_Fresh|D:\\\\?bev_Fresh|C:\\\\?Users|/c/Users")
VENV = re.compile(r"venv[/\\]Scripts[/\\]python\.exe")


def main():
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    a_hits, v_hits = [], []
    for f in files:
        if not f.endswith((".sh", ".ps1", ".py", ".md", ".tex")) or f.startswith(("report/latest", "docs/CLEANUP", "docs/GIT_WORKFLOW", "tools/")):
            continue
        t = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        n, v = len(ABS.findall(t)), len(VENV.findall(t))
        if n:
            a_hits.append((f, n))
        if v:
            v_hits.append((f, v))
    print("files with absolute machine paths: %d (%d occurrences)" % (len(a_hits), sum(n for _, n in a_hits)))
    for f, n in a_hits:
        print("  %2d  %s" % (n, f))
    print("files hardcoding venv/Scripts/python.exe: %d" % len(v_hits))
    for f, n in v_hits[:60]:
        print("  %2d  %s" % (n, f))


if __name__ == "__main__":
    main()
