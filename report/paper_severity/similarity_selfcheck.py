"""Local overlap self-check (NOT a Turnitin/iThenticate report).  Share of the paper's prose 8-word shingles that also occur in our own earlier documents and in the reference paper.
    python similarity_selfcheck.py main.tex
A real plagiarism tool also compares against the web, journals and previously submitted student work (incl. this team's earlier report); this script only measures those local sources."""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N = 8


def strip_tex(t):
    t = re.sub(r"(?m)%.*$", " ", t); t = re.sub(r"\\begin\{(equation|align|multline|tabular|table|figure|tikzpicture|thebibliography)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t); t = re.sub(r"\\(cite|ref|label|eqref|includegraphics)\{[^}]*\}", " ", t); t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t); t = re.sub(r"[{}\\]", " ", t)
    return t


def words(t): return re.findall(r"[a-z0-9]+", t.lower())


def shingles(w): return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def main(f):
    mine = words(strip_tex(Path(f).read_text(encoding="utf8"))); S = shingles(mine)
    srcs = [p for p in list((ROOT / "report").glob("*.tex")) + list((ROOT / "archive").rglob("*.tex")) + [ROOT / "README.md"] + list(ROOT.glob("bev_rl/adaptive/*.md")) if p.exists() and p.resolve() != Path(f).resolve()]
    import os
    ref = Path(os.environ.get("SIM_EXTRA_REF", "ref_paper.txt"))   # optional: text of any extra paper to compare against
    if ref.exists(): srcs.append(ref)
    total = set(); rows = []
    for p in srcs:
        try: txt = p.read_text(encoding="utf8", errors="ignore")
        except Exception: continue
        sh = shingles(words(strip_tex(txt) if p.suffix == ".tex" else txt)); hit = S & sh; total |= hit; rows.append((len(hit), p.name))
    print(f"prose words: {len(mine)}   {N}-word shingles: {len(S)}")
    print(f"overlap with all local sources: {100 * len(total) / max(1, len(S)):.1f}% of shingles")
    for n, name in sorted(rows, reverse=True)[:8]: print(f"  {n:5d} shingles  {name}")
    return total, S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main.tex")
