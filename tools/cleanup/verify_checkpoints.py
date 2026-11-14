"""Read-only checkpoint integrity check.

1. Verifies every entry of trained_models/MODEL_MANIFEST.json (SHA-256) against the file in trained_models/.
2. If bev_rl/adaptive/runs/ exists (a working copy that has run directories), reports which runs/<run>/best.pt are byte-identical to a trained_models/ copy.
Nothing is written or changed.

    python tools/cleanup/verify_checkpoints.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    man = json.loads((ROOT / "trained_models" / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    ents = man if isinstance(man, list) else man.get("models", man.get("files", man))
    if isinstance(ents, dict):
        ents = [dict(path=k, **(v if isinstance(v, dict) else {"sha256": v})) for k, v in ents.items()]
    ok = bad = miss = 0
    hashes = {}
    for e in ents:
        rel = e.get("path") or e.get("file") or e.get("new_name")
        h = e.get("sha256") or e.get("hash")
        q = ROOT / "trained_models" / rel
        if not q.exists():
            q = ROOT / rel
        if not q.exists():
            if "06_Third_Party" in rel:      # git-ignored third-party weights are absent in a fresh clone by design
                print("NOT IN CLONE (git-ignored third-party weights):", rel)
                continue
            miss += 1
            print("MISSING", rel)
        elif h and sha(q) == h:
            ok += 1
            hashes[h] = q
        else:
            bad += 1
            print("HASH MISMATCH", rel)
    print("trained_models/MODEL_MANIFEST.json: %d OK, %d mismatch, %d missing" % (ok, bad, miss))
    runs = ROOT / "bev_rl" / "adaptive" / "runs"
    if runs.exists():
        same = sum(1 for p in runs.glob("*/best.pt") if sha(p) in hashes)
        total = len(list(runs.glob("*/best.pt")))
        print("runs/*/best.pt identical to a trained_models copy: %d of %d" % (same, total))
    else:
        print("(no bev_rl/adaptive/runs/ in this clone; run it in a working copy to compare runs/ with trained_models/)")


if __name__ == "__main__":
    main()
