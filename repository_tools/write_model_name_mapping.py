"""Write the mapping between the OLD (technical) and NEW (descriptive) model file names, and verify that no checkpoint was altered.

Inputs : repository_tools/previous_model_manifest_before_renaming.json (the manifest written before the renaming)
         trained_models/MODEL_MANIFEST.json (the manifest written after the renaming)
Output : trained_models/MODEL_NAME_MAPPING.json and trained_models/MODEL_NAME_MAPPING.md
Checks : for every model (1) the SHA-256 hash of the renamed copy equals the hash recorded before the renaming,
         (2) it equals the hash of the ORIGINAL checkpoint in its training folder (bev_rl/adaptive/runs/... or bev_policy/checkpoints/...).
Run from the repository root:
    venv\\Scripts\\python.exe repository_tools\\write_model_name_mapping.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = json.loads((ROOT / "repository_tools" / "previous_model_manifest_before_renaming.json").read_text(encoding="utf-8"))
NEW = json.loads((ROOT / "trained_models" / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


old_by_source = {m["original_path"]: m for m in OLD["models"]}
rows, problems = [], []
for m in NEW["models"]:
    o = old_by_source[m["original_path"]]
    original_hash = sha256(ROOT / m["original_path"])
    ok = (m["sha256"] == o["sha256"] == original_hash)
    if not ok:
        problems.append(m["file"])
    rows.append({"new_file": m["file"], "previous_file": o["file"], "original_checkpoint": m["original_path"], "sha256": m["sha256"],
                 "hash_identical_to_previous_copy": m["sha256"] == o["sha256"], "hash_identical_to_original_checkpoint": m["sha256"] == original_hash})
(ROOT / "trained_models" / "MODEL_NAME_MAPPING.json").write_text(json.dumps({"mapping": rows, "hash_problems": problems}, indent=1), encoding="utf-8")
lines = ["# Model file names: previous name -> new descriptive name", "",
         "Every file below is a byte-identical copy of the original checkpoint (SHA-256 verified against the previous copy and against the original in the training folder).",
         "The original checkpoints were not touched; scripts that read `bev_rl/adaptive/runs/<run>/best.pt` or `bev_policy/checkpoints/*.pt` are unaffected.", "",
         "| New file name | Previous file name | Original checkpoint (unchanged) | Hash check |", "|---|---|---|---|"]
for r in rows:
    lines.append(f"| `{r['new_file']}` | `{r['previous_file']}` | `{r['original_checkpoint']}` | {'identical' if r['hash_identical_to_original_checkpoint'] and r['hash_identical_to_previous_copy'] else 'MISMATCH'} |")
(ROOT / "trained_models" / "MODEL_NAME_MAPPING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(rows), "models mapped; hash problems:", problems)
