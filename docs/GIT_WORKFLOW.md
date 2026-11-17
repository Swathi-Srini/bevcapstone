# Git workflow for this repository (14 November 2026)

## Branch roles

| Branch | Role | Rule |
|---|---|---|
| `main` | integration branch; contains everything merged so far, including the camera-pipeline work (PR #14, merge `7e31b6f`) | change only through reviewed pull requests |
| `cleaned-latest` (`4d45920`) | frozen snapshot of the 13 Nov work, before PR #14 | never modified, never deleted |
| `camera-pipeline-base-models-2026-11-14` | feature branch, already merged | keep; no further commits |
| `cleanup/...` | cleanup pull requests (documentation first, code changes only after the experiments) | always branched from the latest `main` in a **separate clone** |

## Never run git in a working copy that has running jobs

Training and evaluation workers import code from the working folder **every round**, and they check a hash manifest of the frozen files before they start. In such a folder do **not** run `git checkout`, `git switch`, `git pull`, `git merge`, `git rebase`, `git reset`, `git stash` or `git clean`:
1. Files that exist only on the current branch (for example `bev_rl/adaptive/camera_dagger.py`) would disappear from the working tree when another branch is checked out.
2. `core.autocrlf` is `true` on the maintainer's Windows machine, so Git rewrites line endings. Several frozen files are LF in the working copy while the manifest was computed on those bytes; a rewrite changes their hash and every worker aborts at the next round.

Do any git work in a second clone instead:

```bash
git clone --branch main https://github.com/Swathi-Srini/bevcapstone <other-folder>
cd <other-folder>
git config core.autocrlf false          # local to this clone; avoids whole-file line-ending diffs
git checkout -b cleanup/<topic>-<date>
# edit, commit, push the branch, open a pull request into main
```

## Line endings (current state and plan)

* Git stores LF. On the maintainer's machine `core.autocrlf=true`, there is no `.gitattributes`, and 343 tracked files have CRLF working copies, 110 have LF. Two active queue scripts are CRLF.
* `FROZEN_MANIFEST.json` hashes the **working-copy bytes**. Consequence: the committed blobs of `assembler.py`, `bev_state/__init__.py`, `weights.json` and `retention_table.json` do not match the manifest, and a clone on another machine (or with `autocrlf=false`) cannot pass the freeze check for those files.
* Planned fix (after the experiments, see `docs/CLEANUP_LOG.md`): add a `.gitattributes`, normalise endings, and make the freeze check hash with CRLF normalised to LF, using a **new** manifest file; the existing manifest stays unchanged.

## Opening a cleanup pull request

1. Clone as above; branch from `main`.
2. Keep each pull request to one theme (documentation, paths, line endings, archiving).
3. Validate before pushing: `python tools/cleanup/check_refs.py` (no new broken references), `git ls-files -ci --exclude-standard` (no tracked file became ignored), `git diff --stat` (only the intended files).
4. Push the branch and open the pull request into `main`; merge only after review.
