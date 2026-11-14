"""Dated, verified copy of a project folder. COPIES (never moves); the source is only READ. Refuses to write into an existing destination.

    python tools/cleanup/backup_project.py --src D:\\bev_Fresh\\bevcapstone --dst-root D:\\BEV_Capstone_Backups\\2026-11-14 --dry-run   # count files and size, check free space
    python tools/cleanup/backup_project.py --src ... --dst-root ...                                                                    # copy + verify

What it does: copies every file (default exclusions: venv, __pycache__) to <dst-root>/<source folder name>/, hashing the bytes as they are read, retries a file that changes while it is
being copied (for example the active training run), re-hashes every copy, then runs a second full verification pass over the destination only. Writes SHA256SUMS.txt (sha256sum format),
MANIFEST.json and BACKUP_REPORT.md into <dst-root>. Re-verify any time with:  cd <dst-root>/<name> && sha256sum -c ../SHA256SUMS.txt

This is the same logic that produced the 14 Nov 2026 backup (5,072 files, 1,431.8 MB, 0 failures), made configurable.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

CHUNK = 1 << 20


def lp(p):
    p = os.path.abspath(p)
    return p if p.startswith("\\\\?\\") or os.name != "nt" else "\\\\?\\" + p


def sha_file(path):
    h = hashlib.sha256()
    with open(lp(path), "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def stat_key(path):
    s = os.stat(lp(path))
    return (s.st_size, s.st_mtime_ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--exclude", nargs="*", default=["venv", "__pycache__"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src, dst_root, excl = os.path.abspath(a.src), os.path.abspath(a.dst_root), set(a.exclude)
    dst = os.path.join(dst_root, os.path.basename(src))
    files = []
    for dirpath, dirnames, filenames in os.walk(lp(src)):
        dirnames[:] = sorted(d for d in dirnames if d not in excl)
        for fn in sorted(filenames):
            files.append(os.path.relpath(os.path.join(dirpath, fn), lp(src)))
    total = sum(os.stat(lp(os.path.join(src, r))).st_size for r in files)
    free = shutil.disk_usage(os.path.splitdrive(dst_root)[0] + os.sep if os.name == "nt" else os.path.dirname(dst_root)).free
    print("files: %d | size: %.1f MB | free space at destination: %.1f GB" % (len(files), total / 1e6, free / 1e9))
    if a.dry_run:
        print("dry run: nothing copied; free space %s 2x size" % ("OK:" if free > 2 * total else "NOT enough vs"))
        return
    if os.path.exists(dst_root):
        sys.exit("destination already exists, refusing to overwrite: " + dst_root)
    if free < 2 * total:
        sys.exit("not enough free space (need at least 2x the data size)")
    os.makedirs(dst)
    entries, retried, in_flux = [], 0, []
    t0 = time.time()
    for rel in files:
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(lp(d)), exist_ok=True)
        ok = False
        for attempt in range(1, 5):
            k1 = stat_key(s)
            h, n = hashlib.sha256(), 0
            with open(lp(s), "rb") as fi, open(lp(d), "wb") as fo:
                for b in iter(lambda: fi.read(CHUNK), b""):
                    h.update(b)
                    fo.write(b)
                    n += len(b)
            if stat_key(s) != k1 or n != k1[0]:
                retried += 1
                time.sleep(2)
                continue
            shutil.copystat(lp(s), lp(d))
            if sha_file(d) != h.hexdigest():
                retried += 1
                continue
            entries.append(dict(path=rel.replace("\\", "/"), size=n, sha256=h.hexdigest(), verified=True, attempts=attempt))
            ok = True
            break
        if not ok:
            entries.append(dict(path=rel.replace("\\", "/"), size=-1, sha256="", verified=False, attempts=4, note="IN_FLUX"))
            in_flux.append(rel.replace("\\", "/"))
    bad = [e["path"] for e in entries if e["verified"] and sha_file(os.path.join(dst, e["path"])) != e["sha256"]]
    with open(os.path.join(dst_root, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), source=src, destination=dst, excluded=sorted(excl), files=len(entries), in_flux=in_flux, failures=bad, entries=entries), f, indent=1)
    with open(os.path.join(dst_root, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines("%s  %s\n" % (e["sha256"], e["path"]) for e in entries if e["verified"])
    with open(os.path.join(dst_root, "BACKUP_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Backup report\n- source: %s\n- destination: %s\n- files: %d | seconds: %.0f\n- verified failures: %d | in flux: %d (%s)\n- retries: %d\n"
                % (src, dst, len(entries), time.time() - t0, len(bad), len(in_flux), ", ".join(in_flux[:5]), retried))
    print("done: %d files, %d verification failures, %d in flux" % (len(entries), len(bad), len(in_flux)))


if __name__ == "__main__":
    main()
