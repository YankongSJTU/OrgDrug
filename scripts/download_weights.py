# -*- coding: utf-8 -*-
"""Download the pretrained OrgDrug weights and verify checksums.

The full weight pack (28 checkpoints + configs, ~710 MB) is hosted on
Google Drive (link in the README / WEIGHTS_URL below). After download the
files are placed in `models/` at the repo root, exactly where the training,
transfer, and toxicity scripts expect them.

Usage:
  python scripts/download_weights.py                 # download + verify
  python scripts/download_weights.py --dest /path    # custom destination
  python scripts/download_weights.py --verify-only   # check existing files
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------[ EDIT ME ]
WEIGHTS_URL = ("https://drive.usercontent.google.com/download?"
               "id=1chcFRRfo2bC6sWSxcDfGPTI0-pVG1uGQ&export=download&confirm=t")
# Human-readable share link:
#   https://drive.google.com/file/d/1chcFRRfo2bC6sWSxcDfGPTI0-pVG1uGQ/view
# ---------------------------------------------------------------------------

DEST_DEFAULT = Path(__file__).resolve().parents[1] / "models"
CHECKSUMS = "SHA256SUMS.txt"          # shipped inside the pack


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(models_dir: Path) -> int:
    sums = models_dir / CHECKSUMS
    if not sums.exists():
        print(f"[verify] {sums} not found — skip")
        return 0
    bad = 0
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip()
        f = models_dir / name
        if not f.exists():
            print(f"[MISSING] {name}")
            bad += 1
            continue
        if sha256(f) != digest:
            print(f"[CORRUPT] {name}")
            bad += 1
        else:
            print(f"[ok] {name}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEST_DEFAULT))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        sys.exit(1 if verify(dest) else 0)

    zip_path = dest / "orgdrug_weights.zip"
    print("downloading", WEIGHTS_URL)
    req = urllib.request.Request(WEIGHTS_URL,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"
  {done/1e6:7.1f} / {total/1e6:.1f} MB",
                      end="", flush=True)
    print()
    if zip_path.stat().st_size < 1e6:
        print("downloaded file is too small — Google Drive may have shown an "
              "interstitial page. Open the share link in a browser, download "
              "manually, and re-run with --verify-only.")
        sys.exit(3)
    print("downloaded", zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()

    bad = verify(dest)
    if bad:
        print(f"{bad} file(s) failed verification")
        sys.exit(1)
    print("all weights verified →", dest)


if __name__ == "__main__":
    main()
