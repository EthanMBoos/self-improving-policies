# /// script
# requires-python = ">=3.10"
# dependencies = ["google-crc32c"]
# ///
"""Download openpi checkpoints from the public gs://openpi-assets bucket and verify their checksums.

Usage:
    uv run scripts/download_checkpoints.py                  # pi05_base + pi05_libero -> ./checkpoints/
    uv run scripts/download_checkpoints.py pi05_libero --out /scratch/ckpts

No GCP credentials needed (the bucket is public). Re-running skips files that are already present and valid.
"""

import argparse
import base64
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import google_crc32c

BUCKET = "openpi-assets"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def list_objects(prefix):
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size,md5Hash,crc32c),nextPageToken", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url) as r:
            d = json.load(r)
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def verify(path, obj):
    """Composite GCS objects (the large param shards) have only crc32c, no md5Hash."""
    if path.stat().st_size != int(obj["size"]):
        return False
    if "md5Hash" in obj:
        h, want = hashlib.md5(), obj["md5Hash"]
    else:
        h, want = google_crc32c.Checksum(), obj["crc32c"]
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode() == want


def fetch(obj, out):
    # checkpoints/<name>/... in the bucket -> <out>/<name>/...
    dest = out / obj["name"].removeprefix("checkpoints/")
    if dest.exists() and verify(dest, obj):
        return obj["name"], "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(obj["name"])
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    if not verify(tmp, obj):
        tmp.unlink()
        raise RuntimeError(f"checksum/size mismatch for {obj['name']}")
    tmp.rename(dest)
    return obj["name"], f"ok {int(obj['size']) / 1e9:.2f} GB"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", default=["pi05_base", "pi05_libero"], help="checkpoint names in the bucket")
    p.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    objs = [o for n in args.names for o in list_objects(f"checkpoints/{n}/")]
    if not objs:
        raise SystemExit(f"no objects found for {args.names}")
    print(f"{len(objs)} files, {sum(int(o['size']) for o in objs) / 1e9:.2f} GB -> {args.out}", flush=True)
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, o, args.out) for o in objs]):
            try:
                name, status = fut.result()
                print(f"[{status}] {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {e}", flush=True)
    if failed:
        raise SystemExit(f"FAILED: {failed} files (re-run to retry)")
    print("DONE, all files verified")


if __name__ == "__main__":
    main()
