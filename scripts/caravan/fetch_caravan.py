"""Get REAL Caravan data without downloading the whole 29 GB archive.

Caravan v1.6 (CSV version) is one Caravan-csv.tar.gz on Zenodo (CC-BY-4.0).
Inside it, the attribute tables and licenses come first, then the time series
grouped by source dataset, starting with CAMELS-US (671 basins, ~1.2 GB compressed).
So we only download the first part of the archive.

Step 1, download the head of the archive (resumes after dropped connections):
    python scripts/fetch_caravan.py download --gb 1.5
Step 2, extract attributes, licenses and the time series of the chosen sources:
    python scripts/fetch_caravan.py extract --sources camels

Files are written byte for byte; nothing is edited. A truncated last file is skipped.
`extract` reports which sources are complete (the archive moved on to the next
source). If a source you need is incomplete, run `download` with a larger --gb.

Default location ~/BasinPulse-data (outside OneDrive so it does not sync):
    ~/BasinPulse-data/Caravan-csv.head.tar.gz
    ~/BasinPulse-data/caravan/{attributes,licenses,timeseries/csv/<source>}/...
    ~/BasinPulse-data/caravan/fetch_manifest.json  (source, retrieval time, sha256 per file)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

RECORD = "15530022"  # Caravan v1.6, CSV version
URL = f"https://zenodo.org/records/{RECORD}/files/Caravan-csv.tar.gz?download=1"
TOTAL_BYTES = 28_951_089_034
ROOT_PREFIX = "Caravan-csv/"
BASE = Path.home() / "BasinPulse-data"


def download(dest: Path, n_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = min(n_bytes, TOTAL_BYTES)
    failures = 0
    while True:
        have = dest.stat().st_size if dest.exists() else 0
        if have >= n_bytes:
            print(f"Done: {have / 1e9:.2f} GB in {dest}", flush=True)
            return
        req = urllib.request.Request(URL, headers={
            "User-Agent": "basinpulse-fetch", "Range": f"bytes={have}-{n_bytes - 1}"})
        t0, got, last = time.time(), 0, 0.0
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                if r.status != 206:
                    raise RuntimeError(f"server ignored Range (HTTP {r.status}); not appending")
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last > 20:
                        last = now
                        tot = have + got
                        rate = got / 1e6 / max(now - t0, 1e-6)
                        eta = (n_bytes - tot) / 1e6 / max(rate, 1e-6) / 60
                        print(f"  {tot / 1e9:.2f}/{n_bytes / 1e9:.2f} GB, {rate:.2f} MB/s, ~{eta:.0f} min left",
                              flush=True)
            failures = 0
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            failures += 1
            wait = min(60, 5 * failures)
            print(f"  connection problem ({e}); resuming in {wait}s (attempt {failures})", flush=True)
            if failures > 50:
                raise
            time.sleep(wait)


def extract(archive: Path, out: Path, sources: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    order: list[str] = []  # directory order as seen in the archive
    counts: dict[str, int] = {}
    truncated = False
    try:
        with tarfile.open(archive, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.startswith(ROOT_PREFIX):
                    continue
                rel = m.name[len(ROOT_PREFIX):]
                parts = rel.split("/")
                is_ts = parts[0] == "timeseries"
                top = "/".join(parts[:3]) if is_ts else parts[0]
                if not order or order[-1] != top:
                    order.append(top)
                if is_ts and (len(parts) != 4 or parts[1] != "csv" or parts[2] not in sources):
                    continue
                data = tf.extractfile(m).read()
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                hashes[rel] = hashlib.sha256(data).hexdigest()
                if is_ts:
                    counts[parts[2]] = counts.get(parts[2], 0) + 1
                    if counts[parts[2]] % 50 == 0:
                        print(f"  {parts[2]}: {counts[parts[2]]} basins", flush=True)
    except (tarfile.ReadError, EOFError, zlib.error) as e:
        truncated = True
        print(f"Reached the end of the downloaded part ({type(e).__name__}).", flush=True)

    ts_dirs = [d for d in order if d.startswith("timeseries/csv/")]
    complete = {d.split("/")[2] for d in ts_dirs[:-1]} if truncated else {d.split("/")[2] for d in ts_dirs}
    status = {s: {"basins_saved": counts.get(s, 0), "complete": s in complete} for s in sources}
    for s, st in status.items():
        print(f"  {s}: {st['basins_saved']} basins, {'COMPLETE' if st['complete'] else 'INCOMPLETE'}")
    manifest = {
        "dataset": "Caravan v1.6 (CSV version)",
        "zenodo_record": f"https://zenodo.org/records/{RECORD}",
        "paper_doi": "https://doi.org/10.1038/s41597-023-01975-w",
        "license": "CC-BY-4.0; per-source licenses in licenses/",
        "url": URL,
        "archive_bytes_downloaded": archive.stat().st_size,
        "archive_sha256_of_downloaded_part": _sha256_file(archive),
        "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": status,
        "archive_order": order,
        "sha256": hashes,
    }
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {out / 'fetch_manifest.json'}")
    return manifest


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--gb", type=float, default=1.5, help="how much of the archive head to download")
    d.add_argument("--dest", type=Path, default=BASE / "Caravan-csv.head.tar.gz")
    e = sub.add_parser("extract")
    e.add_argument("--archive", type=Path, default=BASE / "Caravan-csv.head.tar.gz")
    e.add_argument("--out", type=Path, default=BASE / "caravan")
    e.add_argument("--sources", default="camels")
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.dest, int(a.gb * 1e9))
    else:
        extract(a.archive, a.out, [s.strip() for s in a.sources.split(",") if s.strip()])


if __name__ == "__main__":
    sys.exit(main())
