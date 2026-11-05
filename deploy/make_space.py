"""Build (and optionally publish) the public BasinPulse website as a Hugging Face Docker Space.

    python deploy/make_space.py                                   # build deploy/space_build/ only
    python deploy/make_space.py --publish <hf-user>/basinpulse    # build and upload (after: hf auth login)

The bundle holds only what the website needs: web/ (site + data built from the logged test run), predict.py,
src/, the climate-index recipe from scripts/caravan/prepare_caravan.py, and the submitted model
models/basinpulse_final.joblib. No training or test data is included. The container serves the site on port 7860.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "space_build"
FILES = ["predict.py", "web/server.py", "models/basinpulse_final.joblib", "models/basinpulse_final.json",
         "scripts/caravan/prepare_caravan.py"]
TREES = ["src", "web/static"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py")


def build() -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    for t in TREES:
        shutil.copytree(ROOT / t, OUT / t, ignore=SKIP)
    for pkg in ("scripts", "scripts/caravan"):          # importable as scripts.caravan.prepare_caravan
        (OUT / pkg / "__init__.py").touch()
    shutil.copy2(HERE / "Dockerfile.space", OUT / "Dockerfile")
    shutil.copy2(HERE / "requirements-web.txt", OUT / "requirements.txt")
    shutil.copy2(HERE / "SPACE_README.md", OUT / "README.md")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"built {OUT} ({size:.1f} MB)")
    return OUT


def publish(repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()   # uses the token saved by `hf auth login`; it is never read or printed here
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(OUT),
                      commit_message="Deploy BasinPulse website")
    print(f"published: https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--publish", metavar="USER/SPACE", default=None)
    a = ap.parse_args()
    build()
    if a.publish:
        publish(a.publish)
