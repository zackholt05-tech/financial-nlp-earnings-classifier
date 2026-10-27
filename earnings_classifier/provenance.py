"""Run directories and manifests.

Every run writes to its own directory, `runs/<UTC timestamp>_<mode>_<fp8>/`,
and refuses to reuse an existing one. The manifest records everything needed
to identify and reproduce the run: config, seed, argv, data mode, dataset
fingerprint, cache keys, model identity, sample counts, class balance, date
ranges, coverage, dependency versions, platform and git commit. Environment
variables are never recorded, so API keys cannot leak into run artifacts.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

MANIFEST_VERSION = 1
_PACKAGES = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy",
             "pandas", "scikit-learn", "yfinance")


def create_run_dir(runs_dir: str, data_mode: str, fingerprint: str,
                   now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
    name = f"{stamp}_{data_mode}_{fingerprint[:8]}"
    path = os.path.join(runs_dir, name)
    os.makedirs(runs_dir, exist_ok=True)
    os.mkdir(path)  # raises FileExistsError: a run directory is never reused
    return path


def dependency_versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in _PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def platform_info(device: str) -> Dict[str, str]:
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "device": device}


def git_info(repo_dir: str = ".") -> Dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True,
                              timeout=10)
    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "dirty": None, "note": "not a git repository"}
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"commit": None, "dirty": None, "note": f"git unavailable: {e}"}


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json(path: str, obj) -> None:
    with open(path, "x") as f:  # exclusive: never overwrite run artifacts
        json.dump(obj, f, indent=2, default=_json_default, allow_nan=True)


def write_predictions(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["event_id"])
        w.writeheader()
        w.writerows(rows)


def portable(obj, root: str):
    """Replace absolute paths under `root` (e.g. the repository) with relative ones,
    recursively, so manifests never record the local user's directory layout."""
    root = os.path.abspath(root).rstrip(os.sep) + os.sep
    if isinstance(obj, str):
        return obj[len(root):] if obj.startswith(root) else obj
    if isinstance(obj, dict):
        return {k: portable(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v, root) for v in obj]
    return obj


def build_manifest(root: str = ".", **sections) -> Dict:
    return {"manifest_version": MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(), **portable(sections, root)}
