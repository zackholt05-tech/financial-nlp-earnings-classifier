"""Pre-registered model selection and the guard that keeps the test set held out."""

from __future__ import annotations

import subprocess
from typing import Dict, List, Sequence


def select_best(rows: List[Dict], score_key: str, simplicity: Sequence[str],
                decimals: int = 4) -> Dict:
    """Highest `score_key` (rounded to `decimals`, so float noise is a tie); ties go
    to the simplest row, comparing the `simplicity` fields in order (smaller = simpler)."""
    if not rows:
        raise ValueError("no candidate configurations")
    if any(r[score_key] != r[score_key] for r in rows):   # NaN check
        raise ValueError(f"undefined {score_key} among candidates")
    return min(rows, key=lambda r: (-round(r[score_key], decimals),
                                    *[r[k] for k in simplicity]))


def require_committed(path: str, repo_dir: str = ".") -> str:
    """Raise unless `path` is tracked and identical to HEAD; return the HEAD commit.

    The test stage calls this on the selection file, so the chosen configuration
    is provably fixed before any test metric is computed."""
    def git(*args):
        return subprocess.run(["git", "-C", repo_dir, *args], capture_output=True, text=True)

    if git("ls-files", "--error-unmatch", path).returncode != 0:
        raise RuntimeError(f"{path} is not committed; commit the selection before testing")
    if git("diff", "--quiet", "HEAD", "--", path).returncode != 0:
        raise RuntimeError(f"{path} differs from the committed version")
    return git("rev-parse", "HEAD").stdout.strip()
