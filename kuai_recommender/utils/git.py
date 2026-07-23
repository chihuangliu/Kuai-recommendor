import kuai_recommender
from pathlib import Path
import subprocess

_PROJECT_ROOT = Path(kuai_recommender.__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
