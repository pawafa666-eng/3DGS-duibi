"""Complete source repositories, then continue training methods 02-09 in run 05."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    result = subprocess.call(
        [sys.executable, "-u", str(ROOT / "setup_repositories.py")], cwd=ROOT
    )
    if result != 0:
        print("Source setup failed. Training was not started.", flush=True)
        return result

    return subprocess.call(
        [sys.executable, "-u", str(ROOT / "main.py"),
         "--mode", "train", "--resume", "05",
         "--only", "02", "03", "04", "05", "06", "07", "08", "09",
         "--continue-on-error"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
