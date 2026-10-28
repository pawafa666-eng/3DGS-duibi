"""Prepare sources and isolated CUDA extensions, then run training only.

Ubuntu: python setup_and_train.py --resume 03 --only 01 03 04 06 07 08 09 10
Main-runner arguments are forwarded. --cuda can select an existing toolkit.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from common.config import discover_experiments
from main import parse_args, select_experiments


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--extension-root", type=Path, default=Path.home() / "3dgs-comparison-extensions")
    options, forwarded = parser.parse_known_args(argv)
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    training = parse_args(forwarded)
    if training.mode != "train":
        parser.error("This launcher only trains; use main.py explicitly for test/both modes")
    selected = select_experiments(discover_experiments(ROOT), training.only or training.only_alias)
    ids = [definition.id for definition in selected]
    steps = [[sys.executable, "-u", str(ROOT / "setup_repositories.py"), "--only", *ids]]
    if sys.platform == "linux":
        build = [sys.executable, "-u", str(ROOT / "wsl_build_extensions.py"),
                 "--jobs", str(options.jobs), "--extension-root", str(options.extension_root)]
        if options.cuda:
            build.extend(["--cuda", str(options.cuda)])
        build.extend(["--only", *ids])
        steps.append(build)
    for command in steps:
        result = subprocess.call(command, cwd=ROOT)
        if result:
            print("Preparation failed. Training was not started; fix the preparation error first.", flush=True)
            return result
    command = [sys.executable, "-u", str(ROOT / "main.py"), *forwarded]
    if sys.platform == "linux":
        command.extend(["--extension-root", str(options.extension_root)])
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
