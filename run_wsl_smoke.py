"""Run the real short-training checks in the configured WSL Ubuntu environment.

Windows: python run_wsl_smoke.py [--build] [--only 02 03] [--timeout 180]
The Linux environment and dependencies must already be installed. --build first
builds isolated CUDA extensions, then runs the checks. All other arguments are
forwarded to synthetic_smoke.py. No BAT file or Windows CUDA runtime is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--linux-python", help="Override the WSL environment's Python path")
    parser.add_argument("--build", action="store_true", help="Build CUDA extensions before short training")
    args, forwarded = parser.parse_known_args(argv)
    if sys.platform != "win32":
        parser.error("This launcher runs on Windows; inside Ubuntu use synthetic_smoke.py")
    prefix = ["wsl", "-d", args.distro, "--exec"]
    try:
        home = subprocess.check_output(prefix + ["printenv", "HOME"], encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Cannot open WSL {args.distro}: {error}", file=sys.stderr)
        return 2
    if not home.startswith("/") or "\n" in home:
        print(f"Unexpected Linux home directory: {home!r}", file=sys.stderr)
        return 2
    python = args.linux_python or f"{home}/3dgs-comparison-env/bin/python"
    extension_root = f"{home}/3dgs-comparison-extensions"
    if args.build:
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument("--only", nargs="+")
        selection, _ = selector.parse_known_args(forwarded)
        build = prefix + [python, "-u", "./wsl_build_extensions.py", "--extension-root", extension_root]
        if selection.only:
            build += ["--only"] + selection.only
        code = subprocess.call(build, cwd=ROOT)
        if code:
            return code
    command = prefix + [python, "-u", "./synthetic_smoke.py", "--extension-root", extension_root] + forwarded
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
