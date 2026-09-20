"""Download comparison sources and their submodules; Python standard library only."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


class SetupError(RuntimeError):
    pass


def inside(parent: Path, relative: str) -> Path:
    parent = parent.resolve()
    path = (parent / relative).resolve()
    if path == parent or parent not in path.parents:
        raise SetupError(f"Path is outside the source directory: {relative}")
    return path


class Git:
    def __init__(self, retries: int, timeout: int):
        self.retries = retries
        self.timeout = timeout
        self.env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")

    def run(self, cwd: Path, *args: str, capture=False, check=True, retry=False):
        command = ["git", "--literal-pathspecs", "-c", "core.longpaths=true", "-c", "core.quotepath=false",
                   "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=60", *args]
        attempts = self.retries if retry else 1
        # A copied repo has no .git: never let 'git apply' discover the parent
        # comparison repository and silently skip all patch paths as out of scope.
        env = dict(self.env, GIT_CEILING_DIRECTORIES=str(cwd.resolve().parent))
        for attempt in range(1, attempts + 1):
            if not capture:
                print(f"  [{attempt}/{attempts}] " + subprocess.list2cmdline(command), flush=True)
            try:
                result = subprocess.run(
                    command, cwd=cwd, env=env, capture_output=capture,
                    text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as error:
                if attempt == attempts:
                    raise SetupError(f"Git timed out after {self.timeout}s; rerun setup to retry.") from error
            else:
                if result.returncode == 0 or not check:
                    return result
                if attempt == attempts:
                    detail = (result.stderr or "See Git output above.").strip()
                    raise SetupError(f"Git failed ({result.returncode}): {detail}")
            print("  Download failed; retrying in 2 seconds...", flush=True)
            time.sleep(2)


def required_files(root: Path, method: dict) -> list[str]:
    config = json.loads((inside(root, method["directory"]) / "experiment.json").read_text(encoding="utf-8-sig"))
    files = set(method.get("required_files", []))
    for stage in ("train", "test"):
        for command in config[stage]["commands"]:
            for part in command:
                if part.startswith("{repo}/") and part.endswith((".py", ".bat", ".ps1")):
                    files.add(part[len("{repo}/"):])
    if not files:
        raise SetupError("No source entrypoints found in experiment.json.")
    return sorted(files)


def validate_files(target: Path, required: list[str]) -> None:
    missing = [name for name in required if not inside(target, name).is_file()]
    if missing:
        raise SetupError(f"Incomplete source: {target}; missing: {', '.join(missing)}")


def complete_checkout(git: Git, repo: Path, revision="HEAD") -> None:
    """Recover an indexless interrupted checkout without replacing existing files."""
    if not (repo / ".git").exists():
        return
    oid = git.run(repo, "rev-parse", "--verify", revision + "^{commit}", capture=True).stdout.strip()
    index = repo / git.run(repo, "rev-parse", "--git-path", "index", capture=True).stdout.strip()
    pending = repo / git.run(repo, "rev-parse", "--git-path", "comparison-setup.pending", capture=True).stdout.strip()
    recovering = not index.exists() or pending.exists()
    if recovering:
        if pending.exists() and pending.read_text(encoding="utf-8").strip() != oid:
            raise SetupError(f"Interrupted checkout revision changed; files preserved: {repo}")
        # Keep a marker until all missing blobs are written, even if setup is
        # interrupted again after read-tree has created the index.
        pending.write_text(oid, encoding="utf-8")
        if not index.exists():
            git.run(repo, "read-tree", oid)
    entries = git.run(repo, "ls-tree", "-r", "-z", oid, capture=True).stdout
    missing = []
    for entry in entries.split("\0"):
        if not entry:
            continue
        metadata, relative = entry.split("\t", 1)
        if metadata.split()[1] != "blob":  # gitlinks are handled by fill_submodules
            continue
        path = inside(repo, relative)
        if not path.exists():
            missing.append(relative)
        elif not path.is_file():
            raise SetupError(f"Expected source file, found a directory: {path}")
    if missing:
        if not recovering:
            raise SetupError(
                f"Tracked source files are missing: {repo}: {', '.join(missing[:10])}. "
                "Existing checkout/deletions preserved; restore these files or move this repo aside and rerun."
            )
        print(f"  [recover] Restoring {len(missing)} missing source files; keeping existing files", flush=True)
        for start in range(0, len(missing), 16):
            git.run(repo, "restore", "--source", oid, "--worktree", "--", *missing[start:start + 16])
    if recovering:
        pending.unlink()


def fill_submodules(git: Git, repo: Path) -> None:
    """Initialize missing modules recursively, retaining existing revisions and edits."""
    complete_checkout(git, repo)
    modules = repo / ".gitmodules"
    if not modules.is_file():
        return
    entries = git.run(repo, "config", "-z", "--file", ".gitmodules", "--get-regexp",
                      r"^submodule\..*\.path$", capture=True, check=False)
    if entries.returncode not in (0, 1):
        raise SetupError(f"Cannot read {modules}: {entries.stderr}")
    for entry in entries.stdout.split("\0"):
        if not entry:
            continue
        _, relative = entry.split("\n", 1)
        child = inside(repo, relative)
        if (repo / ".git").exists():
            status = git.run(repo, "submodule", "status", "--", relative, capture=True).stdout
            if not status:
                raise SetupError(f"Submodule is not registered in the Git index: {relative}")
            if status.startswith("U"):
                raise SetupError(f"Submodule has an unresolved merge: {relative}")
            if status.startswith("-"):
                git.run(repo, "submodule", "update", "--init", "--progress", "--", relative, retry=True)
            elif status.startswith("+"):
                print(f"  [keep] Local submodule revision: {relative}", flush=True)
        if not child.is_dir():
            raise SetupError(f"Submodule source is missing: {child}")
        fill_submodules(git, child)
        if not any(p.name != ".git" for p in child.iterdir()):
            raise SetupError(f"Submodule source is missing: {child}")


def download(git: Git, target: Path, method: dict) -> None:
    """Publish a checkout only after checkout succeeds; retain interrupted downloads."""
    staging = inside(target.parent, "." + target.name + "-download")
    url = method["url"]
    if not staging.exists() or not any(staging.iterdir()):
        git.run(target.parent, "clone", "--depth", "1", "--no-checkout", "--progress",
                "--", url, str(staging), retry=True)
    if not (staging / ".git").exists():
        raise SetupError(f"Unrecognized download directory; files preserved: {staging}")
    origin = git.run(staging, "remote", "get-url", "origin", capture=True).stdout.strip()
    if origin != url:
        raise SetupError(f"Download directory has a different origin; files preserved: {staging}")
    revision = method.get("revision", "HEAD")
    available = git.run(staging, "rev-parse", "--verify", revision + "^{commit}", capture=True, check=False)
    if available.returncode:
        git.run(staging, "fetch", "--depth", "1", "--progress", "origin", revision, retry=True)
        revision = "FETCH_HEAD"
    complete_checkout(git, staging, revision)
    git.run(staging, "checkout", "--detach", revision)
    if target.exists():
        target.rmdir()  # only a verified empty directory; never recursive deletion
    staging.rename(target)


def apply_patches(git: Git, root: Path, repo: Path, method: dict) -> None:
    for relative in method.get("patches", []):
        patch = inside(root, relative)
        if not patch.is_file():
            raise SetupError(f"Missing bundled patch: {patch}; git pull origin main first.")
        options = ("--ignore-whitespace", "--whitespace=nowarn")
        already = git.run(repo, "apply", *options, "--reverse", "--check", str(patch), capture=True, check=False)
        if already.returncode == 0:
            print(f"  [patch present] {relative}", flush=True)
            continue
        valid = git.run(repo, "apply", *options, "--check", str(patch), capture=True, check=False)
        if valid.returncode:
            raise SetupError(f"Patch conflicts with local source; nothing overwritten: {relative}\n{valid.stderr}")
        git.run(repo, "apply", *options, str(patch))


def prepare_method(git: Git, root: Path, method: dict) -> dict:
    directory = inside(root, method["directory"])
    target = inside(directory, method.get("repo_dir", "repo"))
    required = required_files(root, method)
    directory.mkdir(parents=True, exist_ok=True)
    bundled = method.get("source_mode") == "bundled"
    if bundled:
        # These are custom implementations. Never replace them with upstream 3DGS.
        validate_files(target, required)
        print("  [bundled] Source included in this comparison repository", flush=True)
    else:
        if not target.exists() or (target.is_dir() and not any(target.iterdir())):
            print(f"  [download] {method['url']}", flush=True)
            download(git, target, method)
        else:
            print(f"  [keep] {target}", flush=True)
        fill_submodules(git, target)
        validate_files(target, required)
        apply_patches(git, root, target, method)
    commit = "bundled" if bundled else "local-copy"
    if (target / ".git").exists():
        commit = git.run(target, "rev-parse", "HEAD", capture=True).stdout.strip()
    record = {"id": method["id"], "name": method["name"], "status": "ready",
              "path": str(target), "source": method["url"], "commit": commit,
              "patches": method.get("patches", [])}
    (directory / "source_commit.txt").write_text(
        f"source={method['url']}\ncommit={commit}\n"
        f"patches={','.join(record['patches'])}\nrecorded_at={datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Comparison project directory")
    parser.add_argument("--only", nargs="+", help="Method ids, e.g. 02 03; default: all ten")
    parser.add_argument("--retries", type=int, default=3, help="Network attempts per command")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per Git command")
    args = parser.parse_args(argv)
    if args.retries < 1 or args.timeout < 1:
        parser.error("--retries and --timeout must be positive")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if shutil.which("git") is None:
        print("Git not found. Install Git for Windows, reopen this terminal, then rerun setup.")
        return 2
    root = args.root.resolve()
    try:
        methods = json.loads((root / "algorithms.json").read_text(encoding="utf-8-sig"))["methods"]
        if args.only:
            wanted = {str(value).zfill(2) for value in args.only}
            unknown = wanted - {method["id"] for method in methods}
            if unknown:
                parser.error("Unknown method ids: " + ", ".join(sorted(unknown)))
            methods = [method for method in methods if method["id"] in wanted]
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot load algorithms.json: {error}")
        return 2
    git = Git(args.retries, args.timeout)
    results = []
    print(f"Source setup: {root}\nNo training or package installation is started.", flush=True)
    try:
        for index, method in enumerate(methods, 1):
            print(f"\n[{index}/{len(methods)}] [{method['id']}] {method['name']}", flush=True)
            try:
                result = prepare_method(git, root, method)
                print(f"[{method['id']}] ready", flush=True)
            except (OSError, ValueError, KeyError, SetupError) as error:
                result = {"id": method["id"], "name": method["name"], "status": "failed", "error": str(error)}
                print(f"[{method['id']}] FAILED: {error}", flush=True)
            results.append(result)
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun the same command to continue.", flush=True)
        return 130
    report = root / "output/.repository_setup/report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"),
                                  "methods": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item["id"] for item in results if item["status"] == "failed"]
    print(f"\nSource setup: {len(results) - len(failed)}/{len(results)} ready.\nReport: {report}")
    if failed:
        print("Retry failed methods: python setup_repositories.py --only " + " ".join(failed))
        return 1
    print("All selected source repositories are ready. CUDA/Python dependencies require their own environments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
