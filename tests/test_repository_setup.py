from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "setup_repositories.py"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")


def git(root, *args):
    result = subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-c", "core.autocrlf=false",
         "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def upstream(path):
    path.mkdir(parents=True)
    git(path, "init")
    (path / "train.py").write_text("# training entry\n", encoding="utf-8")
    (path / "render.py").write_text("# rendering entry\n", encoding="utf-8")
    commit(path)
    return path


def commit(path):
    git(path, "add", ".")
    git(path, "-c", "user.name=Setup Test", "-c", "user.email=setup@example.invalid",
        "commit", "-m", "test fixture")


def configure(root, methods):
    root.mkdir(parents=True, exist_ok=True)
    (root / "algorithms.json").write_text(json.dumps({"methods": methods}), encoding="utf-8")
    for method in methods:
        directory = root / method["directory"]
        directory.mkdir(exist_ok=True)
        (directory / "experiment.json").write_text(json.dumps({
            "train": {"commands": [["python", "{repo}/train.py"]]},
            "test": {"commands": [["python", "{repo}/render.py"]]},
        }), encoding="utf-8")


def method(number, url, **extra):
    return {"id": number, "name": "Demo", "directory": number + "_demo",
            "url": str(url), **extra}


def run_setup(root, *args):
    env = dict(os.environ, GIT_ALLOW_PROTOCOL="file", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--retries", "1", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=60,
    )


def test_fresh_setup_downloads_nested_submodules_and_preserves_edits_on_rerun(tmp_path):
    leaf = upstream(tmp_path / "leaf")
    library = upstream(tmp_path / "library")
    git(library, "submodule", "add", str(leaf), "third party/leaf")
    commit(library)
    source = upstream(tmp_path / "upstream")
    git(source, "submodule", "add", str(library), "submodules/library")
    commit(source)
    root = tmp_path / "中文 path" / "comparison"
    configure(root, [method("02", source)])
    target = root / "02_demo/repo"
    target.mkdir()  # an empty directory must not be mistaken for ready source

    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(root) in result.stdout  # Chinese paths remain readable in captured logs
    assert (target / "submodules/library/third party/leaf/train.py").is_file()
    (target / "train.py").write_text("# user changes\n", encoding="utf-8")
    assert run_setup(root).returncode == 0
    assert (target / "train.py").read_text() == "# user changes\n"


def test_existing_clone_completes_uninitialized_submodules(tmp_path):
    leaf = upstream(tmp_path / "leaf")
    source = upstream(tmp_path / "upstream")
    git(source, "submodule", "add", str(leaf), "submodules/library")
    commit(source)
    root = tmp_path / "comparison"
    configure(root, [method("02", source)])
    git(root, "clone", str(source), str(root / "02_demo/repo"))
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "02_demo/repo/submodules/library/train.py").is_file()


def test_failure_continues_to_later_methods_and_is_reported(tmp_path):
    source = upstream(tmp_path / "upstream")
    root = tmp_path / "comparison"
    configure(root, [method("02", tmp_path / "missing"), method("03", source)])
    result = run_setup(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (root / "03_demo/repo/train.py").is_file()
    report = json.loads((root / "output/.repository_setup/report.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in report["methods"]] == ["failed", "ready"]


def test_existing_non_git_files_are_preserved_when_incomplete(tmp_path):
    source = upstream(tmp_path / "upstream")
    root = tmp_path / "comparison"
    configure(root, [method("02", source)])
    target = root / "02_demo/repo"
    target.mkdir()
    keep = target / "notes.txt"
    keep.write_text("keep this", encoding="utf-8")
    result = run_setup(root)
    assert result.returncode == 1
    assert keep.read_text() == "keep this"
    assert not (target / ".git").exists()


def test_bundled_source_is_checked_without_external_clone(tmp_path):
    root = tmp_path / "comparison"
    configure(root, [method("10", "bundled runtime", repo_dir="runtime", source_mode="bundled")])
    target = root / "10_demo/runtime"
    target.mkdir()
    (target / "train.py").write_text("# custom train", encoding="utf-8")
    (target / "render.py").write_text("# custom render", encoding="utf-8")
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / ".git").exists()


def test_pinned_patch_is_applied_once_and_existing_local_changes_survive(tmp_path):
    source = upstream(tmp_path / "upstream")
    revision = git(source, "rev-parse", "HEAD")
    (source / "render.py").write_text("# a newer upstream revision\n", encoding="utf-8")
    commit(source)
    root = tmp_path / "comparison"
    # file:// honors --depth, so setup must fetch the older pinned commit.
    configure(root, [method("02", source.as_uri(), revision=revision, patches=["mask.patch"])])
    (root / "mask.patch").write_text(
        "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n"
        "@@ -1 +1,2 @@\n # training entry\n+# mask support\n", encoding="utf-8",
    )
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    target = root / "02_demo/repo"
    assert "# mask support" in (target / "train.py").read_text()
    assert (target / "render.py").read_text() == "# rendering entry\n"
    (target / "render.py").write_text("# user edits", encoding="utf-8")
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "train.py").read_text().count("# mask support") == 1
    assert (target / "render.py").read_text() == "# user edits"
    assert git(target, "rev-parse", "HEAD") == revision


def test_interrupted_checkout_is_reused_and_only_limits_selection(tmp_path):
    source = upstream(tmp_path / "upstream")
    root = tmp_path / "comparison"
    configure(root, [method("02", source), method("03", source)])
    staging = root / "02_demo/.repo-download"
    git(root, "clone", "--no-checkout", str(source), str(staging))
    # Interrupted checkout can write blobs before it publishes the index.
    (staging / "train.py").write_text("# training entry\n", encoding="utf-8")
    result = run_setup(root, "--only", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "02_demo/repo/train.py").is_file()
    assert not staging.exists()
    assert not (root / "03_demo/repo").exists()


@pytest.mark.parametrize("index_written", [False, True])
def test_partial_submodule_checkout_restores_missing_sources_and_nested_modules(tmp_path, index_written):
    leaf = upstream(tmp_path / "leaf")
    library = upstream(tmp_path / "library")
    git(library, "submodule", "add", str(leaf), "third_party/leaf")
    commit(library)
    source = upstream(tmp_path / "upstream")
    git(source, "submodule", "add", str(library), "submodules/library")
    commit(source)
    root = tmp_path / "comparison"
    configure(root, [method("02", source)])
    target = root / "02_demo/repo"
    git(root, "clone", str(source), str(target))
    git(target, "submodule", "init")
    child = target / "submodules/library"
    git(target, "clone", "--no-checkout", str(library), str(child))
    (child / "train.py").write_text("# user edits survived interruption\n", encoding="utf-8")
    if index_written:
        # A second interruption, after index publication but before blob recovery.
        oid = git(child, "rev-parse", "HEAD")
        git(child, "read-tree", oid)
        pending = child / git(child, "rev-parse", "--git-path", "comparison-setup.pending")
        pending.write_text(oid, encoding="utf-8")
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (child / "render.py").is_file()
    assert (child / "third_party/leaf/train.py").is_file()
    assert (child / "train.py").read_text() == "# user edits survived interruption\n"


def test_missing_tracked_source_is_reported_without_overwriting_user_deletion(tmp_path):
    source = upstream(tmp_path / "upstream")
    (source / "helper.py").write_text("# library", encoding="utf-8")
    commit(source)
    root = tmp_path / "comparison"
    configure(root, [method("02", source)])
    target = root / "02_demo/repo"
    git(root, "clone", str(source), str(target))
    (target / "helper.py").unlink()
    result = run_setup(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "helper.py" in result.stdout
    assert not (target / "helper.py").exists()


def test_recovery_uses_literal_filenames_and_preserves_similar_named_edits(tmp_path):
    source = upstream(tmp_path / "upstream")
    (source / "values[1].py").write_text("# bracket name\n", encoding="utf-8")
    (source / "values1.py").write_text("# upstream\n", encoding="utf-8")
    commit(source)
    root = tmp_path / "comparison"
    configure(root, [method("02", source)])
    staging = root / "02_demo/.repo-download"
    git(root, "clone", "--no-checkout", str(source), str(staging))
    (staging / "values1.py").write_text("# user edits\n", encoding="utf-8")
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    target = root / "02_demo/repo"
    assert (target / "values[1].py").read_text() == "# bracket name\n"
    assert (target / "values1.py").read_text() == "# user edits\n"


def test_patch_applies_to_copied_source_inside_parent_git_repository(tmp_path):
    root = tmp_path / "comparison"
    configure(root, [method("02", "unused", patches=["mask.patch"])])
    git(root, "init")
    target = root / "02_demo/repo"
    target.mkdir()
    (target / "train.py").write_text("# training entry\n", encoding="utf-8")
    (target / "render.py").write_text("# rendering entry\n", encoding="utf-8")
    (root / "mask.patch").write_text(
        "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n"
        "@@ -1 +1,2 @@\n # training entry\n+# mask support\n", encoding="utf-8",
    )
    result = run_setup(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# mask support" in (target / "train.py").read_text()
    assert not (root / "train.py").exists()
    assert run_setup(root).returncode == 0
    assert (target / "train.py").read_text().count("# mask support") == 1
