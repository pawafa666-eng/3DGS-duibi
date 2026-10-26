"""Exercise the entry-point preambles without importing CUDA or training packages.

Run with: python -B -m unittest discover -s tests -p test_gpu_selection_portability.py -v
"""

import ast
import builtins
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("06_Scaffold_GS", "08_Octree_GS")
ENTRIES = tuple(ROOT / method / "runtime" / entry
                for method in METHODS for entry in ("train.py", "render.py"))


def run_startup(path, stdout="0, 400\n1, 20\n", environ=None, error=None):
    """Run real startup code; stub only external packages and the GPU query."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
                name.name == "torchvision" for name in node.names):
            break
        if isinstance(node, ast.ImportFrom) and node.module == "scene":
            break
        prefix.append(node)
    preamble = ast.Module(body=prefix, type_ignores=[])
    torch_environ = []

    def import_package(name, *args, **kwargs):
        if name == "torch":
            torch_environ.append(dict(os.environ))
            return SimpleNamespace()
        if name == "numpy":
            # Only the old, broken startup uses argmin. Keep its empty-input
            # failure reproducible without requiring NumPy in this test runtime.
            return SimpleNamespace(argmin=lambda values: min(
                range(len(values)), key=values.__getitem__))
        return builtins.__import__(name, *args, **kwargs)

    builtin_namespace = dict(vars(builtins), __import__=import_package)

    def query_result(argv, **options):
        if error is not None:
            raise error
        output = stdout if options.get("text") else stdout.encode("utf-8")
        return subprocess.CompletedProcess(argv, 0, output, "")

    with patch.dict(os.environ, environ or {}, clear=True), \
            patch("subprocess.run") as query, \
            patch("os.system", side_effect=AssertionError("Startup invoked a shell")):
        query.side_effect = query_result
        exec(compile(preamble, str(path), "exec"), {"__builtins__": builtin_namespace})
        return dict(os.environ), query.call_args_list, torch_environ


class GPUSelectionPortabilityTests(unittest.TestCase):
    def test_selects_least_used_gpu_before_importing_torch(self):
        for path in ENTRIES:
            with self.subTest(entry=str(path.relative_to(ROOT))):
                env, calls, torch_envs = run_startup(path)
                self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "1")
                self.assertEqual(torch_envs[0].get("CUDA_VISIBLE_DEVICES"), "1")
                self.assertEqual(len(calls), 1)

    def test_preserves_every_explicit_visibility_value_without_querying(self):
        for path in ENTRIES:
            for value in ("2", "3,1", "GPU-aabb-ccdd", "MIG-aabb/1/0", "", "-1"):
                with self.subTest(entry=str(path.relative_to(ROOT)), value=value):
                    env, calls, torch_envs = run_startup(
                        path, environ={"CUDA_VISIBLE_DEVICES": value})
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], value)
                    self.assertEqual(torch_envs[0]["CUDA_VISIBLE_DEVICES"], value)
                    self.assertEqual(calls, [])

    def test_windows_csv_uses_reported_index_not_row_position(self):
        for path in ENTRIES:
            with self.subTest(entry=str(path.relative_to(ROOT))):
                env, _, _ = run_startup(path, stdout=" 7, 500\r\n 3, 12")
                self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "3")

    def test_equal_memory_keeps_first_reported_gpu(self):
        for path in ENTRIES:
            with self.subTest(entry=str(path.relative_to(ROOT))):
                env, _, _ = run_startup(path, stdout="7, 10\n3, 10\n")
                self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "7")

    def test_skips_invalid_rows_when_valid_devices_are_available(self):
        output = "warning\n0, N/A\n-1, 1\n2, -1\n4, 2, extra\n7, 40\n3, 12\n"
        for path in ENTRIES:
            with self.subTest(entry=str(path.relative_to(ROOT))):
                env, _, _ = run_startup(path, stdout=output)
                self.assertEqual(env.get("CUDA_VISIBLE_DEVICES"), "3")

    def test_empty_or_unusable_output_keeps_cuda_default(self):
        for path in ENTRIES:
            for output in ("", "\r\n", "Failed to initialize NVML: Unknown Error\n",
                           "No devices were found\n", "0, N/A\n1, [Not Supported]\n",
                           "-1, 0\n1, -2\n", "0, 1, extra\n"):
                with self.subTest(entry=str(path.relative_to(ROOT)), output=output):
                    env, _, torch_envs = run_startup(path, stdout=output)
                    self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
                    self.assertNotIn("CUDA_VISIBLE_DEVICES", torch_envs[0])

    def test_query_failures_keep_cuda_default(self):
        errors = (
            FileNotFoundError("nvidia-smi not installed"),
            PermissionError("nvidia-smi not executable"),
            subprocess.CalledProcessError(1, ["nvidia-smi"],
                                          stderr="Failed to initialize NVML"),
            subprocess.TimeoutExpired(["nvidia-smi"], 5),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid output"),
        )
        for path in ENTRIES:
            for error in errors:
                with self.subTest(entry=str(path.relative_to(ROOT)), error=type(error).__name__):
                    env, _, torch_envs = run_startup(path, error=error)
                    self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
                    self.assertNotIn("CUDA_VISIBLE_DEVICES", torch_envs[0])

    def test_query_uses_bounded_argument_list_without_shell(self):
        for path in ENTRIES:
            with self.subTest(entry=str(path.relative_to(ROOT))):
                _, calls, _ = run_startup(path)
                self.assertEqual(len(calls), 1)
                argv = calls[0].args[0]
                options = calls[0].kwargs
                self.assertIsInstance(argv, list)
                self.assertEqual(argv, ["nvidia-smi", "--query-gpu=index,memory.used",
                                        "--format=csv,noheader,nounits"])
                self.assertFalse(options.get("shell", False))
                self.assertTrue(options["check"])
                self.assertGreater(options["timeout"], 0)
                self.assertLessEqual(options["timeout"], 10)


class PortabilityPatchTests(unittest.TestCase):
    def test_manifest_registers_each_portability_patch(self):
        manifest = json.loads((ROOT / "algorithms.json").read_text(encoding="utf-8"))
        for directory in METHODS:
            with self.subTest(method=directory):
                method = next(item for item in manifest["methods"]
                              if item["directory"] == directory)
                relative = "patches/" + directory + "_portability.patch"
                self.assertIn(relative, method.get("patches", []))
                self.assertTrue((ROOT / relative).is_file())

    @unittest.skipUnless(shutil.which("git"), "Git is needed for patch replay")
    def test_patches_apply_to_clean_sources_and_match_local_entries(self):
        for directory in METHODS:
            with self.subTest(method=directory):
                repo = ROOT / directory / "repo"
                patch_path = ROOT / "patches" / (directory + "_portability.patch")
                self.assertTrue(patch_path.is_file())
                if not (repo / ".git").exists():
                    self.skipTest("Patch replay needs the upstream Git checkout")
                with tempfile.TemporaryDirectory(prefix="gpu-portability-") as temp:
                    clean = Path(temp)
                    for entry in ("train.py", "render.py"):
                        original = subprocess.run(
                            ["git", "-C", str(repo), "show", "HEAD:" + entry],
                            check=True, stdout=subprocess.PIPE).stdout
                        (clean / entry).write_bytes(original)
                    command = ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn"]
                    subprocess.run(command + ["--check", str(patch_path)], cwd=clean, check=True)
                    subprocess.run(command + [str(patch_path)], cwd=clean, check=True)
                    for entry in ("train.py", "render.py"):
                        self.assertEqual((clean / entry).read_text(encoding="utf-8"),
                                         (ROOT / directory / "runtime" / entry).read_text(encoding="utf-8"))
                    subprocess.run(command + ["--reverse", "--check", str(patch_path)],
                                   cwd=clean, check=True)


if __name__ == "__main__":
    unittest.main()
