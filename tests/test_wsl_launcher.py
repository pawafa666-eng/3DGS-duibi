import importlib.util
from pathlib import Path


def test_build_honors_selected_algorithms(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "run_wsl_smoke.py"
    spec = importlib.util.spec_from_file_location("wsl_launcher_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "/home/test\n")
    commands = []
    monkeypatch.setattr(module.subprocess, "call", lambda command, **kwargs: commands.append(command) or 0)
    assert module.main(["--build", "--only", "02", "09", "--timeout", "240"]) == 0
    assert len(commands) == 2
    assert commands[0][commands[0].index("--only") + 1:] == ["02", "09"]
    assert commands[1][-5:] == ["--only", "02", "09", "--timeout", "240"]
