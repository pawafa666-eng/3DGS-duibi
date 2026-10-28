import setup_and_train as launcher


def test_launcher_builds_selected_extensions_before_training(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    calls = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert launcher.main(["--resume", "03", "--only", "01", "03", "--cuda", "/usr/local/cuda-11.8",
                          "--extension-root", str(tmp_path), "--continue-on-error"]) == 0
    assert len(calls) == 3
    assert "setup_repositories.py" in calls[0][2]
    assert "wsl_build_extensions.py" in calls[1][2]
    assert calls[1][calls[1].index("--only") + 1:] == ["01", "03"]
    assert calls[1][calls[1].index("--cuda") + 1] == "/usr/local/cuda-11.8"
    assert "main.py" in calls[2][2]
    assert calls[2][calls[2].index("--resume") + 1] == "03"
    assert calls[2][calls[2].index("--extension-root") + 1] == str(tmp_path)


def test_failed_extension_build_never_starts_training(monkeypatch):
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return 0 if len(calls) == 1 else 9
    monkeypatch.setattr(launcher.subprocess, "call", execute)
    assert launcher.main(["--resume", "03", "--only", "01"]) == 9
    assert len(calls) == 2
