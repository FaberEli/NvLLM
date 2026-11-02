"""Tests for `nvllm daemon` (local systemd user service management).

No real systemd: the systemctl seam (``daemon._systemctl``) is monkeypatched,
and XDG_CONFIG_HOME / NVLLM_HOME are pointed at tmp dirs — CI-safe, like the
rest of the suite.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nvllm_service import cli, daemon


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    """Isolated XDG config dir + install root (with an install.json)."""
    xdg = tmp_path / "xdg"
    root = tmp_path / "nvllm"
    root.mkdir()
    (root / "install.json").write_text(json.dumps({"root": str(root), "port": 8004, "service": False}))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("NVLLM_HOME", str(root))
    monkeypatch.delenv("NVLLM_PORT", raising=False)
    return {"xdg": xdg, "root": root}


def _fake_systemctl(calls: list, codes: dict | None = None):
    def fake(*args: str) -> subprocess.CompletedProcess:
        calls.append(args)
        code = (codes or {}).get(args, 0)
        out = {"is-enabled": "enabled", "is-active": "active"}.get(args[0], "")
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="" if code == 0 else "no such unit")
    return fake


# ---- pure helpers --------------------------------------------------------------


def test_render_unit(env):
    u = daemon.render_unit(env["root"], 8004)
    assert f"ExecStart={env['root']}/venv/bin/nvllm serve --port 8004" in u
    assert "Restart=on-failure" in u
    assert "NVLLM_WORKER_MAX_JOBS=2" in u
    assert "WantedBy=default.target" in u


def test_service_port_precedence(env, monkeypatch):
    assert daemon.service_port(env["root"]) == 8004  # from install.json
    (env["root"] / "install.json").unlink()
    monkeypatch.setenv("NVLLM_PORT", "9100")
    assert daemon.service_port(env["root"]) == 9100  # then $NVLLM_PORT
    monkeypatch.delenv("NVLLM_PORT")
    assert daemon.service_port(env["root"]) == 8004  # then the default


def test_service_port_tolerates_broken_install_json(env):
    (env["root"] / "install.json").write_text("not json")
    assert daemon.service_port(env["root"]) == 8004


# ---- install / uninstall -------------------------------------------------------


def test_install_writes_unit_and_enables(env, monkeypatch, capsys):
    calls: list = []
    monkeypatch.setattr(daemon, "_systemctl", _fake_systemctl(calls))
    rc = daemon.install()
    out = capsys.readouterr().out
    assert rc == 0
    unit = daemon.unit_path()
    assert unit.exists()
    assert "nvllm serve --port 8004" in unit.read_text(encoding="utf-8")
    assert ("daemon-reload",) in calls
    assert ("enable", "--now", "nvllm") in calls
    assert "installed and started" in out
    # install.json reflects the outcome
    assert json.loads((env["root"] / "install.json").read_text())["service"] is True


def test_install_enable_failure_returns_1_and_records(env, monkeypatch, capsys):
    calls: list = []
    monkeypatch.setattr(daemon, "_systemctl", _fake_systemctl(calls, {("enable", "--now", "nvllm"): 1}))
    rc = daemon.install()
    out = capsys.readouterr().out
    assert rc == 1
    assert "nvllm serve" in out  # actionable foreground fallback
    assert json.loads((env["root"] / "install.json").read_text())["service"] is False


def test_install_without_systemd(env, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "_systemd_available", lambda: False)
    rc = daemon.install()
    assert rc == 1
    assert daemon.unit_path().exists()  # unit written, just not enable-able
    assert "nvllm serve" in capsys.readouterr().out


def test_uninstall_removes_unit(env, monkeypatch):
    calls: list = []
    monkeypatch.setattr(daemon, "_systemctl", _fake_systemctl(calls))
    daemon.install()
    assert daemon.unit_path().exists()
    rc = daemon.uninstall()
    assert rc == 0
    assert not daemon.unit_path().exists()
    assert ("disable", "--now", "nvllm") in calls


def test_uninstall_when_absent_is_clean(env, capsys):
    rc = daemon.uninstall()
    assert rc == 0
    assert "nothing to remove" in capsys.readouterr().out


# ---- status ---------------------------------------------------------------------


def test_status_active(env, monkeypatch, capsys):
    monkeypatch.setattr(daemon, "_systemctl", _fake_systemctl([]))
    daemon.install()
    capsys.readouterr()
    rc = daemon.status()
    out = capsys.readouterr().out
    assert rc == 0
    assert "enabled:  yes" in out
    assert "active:   active" in out


def test_status_not_installed(env, capsys):
    rc = daemon.status()
    out = capsys.readouterr().out
    assert rc == 1
    assert "nvllm daemon install" in out


# ---- CLI dispatch ---------------------------------------------------------------


def test_cmd_daemon_roundtrip(env, monkeypatch):
    monkeypatch.setattr(daemon, "_systemctl", _fake_systemctl([]))
    assert cli.main(["daemon", "install"]) == 0
    assert cli.main(["daemon", "status"]) == 0
    assert cli.main(["daemon", "uninstall"]) == 0


def test_cmd_daemon_help_and_unknown_action(env, capsys):
    assert cli.main(["daemon", "help"]) == 0
    assert "nvllm daemon" in capsys.readouterr().out
    rc = cli.main(["daemon", "instal"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "did you mean 'install'" in err
