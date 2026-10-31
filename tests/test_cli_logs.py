"""`nvllm logs` + `nvllm doctor` CLI (real HTTP, like the other client commands)."""
from __future__ import annotations

from nvllm_service import cli


def test_logs_lists_or_reports_empty(nvllm_env, capsys):
    # the live server's logs dir starts empty (no workers spawned in tests)
    rc = cli.main(["logs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no log files found" in out or "log files in" in out


def test_logs_named_file_not_found(nvllm_env, capsys):
    rc = cli.main(["logs", "does-not-exist.log"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


def test_doctor_prints_report(nvllm_env, capsys):
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nvllm doctor" in out
    assert "gpu" in out
    assert "models" in out
    assert "router" in out
    assert "logs" in out
    assert "state" in out


def test_doctor_offline_is_actionable(nvllm_env, monkeypatch, capsys):
    monkeypatch.setenv("NVLLM_SERVER", "http://127.0.0.1:1")  # nothing listens here
    rc = cli.main(["doctor"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "nvllm" in err
