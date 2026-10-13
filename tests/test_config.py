"""Settings env-var wiring (deterministic: no GPU, no network, no subprocesses)."""
from __future__ import annotations

from pathlib import Path


def test_log_dir_env(monkeypatch, tmp_path: Path) -> None:
    """NVLLM_LOG_DIR is honored (install.sh / the systemd unit rely on it)."""
    from nvllm_service.config import Settings

    monkeypatch.setenv("NVLLM_LOG_DIR", str(tmp_path / "logs"))
    s = Settings.from_env()
    assert s.log_dir == tmp_path / "logs"
    assert s.logs_dir() == tmp_path / "logs"


def test_log_dir_default(monkeypatch) -> None:
    """Without NVLLM_LOG_DIR, logs go to the shared-data default under $HOME."""
    from nvllm_service.config import Settings

    monkeypatch.delenv("NVLLM_LOG_DIR", raising=False)
    s = Settings.from_env()
    assert s.log_dir is None
    assert s.logs_dir() == Path.home() / ".local" / "share" / "nvllm-serve" / "logs"


def test_worker_extra_args_env(monkeypatch) -> None:
    """Regression guard: NVLLM_WORKER_EXTRA_ARGS is shlex-split (not word-split)."""
    from nvllm_service.config import Settings

    monkeypatch.setenv("NVLLM_WORKER_EXTRA_ARGS", "--enable-auto-tool-choice --tool-call-parser hermes")
    s = Settings.from_env()
    assert s.worker_extra_args == ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]


def test_worker_extra_args_unset(monkeypatch) -> None:
    from nvllm_service.config import Settings

    monkeypatch.delenv("NVLLM_WORKER_EXTRA_ARGS", raising=False)
    s = Settings.from_env()
    assert s.worker_extra_args == []
