"""`attach_file_logging` — the service log file (service.log) the `nvllm logs` command reads."""
from __future__ import annotations

import logging


def test_attach_writes_service_log(tmp_path):
    from nvllm_service.cli import attach_file_logging

    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level  # bare test process defaults to WARNING; run_server sets INFO
    target = attach_file_logging(tmp_path / "logs")
    try:
        root.setLevel(logging.INFO)
        assert str(target).endswith("service.log")
        logging.getLogger("nvllm.test.filelog").info("hello file log")
        assert target.exists()
        assert "hello file log" in target.read_text(encoding="utf-8")
    finally:
        root.handlers = saved  # don't let the handler outlive the tmp dir
        root.setLevel(saved_level)
