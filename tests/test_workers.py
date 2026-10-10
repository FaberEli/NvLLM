"""Real WorkerManager lifecycle: spawn, ready-poll, evict, keep-alive, crash-reap.

Unlike the rest of the suite (which monkeypatches `ensure` to a stub), these tests
run a *fake `vllm`* executable so that `WorkerManager.ensure` genuinely spawns a
subprocess, polls `/health`, and later terminates / reaps it. The fake is a
self-contained stdlib server so no real vLLM (or GPU) is needed.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

import nvllm_service.workers as workers_mod
from nvllm_service.config import Settings
from nvllm_service.discovery import discover
from nvllm_service.workers import STATE_ERROR, STATE_READY, STATE_STOPPED, WorkerManager

# ---- fake `vllm` binaries ----------------------------------------------------

_FAKE_VLLM = """#!__PYTHON__
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _port(argv):
    if "--port" in argv:
        try:
            return int(argv[argv.index("--port") + 1])
        except (IndexError, ValueError):
            pass
    return 8000


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        self._json(
            200,
            {
                "id": "fake",
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "fake-ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", _port(sys.argv[1:])), H).serve_forever()
"""

_FAKE_VLLM_FAIL = """#!__PYTHON__
import sys

sys.exit(3)
"""


def _install_fake(tmp_path: Path, name: str, template: str) -> Path:
    exe = tmp_path / name
    exe.write_text(template.replace("__PYTHON__", sys.executable))
    os.chmod(exe, 0o755)
    return exe


# ---- fixtures -----------------------------------------------------------------


@pytest.fixture()
def fake_vllm(tmp_path: Path, monkeypatch) -> Path:
    exe = _install_fake(tmp_path, "fake_vllm", _FAKE_VLLM)
    monkeypatch.setattr(workers_mod, "_vllm_bin", lambda: str(exe))
    return exe


@pytest.fixture()
def failing_vllm(tmp_path: Path, monkeypatch) -> Path:
    exe = _install_fake(tmp_path, "fake_vllm_fail", _FAKE_VLLM_FAIL)
    monkeypatch.setattr(workers_mod, "_vllm_bin", lambda: str(exe))
    return exe


@pytest.fixture()
def manager(fake_hub: Path) -> WorkerManager:
    # short load_timeout so a broken worker fails the test fast, not in 30 min
    return WorkerManager(Settings(hf_hub_root=fake_hub, load_timeout=20))


def _two_models(fake_hub: Path):
    catalog = discover(fake_hub)
    a = next(m for m in catalog if m.repo == "acme/TinyLM")
    b = next(m for m in catalog if m.repo == "acme/Embedder")
    return a, b


def run_scenario(manager: WorkerManager, scenario) -> None:
    """Run an async scenario in one loop and always tear workers down after."""

    async def _wrap():
        try:
            await scenario()
        finally:
            await manager.stop_all()

    asyncio.run(_wrap())


# ---- tests ---------------------------------------------------------------------


def test_spawn_becomes_ready_and_unloads(manager, fake_vllm, fake_hub):
    info, _ = _two_models(fake_hub)

    async def scenario():
        w = await manager.ensure(info)
        assert w.state == STATE_READY
        assert w.pid
        assert any(d["name"] == info.name for d in await manager.loaded_models())
        assert await manager.unload(info.name) is True
        assert w.process.returncode is not None
        assert manager.get(info.name) is None

    run_scenario(manager, scenario)


def test_ensure_evicts_other_non_judge(manager, fake_vllm, fake_hub):
    a, b = _two_models(fake_hub)

    async def scenario():
        wa = await manager.ensure(a)
        wb = await manager.ensure(b)  # non-judge -> evicts the resident non-judge
        assert wa.state == STATE_STOPPED
        assert wa.process.returncode is not None
        assert wb.state == STATE_READY
        assert manager.get(a.name) is wa  # kept in registry, marked stopped

    run_scenario(manager, scenario)


def test_judge_is_never_evicted(manager, fake_vllm, fake_hub):
    a, b = _two_models(fake_hub)

    async def scenario():
        wjudge = await manager.ensure(a, is_judge=True)
        wb = await manager.ensure(b, is_judge=False)
        assert wjudge.state == STATE_READY
        assert wjudge.process.returncode is None  # still alive
        assert wb.state == STATE_READY
        assert manager.get(a.name) is wjudge

    run_scenario(manager, scenario)


def test_keep_alive_sets_expiry(manager, fake_vllm, fake_hub):
    info, _ = _two_models(fake_hub)

    async def scenario():
        w = await manager.ensure(info, keep_alive=30)
        now = time.time()
        assert w.expires_at is not None
        assert now < w.expires_at <= now + 31
        same = await manager.ensure(info, keep_alive=-1)  # -1 = keep forever
        assert same is w
        assert same.expires_at is None

    run_scenario(manager, scenario)


def test_monitor_reaps_crashed_worker(manager, fake_vllm, fake_hub):
    info, _ = _two_models(fake_hub)

    async def scenario():
        manager.start_monitor()
        try:
            w = await manager.ensure(info)
            assert w.state == STATE_READY
            w.process.kill()
            await w.process.wait()
            deadline = time.time() + 15
            while time.time() < deadline:
                if w.state == STATE_ERROR:
                    break
                await asyncio.sleep(0.5)
            assert w.state == STATE_ERROR
            assert w.error
        finally:
            if manager._monitor_task is not None:
                manager._monitor_task.cancel()

    run_scenario(manager, scenario)


def test_spawn_failure_raises_and_cleans(manager, failing_vllm, fake_hub):
    info, _ = _two_models(fake_hub)

    async def scenario():
        with pytest.raises(RuntimeError):
            await manager.ensure(info)
        assert manager.get(info.name) is None  # cleaned up, not left as zombie

    run_scenario(manager, scenario)
