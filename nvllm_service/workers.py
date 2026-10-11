"""Worker process management.

Each loaded model runs in its own `vllm serve` subprocess (one vLLM engine per
process is a vLLM constraint; this also matches Ollama's runner-per-model design
and isolates crashes). The supervisor loads a model by spawning its worker and
unloads it by terminating the process.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx

from .catalog import ModelInfo
from .config import Settings

log = logging.getLogger(__name__)

STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_STOPPED = "stopped"
STATE_ERROR = "error"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _venv_bin_dir() -> str:
    """The ``bin`` directory of the running interpreter (where ``ninja`` lives)."""
    return os.path.dirname(os.path.abspath(sys.executable))


def _vllm_bin() -> str:
    """The `vllm` executable next to the running interpreter."""
    cand = os.path.join(_venv_bin_dir(), "vllm")
    if os.path.exists(cand):
        return cand
    return "vllm"  # fall back to PATH


def _worker_env(settings: Settings) -> dict[str, str]:
    """Worker process environment.

    Prepend the venv's ``bin`` dir to ``PATH`` so the worker finds tooling
    installed in the venv (notably ``ninja``) even when the service was started
    without activating the venv (e.g. ``.venv/bin/python -m nvllm_service``).
    torch's inductor JIT compiler shells out to ``ninja`` and aborts engine
    init with ``FileNotFoundError: 'ninja'`` if it is not on ``PATH``.

    Also set ``MAX_JOBS`` (honored by FlashInfer's ``run_ninja``) when
    ``worker_max_jobs`` is configured, to cap parallel CUTLASS kernel JIT
    compiles — the default (~nproc) parallelism OOM-kills nvcc on the shared
    unified memory of a DGX Spark (exit 137 / SIGKILL).
    """
    env = dict(os.environ)
    bin_dir = _venv_bin_dir()
    path = env.get("PATH", os.defpath)
    if bin_dir not in path.split(os.pathsep):
        env["PATH"] = bin_dir + os.pathsep + path
    if settings.worker_max_jobs and settings.worker_max_jobs > 0:
        env["MAX_JOBS"] = str(settings.worker_max_jobs)
    return env


@dataclass
class Worker:
    info: ModelInfo
    port: int
    is_judge: bool = False
    state: str = STATE_LOADING
    error: str = ""
    pid: int | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    expires_at: float | None = None  # epoch seconds; None = keep forever
    started_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)  # epoch; LRU basis, refreshed on every ensure() hit
    inflight: int = 0  # in-flight requests; the monitor will not reap a worker while this is > 0
    log_path: str = ""

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def to_ps(self) -> dict:
        d = {
            "name": self.name,
            "model": self.name,
            "size": self.info.size,
            "digest": self.info.commit,
            "details": self.info._details(),
            "size_vram": 0,
            "context_length": self.info.context_length,
            "state": self.state,
            "pid": self.pid,
            "last_used": self.last_used,
        }
        if self.expires_at:
            from datetime import datetime, timezone

            d["expires_at"] = datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()
        else:
            d["expires_at"] = "persistent"
        return d


class WorkerManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.workers: dict[str, Worker] = {}
        self._lock = asyncio.Lock()
        self._monitor_task: asyncio.Task | None = None

    # ---- lifecycle ------------------------------------------------------
    def start_monitor(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.get_running_loop().create_task(self._monitor())

    async def _reap_once(self) -> None:
        """One pass: (1) mark crashed/OOM-killed workers as errored, and
        (2) unload idle workers whose keep-alive has elapsed.

        Runs under a single lock acquisition (so we never re-enter the lock via
        ``unload``), and skips the judge and any worker with an in-flight request
        (inflight > 0) so a keep-alive of 0 cannot kill a worker mid-request.
        """
        async with self._lock:
            now = time.time()
            for w in list(self.workers.values()):
                # (1) process died (crash or OOM) -> mark stopped/error
                if (
                    w.state in (STATE_LOADING, STATE_READY)
                    and w.process
                    and w.process.returncode is not None
                ):
                    rc = w.process.returncode
                    log.error("worker %s exited unexpectedly (rc=%s); marking stopped", w.name, rc)
                    w.state = STATE_ERROR
                    w.error = f"worker process exited with code {rc}"
                    # surface the tail of the worker log for debugging
                    try:
                        with open(w.log_path, "r", encoding="utf-8", errors="replace") as f:
                            tail = f.readlines()[-15:]
                        log.error("worker %s log tail:\n%s", w.name, "".join(tail))
                    except OSError:
                        pass
                    continue
                # (2) idle beyond keep-alive -> unload (never the judge, never mid-request)
                if (
                    not w.is_judge
                    and w.state in (STATE_LOADING, STATE_READY)
                    and w.inflight == 0
                    and w.expires_at is not None
                    and w.expires_at <= now
                ):
                    log.info("worker %s idle beyond keep-alive; unloading", w.name)
                    await self._terminate(w)
                    self.workers.pop(w.name, None)

    async def _monitor(self) -> None:
        """Run a reap pass every ``monitor_interval`` seconds."""
        while True:
            await asyncio.sleep(self.settings.monitor_interval)
            await self._reap_once()

    def get(self, name: str) -> Worker | None:
        return self.workers.get(name)

    @asynccontextmanager
    async def track(self, worker: Worker):
        """Hold an in-flight reference for the duration of a request.

        The monitor will not reap a worker that is serving a request (inflight > 0),
        so a keep_alive of 0 ("unload immediately") cannot kill a worker mid-request.
        """
        worker.inflight += 1
        try:
            yield
        finally:
            worker.inflight = max(0, worker.inflight - 1)

    async def ensure(
        self,
        info: ModelInfo,
        keep_alive: int | None = None,
        is_judge: bool = False,
    ) -> Worker:
        """Return a READY worker for `info`, spawning one if needed.

        Up to ``settings.max_models`` non-judge models may be resident at once
        (the judge is separate: always resident, never counted, never evicted). If
        adding the target would exceed the limit, the least-valued residents are
        evicted first — expired workers (past their keep-alive) before still-valid
        ones, and within each group the least-recently-used. The target itself is
        never evicted. Spawning happens under the lock, so at most one model loads
        at a time (safe on shared/unified memory where two concurrent vLLM loads
        could OOM), while already-ready models serve concurrently.
        """
        async with self._lock:
            existing = self.workers.get(info.name)
            if existing and existing.state == STATE_READY:
                self._apply_keep_alive(existing, keep_alive)
                existing.last_used = time.time()
                return existing

            # judge never evicts, is never evicted
            if not is_judge:
                now = time.time()
                active = [
                    w
                    for w in self.workers.values()
                    if not w.is_judge
                    and w.state in (STATE_LOADING, STATE_READY)
                    and w.name != info.name
                ]
                need = len(active) + 1 - self.settings.max_models
                if need > 0:
                    victims = sorted(
                        active,
                        key=lambda w: (
                            0 if (w.expires_at is not None and w.expires_at <= now) else 1,
                            w.last_used,
                        ),
                    )[:need]
                    for v in victims:
                        log.info("evicting %s to make room for %s", v.name, info.name)
                        await self._terminate(v)

            return await self._spawn(info, keep_alive, is_judge=is_judge)

    async def _spawn(self, info: ModelInfo, keep_alive: int | None, is_judge: bool) -> Worker:
        s = self.settings
        port = _free_port()
        worker = Worker(info=info, port=port, is_judge=is_judge)

        cmd = [
            _vllm_bin(),
            "serve",
            info.snapshot_dir,
            "--host",
            s.worker_host,
            "--port",
            str(port),
            "--served-model-name",
            info.name,
        ]
        if s.worker_max_model_len:
            cmd += ["--max-model-len", str(s.worker_max_model_len)]
        if s.worker_gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(s.worker_gpu_memory_utilization)]
        if s.worker_max_num_seqs:
            cmd += ["--max-num-seqs", str(s.worker_max_num_seqs)]
        if s.worker_trust_remote_code:
            cmd += ["--trust-remote-code"]
        cmd += s.worker_extra_args

        logs_dir = s.logs_dir()
        os.makedirs(logs_dir, exist_ok=True)
        safe = info.name.replace("/", "__")
        worker.log_path = str(logs_dir / f"{safe}.log")
        logf = open(worker.log_path, "ab")

        log.info("spawning worker: %s", " ".join(cmd))
        worker.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=logf,
            stderr=logf,
            cwd=str(logs_dir),
            env=_worker_env(s),
            start_new_session=True,
        )
        worker.pid = worker.process.pid
        self.workers[info.name] = worker

        try:
            await self._wait_ready(worker)
        except Exception as e:
            log.error("worker %s failed to become ready: %s (log: %s)", info.name, e, worker.log_path)
            worker.state = STATE_ERROR
            worker.error = str(e)
            await self._terminate(worker)
            self.workers.pop(info.name, None)
            raise

        self._apply_keep_alive(worker, keep_alive)
        return worker

    async def _wait_ready(self, worker: Worker, timeout: float | None = None) -> None:
        s = self.settings
        deadline = time.time() + (timeout or s.load_timeout)
        url = f"http://127.0.0.1:{worker.port}/health"
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            while time.time() < deadline:
                if worker.process and worker.process.returncode is not None:
                    raise RuntimeError(
                        f"worker exited during startup (rc={worker.process.returncode}); see {worker.log_path}"
                    )
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        worker.state = STATE_READY
                        log.info("worker %s ready on port %d (pid %s)", worker.name, worker.port, worker.pid)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.5)
        raise TimeoutError(f"worker {worker.name} not ready within {deadline:.0f}s; see {worker.log_path}")

    async def unload(self, name: str) -> bool:
        """Terminate a worker and drop it from the registry."""
        async with self._lock:
            w = self.workers.get(name)
            if w is None:
                return False
            await self._terminate(w)
            self.workers.pop(name, None)
            return True

    async def _terminate(self, worker: Worker) -> None:
        p = worker.process
        if p is None or p.returncode is not None:
            worker.state = STATE_STOPPED
            return
        log.info("terminating worker %s (pid %s)", worker.name, worker.pid)
        try:
            p.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(p.wait(), timeout=30)
        except asyncio.TimeoutError:
            log.warning("worker %s did not exit in 30s; killing", worker.name)
            try:
                p.kill()
            except ProcessLookupError:
                pass
            await p.wait()
        worker.state = STATE_STOPPED

    async def stop_all(self) -> None:
        for name in list(self.workers):
            await self.unload(name)

    def _apply_keep_alive(self, worker: Worker, keep_alive: int | None) -> None:
        if worker.is_judge:
            worker.expires_at = None
            return
        ka = self.settings.default_keep_alive if keep_alive is None else keep_alive
        if ka < 0:
            worker.expires_at = None
        elif ka == 0:
            worker.expires_at = time.time()  # expires immediately after the request
        else:
            worker.expires_at = time.time() + ka

    async def loaded_models(self) -> list[dict]:
        out = []
        for w in self.workers.values():
            if w.state in (STATE_LOADING, STATE_READY):
                out.append(w.to_ps())
        out.sort(key=lambda d: d["name"])
        return out
