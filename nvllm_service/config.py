"""Service configuration: CLI flags > environment variables > defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__ as _PKG_VERSION


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return int(v)


def _env_float(name: str, default: float | None = None) -> float | None:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return float(v)


def default_hf_hub_root() -> Path:
    """Default HuggingFace hub cache location (highest precedence first)."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(var)
        if v:
            return Path(v).expanduser()
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class Settings:
    # --- service ---
    host: str = field(default_factory=lambda: _env_str("NVLLM_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("NVLLM_PORT", 8004))

    # --- model discovery ---
    hf_hub_root: Path | None = field(default=None)  # None -> default_hf_hub_root()

    # --- scheduler / lifecycle ---
    # seconds a model stays loaded after its last request; -1 = keep forever, 0 = unload immediately
    default_keep_alive: int = field(default_factory=lambda: _env_int("NVLLM_KEEP_ALIVE", 300))
    # Max non-judge models resident at once (the judge is separate, always resident, never counted).
    # 1 = one model at a time (default; safe on shared/unified memory like DGX Spark). Raise it to
    # co-locate models, and lower worker_gpu_memory_utilization accordingly so they fit.
    max_models: int = field(default_factory=lambda: max(1, _env_int("NVLLM_MAX_MODELS", 1)))
    load_timeout: float = field(default_factory=lambda: _env_float("NVLLM_LOAD_TIMEOUT", 1800.0))
    # how often the monitor reaps crashed / expired workers (seconds)
    monitor_interval: float = field(default_factory=lambda: _env_float("NVLLM_MONITOR_INTERVAL", 2.0))

    # --- worker (vllm serve) ---
    worker_host: str = field(default_factory=lambda: _env_str("NVLLM_WORKER_HOST", "127.0.0.1"))
    worker_max_model_len: int | None = field(default_factory=lambda: _env_int("NVLLM_MAX_MODEL_LEN"))
    # Note: on DGX Spark (GB10) the OS/desktop uses a large share of the 128G
    # unified memory, so vLLM's default 0.92 fails at startup. 0.5 is a safe
    # default there; raise it on a dedicated machine.
    worker_gpu_memory_utilization: float | None = field(
        default_factory=lambda: _env_float("NVLLM_GPU_MEMORY_UTILIZATION", 0.5)
    )
    worker_max_num_seqs: int | None = field(default_factory=lambda: _env_int("NVLLM_MAX_NUM_SEQS"))
    # Cap on parallel kernel-JIT compile jobs (FlashInfer's run_ninja honors MAX_JOBS).
    # On shared/unified-memory boxes (DGX Spark) the default parallelism (~nproc)
    # OOM-kills nvcc mid-compile (SIGKILL / exit 137); 2-4 is safe there.
    # None = let FlashInfer/ninja choose (fine on a dedicated big-memory host).
    worker_max_jobs: int | None = field(default_factory=lambda: _env_int("NVLLM_WORKER_MAX_JOBS"))
    worker_trust_remote_code: bool = field(
        default_factory=lambda: os.environ.get("NVLLM_TRUST_REMOTE_CODE", "0") == "1"
    )
    # extra raw CLI args appended to every `vllm serve` invocation, e.g. NVLLM_WORKER_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
    worker_extra_args: list[str] = field(default_factory=list)

    # --- routing (Router) ---
    router_judge: str | None = field(default_factory=lambda: _env_str("NVLLM_ROUTER_JUDGE"))
    # Optional whitelist of candidate repos ("org/name") the Router may route to;
    # empty = all supported models. Keeps routing to models that actually fit in memory.
    router_candidates: list[str] = field(default_factory=list)

    # --- misc ---
    log_dir: Path | None = field(default=None)  # None -> <service_dir>/logs
    version: str = field(default_factory=lambda: _env_str("NVLLM_VERSION") or _PKG_VERSION)

    def hub_root(self) -> Path:
        return self.hf_hub_root or default_hf_hub_root()

    def logs_dir(self) -> Path:
        if self.log_dir:
            return Path(self.log_dir)
        return Path.home() / ".local" / "share" / "nvllm-serve" / "logs"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        extra = os.environ.get("NVLLM_WORKER_EXTRA_ARGS", "")
        if extra:
            import shlex

            s.worker_extra_args = shlex.split(extra)
        cands = os.environ.get("NVLLM_ROUTER_CANDIDATES", "")
        if cands:
            s.router_candidates = [c.strip() for c in cands.split(",") if c.strip()]
        return s
