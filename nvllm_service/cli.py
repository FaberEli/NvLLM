"""Command-line entry: `nvllm-serve` / `python -m nvllm_service`."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .main import create_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nvllm-serve",
        description="Ollama-style resident inference service backed by vLLM.",
    )
    p.add_argument("--host", default=None, help="bind host (default $NVLLM_HOST or 0.0.0.0)")
    p.add_argument("--port", type=int, default=None, help="bind port (default $NVLLM_PORT or 8004)")
    p.add_argument("--hf-hub-root", default=None, help="HuggingFace hub cache directory to scan")
    p.add_argument("--keep-alive", type=int, default=None, help="default idle seconds before unload (-1 keep, 0 immediate)")
    p.add_argument("--max-models", type=int, default=None, help="max non-judge models resident at once (default $NVLLM_MAX_MODELS or 1)")
    p.add_argument("--max-model-len", type=int, default=None, help="pass --max-model-len to workers")
    p.add_argument("--gpu-memory-utilization", type=float, default=None, help="pass to workers")
    p.add_argument("--router-judge", default=None, help="model name to act as the Router judge")
    p.add_argument("--router-candidates", default=None, help="comma-separated repos to restrict routing candidates (default: all supported)")
    p.add_argument("--log-level", default="info", help="log level")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    s = Settings.from_env()
    if args.host:
        s.host = args.host
    if args.port:
        s.port = args.port
    if args.hf_hub_root:
        s.hf_hub_root = args.hf_hub_root
    if args.keep_alive is not None:
        s.default_keep_alive = args.keep_alive
    if args.max_models is not None:
        s.max_models = max(1, args.max_models)  # guard against 0 / negative
    if args.max_model_len is not None:
        s.worker_max_model_len = args.max_model_len
    if args.gpu_memory_utilization is not None:
        s.worker_gpu_memory_utilization = args.gpu_memory_utilization
    if args.router_judge:
        s.router_judge = args.router_judge
    if args.router_candidates:
        s.router_candidates = [c.strip() for c in args.router_candidates.split(",") if c.strip()]

    app = create_app(s)

    import uvicorn

    uvicorn.run(app, host=s.host, port=s.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
