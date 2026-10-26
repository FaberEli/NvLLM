"""Command-line entry.

``nvllm`` (console script) and ``python -m nvllm_service`` are the same
Ollama-style subcommand CLI: serve / list / run / ps / stop / show / launch /
version. No arguments prints help.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

from . import __version__, client, launch
from .config import Settings
from .main import create_app


def _preflight_vllm() -> None:
    """Fail fast, with actionable guidance, if the vLLM engine binary is missing.

    Without this, the absence of vLLM would only surface on the first request,
    deep inside worker spawn (a confusing 'worker exited' error). Reuses the
    same resolution the worker manager uses, so what we check is what we'd run.
    """
    from .workers import _vllm_bin

    bin_path = _vllm_bin()
    if not os.path.isabs(bin_path) and shutil.which(bin_path) is None:
        sys.exit(
            "nvllm: the vLLM engine was not found on this system.\n"
            "  Install it with the one-click installer (from this repo):\n"
            "      scripts/install.sh\n"
            "  or, in the same Python environment as nvllm:\n"
            "      pip install 'vllm==0.31.0'   # validated baseline; see README\n"
        )


def run_server(argv: list[str] | None = None, prog: str = "nvllm serve") -> int:
    """Start the service (the whole body of the original `nvllm-serve` main)."""
    p = argparse.ArgumentParser(
        prog=prog,
        description="Ollama-style resident inference service backed by vLLM.",
    )
    p.add_argument("--version", action="version", version=f"{prog} {__version__}")
    p.add_argument("--host", default=None, help="bind host (default $NVLLM_HOST or 0.0.0.0)")
    p.add_argument("--port", type=int, default=None, help="bind port (default $NVLLM_PORT or 8004)")
    p.add_argument("--hf-hub-root", default=None, help="HuggingFace hub cache directory to scan")
    p.add_argument("--keep-alive", type=int, default=None, help="default idle seconds before unload (-1 keep, 0 immediate)")
    p.add_argument("--max-models", type=int, default=None, help="max non-judge models resident at once (default $NVLLM_MAX_MODELS or 1)")
    p.add_argument("--max-model-len", type=int, default=None, help="pass --max-model-len to workers")
    p.add_argument("--gpu-memory-utilization", type=float, default=None, help="pass to workers")
    p.add_argument("--router-judge", default=None, help="model name to act as the Router judge (a decision model, or any LLM)")
    p.add_argument("--router-candidates", default=None, help="comma-separated repos to restrict routing candidates (default: all supported)")
    p.add_argument("--router-backend", default=None, help="router judge backend: vllm (default, serve+logprob scoring) or transformers (in-process)")
    p.add_argument("--router-top-logprobs", type=int, default=None, help="top-k logprobs to request from the judge for option scoring (default 32)")
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
    if args.router_backend:
        s.router_backend = args.router_backend
    if args.router_top_logprobs is not None:
        s.router_top_logprobs = max(1, args.router_top_logprobs)

    _preflight_vllm()
    app = create_app(s)

    import uvicorn

    uvicorn.run(app, host=s.host, port=s.port, log_level=args.log_level)
    return 0


# Ollama muscle-memory commands we deliberately do NOT implement: NvLLM neither
# downloads nor deletes models — they live in the local HuggingFace cache.
# (`create`/`rm` ARE implemented now, for *virtual* models over cache models.)
_REFUSALS: dict[str, str] = {
    "pull": (
        "nvllm does not download models.\n"
        "\n"
        "Models are discovered from your local HuggingFace cache — add one there\n"
        "and it will appear in `nvllm list`, e.g.:\n"
        "\n"
        "    pip install -U 'huggingface_hub[cli]'\n"
        "    hf download <org>/<name>\n"
    ),
    "push": (
        "nvllm has no model registry. To publish a model, use the HuggingFace\n"
        "Hub CLI:  hf upload <local-dir> <org>/<name>\n"
    ),
    "login": (
        "nvllm has no accounts — it serves models from your local HuggingFace\n"
        "cache, not a registry. (For Hub auth use `hf auth login`.)\n"
    ),
    "cp": (
        "nvllm does not copy models — they live in your HuggingFace cache.\n"
        "Copy the `models--<org>--<name>` directory if you need a second copy.\n"
    ),
}

_TOP_HELP = f"""\
NvLLM — Ollama-style CLI for models in your local HuggingFace cache
(backend: vLLM workers managed by the nvllm server)

Usage:
  nvllm serve [flags]         start the server (see `nvllm serve --help`)
  nvllm list [PREFIX]         list models found in the HF cache
  nvllm run MODEL [PROMPT]    chat with a model — one-shot, or an interactive REPL
  nvllm create NAME --from MODEL [--system "..."]   virtual model over a cache model
  nvllm rm NAME               delete a created (virtual) model (cache models are never deleted)
  nvllm router on|off|status  activate/deactivate auto-routing (resident decision-model judge -> AutoRouter)
  nvllm route TEXT            preview which model 'auto' would route to (dry-run; runs only the judge)
  nvllm launch [AGENT] [EXTRA]  start an agent CLI (claude/codex/…) on the local service
  nvllm ps                    show loaded (resident) models
  nvllm stop MODEL            unload a model
  nvllm show MODEL            show model details
  nvllm version               show CLI and server versions
  nvllm help                  show this help

Models: nvllm does NOT download models. It serves what is already in your local
HuggingFace cache (default: ~/.cache/huggingface/hub). To add a model:
    pip install -U 'huggingface_hub[cli]'
    hf download <org>/<name>
…then it shows up in `nvllm list`.

Launching agents: `nvllm launch claude -p "hello"` runs Claude Code against the
local models (Anthropic API served at /v1/messages); codex/qwen/gemini/opencode/
aider go through the OpenAI API at /v1. See `nvllm launch --list`.

Routing (opt-in): `nvllm router on` scans the cache for "System One" decision
models, lets you pick the judge (optionally asking a running local model to
recommend one), and persists the choice — on the next `nvllm serve` the judge is
resident and a new model option "AutoRouter" appears; send model "AutoRouter" (or
"auto") and the judge picks the best local model for the request. `nvllm router
status` / `off` show/clear it. (A judge can also be set per-run with
`--router-judge`; the default backend is vLLM + logprob scoring.) Preview a
decision without consuming a backend model: `nvllm route "your text"`.

Environment: NVLLM_SERVER — server URL (default {client.DEFAULT_URL})
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nvllm",
        description="Ollama-style CLI for the nvllm server (vLLM models from your HF cache).",
    )
    p.add_argument("--version", action="version", version=f"nvllm {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    l = sub.add_parser("list", aliases=["ls"], help="list models in the HF cache")
    l.add_argument("prefix", nargs="?", default=None, help="only show names starting with this")
    l.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    ps = sub.add_parser("ps", help="list loaded models")
    ps.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    sh = sub.add_parser("show", help="show model details")
    sh.add_argument("model")
    sh.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    r = sub.add_parser("run", help="chat with a model (one-shot prompt or interactive REPL)")
    r.add_argument("model")
    r.add_argument("prompt", nargs="*", default=[], help="one-shot prompt (omit for the interactive REPL)")
    r.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")
    r.add_argument("-v", "--verbose", action="store_true", help="print throughput stats after the reply")
    r.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    r.add_argument("--top-p", dest="top_p", type=float, default=None, help="nucleus sampling")
    r.add_argument("--num-predict", type=int, default=None, help="max tokens per reply")
    r.add_argument("--image", action="append", default=[], metavar="FILE", help="attach an image (repeatable; one-shot only)")
    r.add_argument("--tools", default=None, metavar="FILE", help="OpenAI-style tools JSON file (one-shot only; REPL: /tools FILE)")

    cr = sub.add_parser("create", help="create a virtual model over a cache model (system prompt + params)")
    cr.add_argument("name", help="short name for the new model (e.g. 'assistant')")
    cr.add_argument("--from", dest="base", required=True, metavar="MODEL", help="base catalog model to wrap")
    cr.add_argument("--system", default="", help="system prompt")
    cr.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    cr.add_argument("--top-p", type=float, default=None, help="nucleus sampling")
    cr.add_argument("--num-predict", type=int, default=None, help="max tokens per reply")
    cr.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    rm = sub.add_parser("rm", help="delete a created (virtual) model — cache models are never deleted")
    rm.add_argument("name")
    rm.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    rt = sub.add_parser("route", help="preview which model 'auto' would route to for a text (dry-run; runs only the judge)")
    rt.add_argument("text", help="the request text to route")
    rt.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    rtr = sub.add_parser("router", help="activate / deactivate / show the auto-routing (a resident decision-model judge)")
    rtr.add_argument("action", choices=["on", "off", "status"], help="on = activate (interactive judge pick), off = deactivate, status = show")
    rtr.add_argument("--judge", default=None, metavar="MODEL", help="for `on`: skip the interactive prompt and use this decision model directly")
    rtr.add_argument("--candidates", default=None, metavar="REPOS", help="for `on`: comma-separated repos to restrict routing to (default: all servable non-decision models)")
    rtr.add_argument("--backend", default=None, choices=["vllm", "transformers"], help="for `on`: judge backend (default vllm)")
    rtr.add_argument("--top-logprobs", dest="top_logprobs", type=int, default=None, help="for `on`: option-logprob depth (default 32)")
    rtr.add_argument("--no-recommend", action="store_true", help="for `on`: skip the 'let a local model recommend' prompt")
    rtr.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    la = sub.add_parser("launch", help="launch an agent CLI (claude/codex/…) wired to the nvllm server")
    la.add_argument("--model", "-m", default=None, help="model to use (default: largest completion-capable model, or $NVLLM_LAUNCH_MODEL)")
    la.add_argument("--list", action="store_true", help="show supported agents and their availability")
    la.add_argument("agent", nargs="?", default=None, help="agent name (default: first available on PATH)")
    la.add_argument("extra", nargs=argparse.REMAINDER, help="arguments passed through to the agent (nvllm flags go BEFORE the agent name)")
    la.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    st = sub.add_parser("stop", help="unload a loaded model")
    st.add_argument("model")
    st.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    v = sub.add_parser("version", help="show CLI and server versions")
    v.add_argument("--server", default=None, help="server URL (default $NVLLM_SERVER or http://127.0.0.1:8004)")

    return p


def main(argv: list[str] | None = None) -> int:
    """``nvllm`` console script (and ``python -m nvllm_service``). No args → help."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(_TOP_HELP)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "serve":
        return run_server(rest, prog="nvllm serve")
    if cmd in ("help", "--help", "-h"):
        print(_TOP_HELP)
        return 0
    if cmd in _REFUSALS:
        print(_REFUSALS[cmd])
        return 1

    p = _build_parser()
    # parse the full argv (command name included): the subparsers need it
    args = p.parse_args(argv)
    url = client.server_url(args.server)
    try:
        if args.command in ("list", "ls"):
            return client.cmd_list(url, args.prefix)
        if args.command == "ps":
            return client.cmd_ps(url)
        if args.command == "show":
            return client.cmd_show(url, args.model)
        if args.command == "run":
            options: dict = {}
            if args.temperature is not None:
                options["temperature"] = args.temperature
            if args.top_p is not None:
                options["top_p"] = args.top_p
            if args.num_predict is not None:
                options["num_predict"] = args.num_predict
            tools = client.load_tools_file(args.tools) if args.tools else None
            prompt = " ".join(args.prompt).strip()
            if prompt:
                return client.run_prompt(
                    url, args.model, prompt, options, verbose=args.verbose,
                    images=args.image or None, tools=tools,
                )
            return client.repl(url, args.model, options, tools=tools)
        if args.command == "create":
            options = {}
            if args.temperature is not None:
                options["temperature"] = args.temperature
            if args.top_p is not None:
                options["top_p"] = args.top_p
            if args.num_predict is not None:
                options["num_predict"] = args.num_predict
            return client.cmd_create(url, args.name, args.base, args.system, options)
        if args.command == "rm":
            return client.cmd_rm(url, args.name)
        if args.command == "route":
            return client.cmd_route(url, args.text)
        if args.command == "router":
            candidates = [c.strip() for c in (args.candidates or "").split(",") if c.strip()]
            return client.cmd_router(
                url,
                args.action,
                judge=args.judge,
                candidates=candidates or None,
                backend=args.backend,
                top_logprobs=args.top_logprobs,
                recommend=not args.no_recommend,
            )
        if args.command == "launch":
            extra = list(args.extra or [])
            if extra and extra[0] == "--":  # argparse's REMAINDER separator
                extra = extra[1:]
            return launch.cmd_launch(url, args.agent, args.model, extra, list_only=args.list)
        if args.command == "stop":
            return client.cmd_stop(url, args.model)
        if args.command == "version":
            return client.cmd_version(url)
        p.print_help()
        return 2
    except client.CLIError as e:
        print(f"nvllm: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
