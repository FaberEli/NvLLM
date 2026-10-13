# CLAUDE.md — nvllm-serve

Ollama-style resident inference service backed by vLLM. `README.md` is the
user-facing doc; this file is for working on the code.

## Running

- Tests (no GPU, no real vLLM needed): `.venv/bin/python -m pytest -q`
- Dev server (default bind `0.0.0.0:8004`): `.venv/bin/python -m nvllm_service --hf-hub-root <cache>`
- Entry point: `nvllm_service.cli:main` (console script `nvllm-serve`).
- Production install (for users): `scripts/install.sh` — venv + `vllm==0.31.0` (pinned
  validated baseline) into `~/.nvllm/`, symlink in `~/.local/bin`, optional systemd
  user service (`--with-service`). `scripts/update.sh` upgrades (vLLM stays pinned
  unless `--upgrade-vllm`); `scripts/uninstall.sh` restores the environment
  (models/caches kept unless `--purge`). All honor `NVLLM_HOME` to relocate the root.

## Layout

- `discovery.py` — scan HF hub cache → `list[ModelInfo]`; `by_name()` resolves
  `org/name[:tag]` and bare names. Tag-less means `'latest'` and prefers a
  `main`/`latest` ref (HF caches usually only have `main`).
- `catalog.py` — `ModelInfo`, `Capability` (completion / embedding / tools).
- `config.py` — `Settings` dataclass. `hf_hub_root` is a **field** (`Path|None`);
  `hub_root()` is the **method** that applies the default.
  ⚠️ **Call `hub_root()`, not `hf_hub_root()`** — the field is a `Path` (or
  `None`), so calling it raises `TypeError`. This is a known footgun that
  regressed once.
- `workers.py` — `WorkerManager`: one `vllm serve` subprocess per model.
  `_vllm_bin()` resolves the worker binary (next to the running interpreter, else
  `PATH`). Up to `settings.max_models` non-judge models resident at once (default
  `1`); on `ensure()` the least-valued residents are evicted to make room —
  expired (past `keep-alive`) first, then least-recently-used; the Router judge is
  never counted, never evicted, and never evicts. `Worker.inflight` tracks
  in-flight requests; `track(worker)` (an async context manager) is how `main.py`
  holds one during the request, and the **streaming** forwards in `forward.py`
  hold it themselves for the stream's lifetime (a streaming body is delivered
  *after* the endpoint returns, outside `track()`'s window — so a `keep-alive=0`
  worker is never reaped mid-stream). The monitor (`_reap_once()`, every
  `monitor_interval` s) marks crashed workers errored and
  unloads idle workers past their keep-alive (skipping the judge and any worker
  with `inflight > 0`). Loading is serialized (the `ensure` lock spans spawn+
  `_wait_ready`), so at most one model loads at a time; loaded models serve
  concurrently.
- `main.py` — FastAPI assembly; `create_app(settings)` is what tests target.
  `_resolve_model(body)` resolves the request's model — the `"auto"` alias
  triggers the Router (if `--router-judge` is set); otherwise it's `find_model`.
  Endpoints then `ensure()` a worker and delegate to `forward`.
- `forward.py` — Ollama↔OpenAI translation + streaming (OpenAI SSE in, Ollama
  NDJSON out). Keep it worker-agnostic: it only needs a `Worker` exposing
  `base_url`, `name`, and `inflight` (streaming forwards hold `inflight` for the
  stream's lifetime so a `keep-alive=0` worker can't be reaped mid-stream — see
  `tests/test_forward_stream.py`).
- `router.py` — `Router`: the judge model picks a target from the candidates.
  `pick()` calls the judge via `forward.forward_json`, `parse_choice()` resolves
  its free-text reply (verbatim `org/name` first, bare name as a whole token),
  `fallback()` is the largest candidate. `ROUTE_ALIAS` = `"auto"`.
- `cli.py` — argparse entry. Keep it in sync with `config.py` when adding flags.

## Conventions / gotchas

- Most tests monkeypatch `WorkerManager.ensure` to a stub server
  (`tests/conftest.py`). Real subprocess spawning is covered separately by
  `tests/test_workers.py`, which fakes the `vllm` binary so `ensure` actually
  spawns a process.
- Reaping is tested **deterministically** by calling `WorkerManager._reap_once()`
  directly (no sleeps): set `worker.expires_at` to the past to force expiry, and
  toggle `worker.inflight` to exercise the mid-request protection.
- `Settings` precedence is **CLI > env > default**; add new options to both
  `config.py` and `cli.py`. (Internal knobs like `monitor_interval` are env-only,
  matching `worker_max_num_seqs`/`worker_trust_remote_code` — no CLI flag needed.)
- Workers bind `127.0.0.1` and speak OpenAI natively; the proxy rewrites only
  the `model` field on `/v1/*` passthrough.
- Streaming helpers are async so connect-failures / worker 4xx surface as proper
  HTTP errors *before* the response stream starts.
- **Worker process env (`_worker_env`)**: the venv's `bin` dir is prepended to
  `PATH` so a worker finds `ninja` even when the service is started via
  `.venv/bin/python -m nvllm_service` (no activation) — torch's inductor otherwise
  dies with `FileNotFoundError: 'ninja'`. `worker_max_jobs` sets `MAX_JOBS`, which
  FlashInfer's `run_ninja` honors to cap parallel CUTLASS kernel JIT; the default
  (~nproc) parallelism OOM-kills `nvcc` on the Spark's shared 128G (exit 137).
- **vLLM version drift**: the venv pins vLLM 0.31, which *removed*
  `--disable-log-requests` and groups CLI help into "Config Groups" (probe a flag
  with `vllm serve --help=<flag>`). Don't re-add removed flags to the worker cmd —
  a worker that exits `rc=1` during startup is usually an unrecognized argument.
- **Worker log dir**: `NVLLM_LOG_DIR` (env-only, read by `Settings.from_env()`;
  default `~/.local/share/nvllm-serve/logs`). `install.sh` / the systemd unit set it
  to `<install_root>/logs` so an uninstall removes logs with the rest.
- **Startup preflight**: `cli.main` exits early with actionable guidance when the
  `vllm` binary cannot be resolved (`_preflight_vllm`, reuses `workers._vllm_bin()`;
  tests build the app directly and never hit it).
- **CI**: `.github/workflows/ci.yml` runs the suite on ubuntu / Python 3.12 — keep
  the test suite GPU-free and vLLM-free (stub worker + fake hub cache).
