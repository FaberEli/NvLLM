# CLAUDE.md — nvllm-serve

Ollama-style resident inference service backed by vLLM. `README.md` is the
user-facing doc; this file is for working on the code.

## Running

- Tests (no GPU, no real vLLM needed): `.venv/bin/python -m pytest -q` (143 tests)
- Dev server (default bind `0.0.0.0:8004`): `nvllm serve` or
  `.venv/bin/python -m nvllm_service serve --hf-hub-root <cache>`
- **Single entry (0.3.0, breaking):** `nvllm` (= `nvllm_service.cli:main`) and
  `python -m nvllm_service` are the same Ollama-style CLI:
  `serve`/`list`/`run`/`ps`/`stop`/`show`/`launch`/`version`; no args → help.
  The legacy `nvllm-serve` binary was **removed on purpose** (early-stage project;
  no compat layer) — `install.sh` retires old symlinks/units.
- CLI client commands talk to a running server (default `http://127.0.0.1:8004`,
  `NVLLM_SERVER` env / `--server` flag). They must NOT require vLLM locally and
  must NOT call `_preflight_vllm`.
- Production install (for users): `scripts/install.sh` — venv + `vllm==0.31.0` (pinned
  validated baseline) into `~/.nvllm/`, symlink in `~/.local/bin`, optional systemd
  user service (`--with-service`). `scripts/update.sh` upgrades (vLLM stays pinned
  unless `--upgrade-vllm`); `scripts/uninstall.sh` restores the environment
  (models/caches kept unless `--purge`). All honor `NVLLM_HOME` to relocate the root.
- **Self-contained release (0.4.0):** `scripts/build-release.sh` builds
  `dist/nvllm-<ver>-linux-aarch64/` (install.sh + prebuilt `venv/` with vLLM +
  non-editable nvllm + LICENSE/README) → tarball. `scripts/bootstrap.sh` is the
  `curl|bash` front-end (download → extract → bundled install.sh). install.sh
  auto-detects a sibling `venv/` (the staging layout) and **skips pip** — keep
  that layout (install.sh next to venv/) if you change the build script.

## Layout

- `discovery.py` — scan HF hub cache → `list[ModelInfo]`; `by_name()` resolves
  `org/name[:tag]` and bare names. Tag-less means `'latest'` and prefers a
  `main`/`latest` ref (HF caches usually only have `main`).
- `catalog.py` — `ModelInfo`, `Capability` (completion / embedding / tools).
- `config.py` — `Settings` dataclass. `hf_hub_root` is a **field** (`Path|None`);
  `hub_root()` is the **method** that applies the default.
  ⚠️ **Call `hub_root()`, not `hf_hub_root()`** — the field is a `Path` (or
  `None`), so calling it raises `TypeError`. This is a known footgun that
  regressed once. Same pattern again: `state_root` (field) + `state_dir()`
  (method; env `NVLLM_STATE_DIR`, default `~/.local/share/nvllm-serve`) — the
  field *cannot* share the method's name, so it is deliberately `state_root`.
  Every env-backed knob is parsed in its `default_factory` — `from_env()` is a
  documented alias for `cls()`, so there is no second place to wire a new env var.
- `created.py` — `CreatedModel` (name/base/system/options/created_at) +
  `CreatedStore`: the virtual-model store (`created.json`, atomic tmp+rename
  write; `list/get/names/create/delete`). Pure stdlib, independently testable.
  A created model is **metadata only** — it resolves to the base catalog model
  (shares its worker); the wrapper is injected per request in `main.py`.
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
  **CORS**: `CORSMiddleware` allows any origin (browser clients + the `index.html`
  demo panel; the service is an unauthenticated LAN tool — see README "Security").
  `_resolve_model(body)` resolves the request's model — **registered aliases**
  (`app.state.model_aliases`, set via `POST /v1/aliases`) are mapped back to the
  real catalog model first, then the `"auto"` alias triggers the Router (if
  `--router-judge` is set); otherwise it's `find_model`. All inference endpoints
  share this one resolution path. Endpoints then `ensure()` a worker and delegate
  to `forward`.
  **Virtual models (0.4.0):** `app.state.created = CreatedStore(...)`;
  `_resolve_with_spec(body)` = alias/auto resolution + created-model check
  (virtual name → rewrite `body["model"]` to the base; returns the
  `CreatedModel` or `None`). `/api/chat`, `/v1/chat/completions`,
  `/v1/messages` then `_inject_*_spec()` (system message/field + option merge,
  request options win) and echo the **requested** name in the response.
  `POST /api/create` / `POST /api/rm` manage the store (rm only for created
  models; anything else 404 — the CLI turns that into HF-cache-delete guidance).
  `/api/tags` + `/v1/models` + `/api/show` list created models
  (`details.parent_model` = base). Spec injection is shared (`_apply_system_to_messages`
  + `_apply_sampling` under the three `_inject_*_spec()`); `/api/unload` on a
  created name resolves to its base (virtuals share the base's worker);
  `/v1/messages` enforces the COMPLETION capability like `/api/chat`; a created
  model whose base left the cache is an actionable 404.
- `forward.py` — Ollama↔OpenAI translation + streaming (OpenAI SSE in, Ollama
  NDJSON out). Keep it worker-agnostic: it only needs a `Worker` exposing
  `base_url`, `name`, and `inflight` (streaming forwards hold `inflight` for the
  stream's lifetime so a `keep-alive=0` worker can't be reaped mid-stream — see
  `tests/test_forward_stream.py`).
- `router.py` — `Router`: the judge model picks a target from the candidates.
  `pick()` calls the judge via `forward.forward_json`, `parse_choice()` resolves
  its free-text reply (verbatim `org/name` first, bare name as a whole token),
  `fallback()` is the largest candidate. `ROUTE_ALIAS` = `"auto"`.
- `cli.py` — the single entry. `run_server()` is the server body (flags — keep in
  sync with `config.py` when adding flags); `main()` dispatches subcommands;
  `_REFUSALS` holds the polite refusals for `pull`/`push`/`login`/`cp` — **do not
  implement real download/publish** (models live in the HF cache; that is a
  deliberate NvLLM trait). `create` and `rm` are **real subcommands** since 0.4.0:
  `create NAME --from MODEL [--system …] [--temperature …] [--top-p …]
  [--num-predict …]` and `rm NAME` (rm on a non-created model still prints the
  HF-cache-delete guidance, via the server 404).
- `client.py` — the Ollama-style client commands (list/ps/show/run/stop/version/
  create/rm). Synchronous `httpx` (CLI is a foreground process; no event loop).
  `repl()` takes injectable `in_fn`/`out_fn` so tests script the whole session.
  0.4.0 additions: `run_prompt(…, images, tools)` (image files → base64 `images`
  on the last user message; `⚙ tool: name(args)` printed from streamed
  tool_calls) and REPL image-path turns + `/tools <file>` (⚠️ the image-path
  check runs **before** the slash-command check — image paths start with `/`).
- `launch.py` — `nvllm launch`: agent registry (`AGENTS`: claude/codex/qwen/
  gemini/opencode/aider) as pure data (env + args templates with `<base>`/`<model>`/
  `<alias>` placeholders) + shared exec path. Adding an agent = one `Agent(...)` entry.
  `launch_config()` is the testable seam (no exec); `cmd_launch` does require_server
  → model default (largest completion model / `--model` / `$NVLLM_LAUNCH_MODEL`) →
  foreground `subprocess.call` with stdio inherited.
  ⚠️ **Claude Code model-name gate (verified on the native 2026 build):**
  `ANTHROPIC_MODEL` env / `--model` flag are validated against a known-claude
  whitelist (rejected locally, **zero requests sent**); but the `model` key in a
  `settings.json` (a "custom model option") accepts **any** name. So
  `launch claude` writes `<state-dir>/claude-launch/settings.json`
  (base URL + token + real model) and execs with `CLAUDE_CONFIG_DIR` — see
  `claude_config_dir()`. Claude Code also always sends `tools`+`tool_choice:"auto"`
  → the worker needs `NVLLM_WORKER_EXTRA_ARGS="--enable-auto-tool-choice
  --tool-call-parser hermes"` or requests 400.
- `anthropic.py` — Anthropic Messages API ↔ OpenAI translation (request body,
  non-streaming message, `AnthropicStream` state machine for SSE). ⚠️ the
  `AnthropicStream.finish` **attribute** (finish_reason) is distinct from the
  `close()` **method** that emits the tail events — don't rename one onto the
  other (instance attribute would shadow the method). Served at `POST /v1/messages`
  so Claude Code (`nvllm launch claude`) works against local models.
- `render.py` — pure rendering helpers (`human_bytes`, `human_time`, `render_table`);
  no I/O, no deps — keep it that way.
- `index.html` — **single-file interactive demo report** (zero external deps, opens
  offline): feature/usage walkthrough + a live connect-and-chat panel that hits a
  running service's `/api/*` (needs the CORS middleware above).
  **Standing rule: keep it in sync with the code — update it on every GitHub sync**
  (version numbers, features, CLI flags, endpoint changes).

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
- **Startup preflight**: `run_server` (the `serve` path) exits early with
  actionable guidance when the `vllm` binary cannot be resolved
  (`_preflight_vllm`, reuses `workers._vllm_bin()`; tests build the app directly
  and never hit it — and client commands must never trigger it either).
- **CLI tests**: `tests/test_cli.py` runs the client commands against the real
  app on a **real port** (uvicorn thread; the `live_server` fixture reuses the
  conftest stub pattern) because the CLI talks real HTTP — TestClient can't be
  used. REPL tests inject `in_fn`/`out_fn` to script the whole session (no sleeps).
- **CI**: `.github/workflows/ci.yml` runs the suite on ubuntu / Python 3.12 — keep
  the test suite GPU-free and vLLM-free (stub worker + fake hub cache).
