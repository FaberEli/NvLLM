# nvllm

Ollama-style inference suite backed by vLLM — the **NvLLM** model-backend
inference engine: one `nvllm` CLI, a resident model service, and local agents
(Claude Code, Codex, …) wired to your models.

> **0.3.0 breaking change:** the CLI is now a single `nvllm <command>` entry
> (Ollama-style). The old `nvllm-serve` binary is gone — `nvllm serve` replaces
> it (the installer retires the old symlink/unit automatically).
>
> **0.4.0 (this release):** `nvllm create` virtual models (base + system prompt
> + sampling params — metadata only, weights untouched), image & tool-call UX in
> `run`/REPL, and a self-contained one-line install (tarball bundles python +
> vLLM + nvllm).

Built to serve **NVFP4-quantized** models at full speed on **DGX Spark (GB10)**
unified memory. It discovers models from your local HuggingFace hub cache,
serves them on a single fixed port (default **8004**), and exposes both the
**Ollama** (`/api/*`) and **OpenAI** (`/v1/*`) APIs. Each model runs in its own
`vllm serve` subprocess so crashes are isolated and the supervisor can reap and
respawn.

## Ollama-style CLI (`nvllm`)

The usage feels like Ollama — `nvllm <command>` — while keeping NvLLM's own
traits: **models come from your local HuggingFace cache** (discovered, not
downloaded), and model names are `org/name[:tag]`.

```bash
nvllm serve                          # start the service (foreground)
nvllm list                           # models in the local HF cache (table)
nvllm list nvidia/Qwen               # …filtered by name prefix
nvllm show nvidia/Qwen3-8B-NVFP4     # details: family, params, quant, context, capabilities
nvllm run nvidia/Qwen3-8B-NVFP4 "用一句话介绍你自己"   # one-shot prompt
nvllm run nvidia/Qwen3-8B-NVFP4      # interactive REPL: /help /bye /clear /set /show
nvllm create assistant --from nvidia/Qwen3-8B-NVFP4 \
  --system "你是一个简洁的中文助手" --temperature 0.3    # virtual model (metadata only)
nvllm rm assistant                   # remove a virtual model
nvllm launch claude -p "hello"       # run Claude Code on the local models
nvllm launch codex exec "say hi"     # run Codex CLI on the local models
nvllm launch --list                  # supported agents + availability
nvllm ps                             # loaded (resident) models + UNTIL
nvllm stop nvidia/Qwen3-8B-NVFP4     # unload a model
nvllm version                        # CLI + server version
```

### Launching local agents (`nvllm launch`)

`nvllm launch` is the Ollama `ollama launch` counterpart: it execs an agent CLI
in the foreground with its model backend pointed at the running nvllm server —
no cloud, no API keys.

- **Claude Code** (`nvllm launch claude …`) — the server serves the Anthropic
  Messages API at `/v1/messages`. The native Claude Code build gates the model
  name on `ANTHROPIC_MODEL`/`--model` (only known `claude-*` names pass), but
  accepts **any** model name in `settings.json`'s `model` key — so `launch`
  writes a small config dir (`<state-dir>/claude-launch/settings.json` with the
  base URL, token, and your real model) and execs `claude` with
  `CLAUDE_CONFIG_DIR`. Note: Claude Code always sends `tools` +
  `tool_choice:"auto"`, so the worker must run with tool support — start the
  service with
  `NVLLM_WORKER_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"`
  (parser name depends on the model family).
- **Codex / Qwen Code / Gemini CLI / opencode / aider** — OpenAI-protocol agents;
  `OPENAI_BASE_URL` points at the server's `/v1`.

```bash
nvllm launch claude                      # interactive (default model: largest
                                         # completion-capable model, or --model/-m,
                                         # or $NVLLM_LAUNCH_MODEL)
nvllm launch claude -p "review this diff"
nvllm launch --model nvidia/Qwen3-8B-NVFP4 codex exec "refactor main()"
```

Everything after the agent name is passed through to the agent verbatim
(`nvllm`'s own flags go before the agent name).

The `nvllm run` REPL keeps multi-turn history; `/set temperature 0.7` (or
`top_p` / `num_predict`) changes sampling for the session. End a line with `\`
to continue it.

### Virtual models (`nvllm create`)

The Ollama `ollama create` counterpart — with NvLLM's trait intact: **no
weights are ever copied or downloaded**. A created model is pure metadata —
a base model (from your HF cache) plus a system prompt and sampling params:

```bash
nvllm create assistant --from nvidia/Qwen3-8B-NVFP4 \
  --system "你是一个简洁的中文助手" --temperature 0.3 --top-p 0.9 --num-predict 512

nvllm show assistant     # base, system, params (details.parent_model marks it)
nvllm run assistant "用一句话介绍你自己"
nvllm rm assistant       # removes the virtual model; base model stays in the HF cache
```

Two virtual models on the same base **share one worker process** — the wrapper
is injected per request (system message + option merge, request options win),
so `nvllm ps` only ever shows the base. `nvllm rm <real-model>` is still
refused with guidance (delete the `models--*` cache dir yourself); only created
models can be removed.

### Images & tools

The proxy already speaks vision and tool-calls to vLLM; the CLI exposes both:

```bash
nvllm run <VLM模型> --image cat.png "图里有什么"            # --image repeats
nvllm run <模型> --tools tools.json "查一下北京的天气"      # OpenAI tool array, JSON file
```

In the `run` REPL, a line that is an existing **image path**
(`.png/.jpg/.jpeg/.webp/.gif/.bmp`) sends an image turn (like Ollama's paste);
`/tools tools.json` loads tool definitions for the rest of the session
(`/tools` shows the current set). Assistant tool calls print as
`⚙ tool: name({"arg": …})` and are kept in the history so follow-up turns
include them.

**`nvllm` ↔ `ollama`**

| nvllm | ollama | note |
|---|---|---|
| `nvllm serve` | `ollama serve` | start the server |
| `nvllm list` | `ollama list` | NvLLM lists the **local HF cache**; Ollama lists registry-downloaded models |
| `nvllm run` | `ollama run` | one-shot prompt or interactive REPL |
| `nvllm ps` | `ollama ps` | loaded models |
| `nvllm stop` | `ollama stop` | unload a model |
| `nvllm show` | `ollama show` | model details |
| `nvllm launch` | `ollama launch` | start a local agent (Claude Code / Codex / …) on the local models |
| `nvllm create NAME --from M` | `ollama create` | **metadata only**: base + system + params (NvLLM never copies weights; Ollama builds a GGUF) |
| `nvllm pull` | `ollama pull` | **not integrated** — NvLLM never downloads models; it prints guidance (`hf download <org>/<name>`) |
| `nvllm rm NAME` | `ollama rm` | works for **created** models; for HF-cache models it prints guidance (delete the `models--*` dir) |
| `model: "auto"` | — | NvLLM-only: the Router picks the best model per request |

Client commands talk to a running server over HTTP (default
`http://127.0.0.1:8004`; override with `NVLLM_SERVER` or `--server`).

## How it works

- **Discovery** — scans `HF_HUB_CACHE` (or `~/.cache/huggingface/hub`) and builds a
  catalog of models, detecting capabilities (completion / embedding / tools) and
  flagging anything vLLM can't serve.
- **Workers** — one `vllm serve` process per model. The manager starts it on a
  free port, polls `/health` until ready, and reaps crashed workers. Up to
  `--max-models` non-judge models (default `1`) may be resident at once; when a
  new model would exceed that, the least-valued resident is evicted (expired
  workers first, then least-recently-used). Idle workers past their `--keep-alive`
  are unloaded automatically. The configured Router judge is never evicted.
  Loading is **serialized** (one model loads at a time — safe on shared/unified
  memory), while already-loaded models serve **concurrently**.
- **Proxy** — each request is resolved against the catalog, a worker is ensured,
  then the call is forwarded. Ollama bodies are translated to OpenAI and back,
  including streaming (OpenAI SSE in, Ollama NDJSON out).

## Routing (auto model selection)

Set `--router-judge` to a small, fast model (it stays resident and is never
evicted). Then any inference request can use `model: "auto"`: the judge reads the
request plus the list of servable models, picks the best one, and the service
forwards to it. If the judge can't be consulted or returns something
unrecognizable, the router falls back to the largest model rather than failing.

> **Pick a non-reasoning judge.** A judge that emits chain-of-thought (reasoning
> models) instead of a bare model name will not parse cleanly and will fall back.
> A small, well-instructed non-reasoning model (e.g. `Llama-3.1-8B-Instruct`)
> routes cleanly. Constrain the candidates with `--router-candidates` so the judge
> can only pick models that actually fit in memory.

```bash
nvllm serve \
  --router-judge nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  --router-candidates "nvidia/Qwen3-8B-NVFP4,nvidia/Qwen3-14B-NVFP4"

curl -s localhost:8004/api/router | jq          # routing status + candidates
curl -s localhost:8004/api/chat -d '{
  "model": "auto",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Installation

### One-click (recommended)

Prerequisites: **Linux** (validated on DGX Spark GB10, aarch64) with a CUDA
GPU, and **Python ≥ 3.11**.

```bash
git clone https://github.com/FaberEli/NvLLM.git && cd NvLLM
./scripts/install.sh
```

The installer:

1. creates a self-contained install root at `~/.nvllm/` (venv + logs + metadata);
2. **pulls the vLLM engine** — pinned to the validated baseline **`vllm==0.31.0`**
   (with its matching torch build for CUDA 13.0); override with `--vllm-version V`;
3. installs the `nvllm` CLI + its (small) Python dependencies;
4. symlinks `nvllm` into `~/.local/bin` (retiring any legacy `nvllm-serve` link);
5. optionally installs a systemd **user** service (autostart, restart-on-crash):
   `./scripts/install.sh --with-service`.

It is idempotent — re-running it upgrades in place.

### Updating

```bash
./scripts/update.sh                 # updates nvllm; keeps vLLM pinned
./scripts/update.sh --upgrade-vllm  # also follow the latest vLLM release
```

### Uninstalling (restores the environment)

```bash
./scripts/uninstall.sh              # stops the service, removes ~/.nvllm/ + the PATH symlink
./scripts/uninstall.sh --purge      # …and also deletes the model/JIT caches (destructive)
```

By default your **model weights and caches are kept** — the script lists what it
removes and what it keeps before asking.

### Self-contained release (one line)

For machines without python/pip: the release tarball bundles a prebuilt venv
(python + vLLM + nvllm, ~2-4 GB), so the installer only moves it into place —
no interpreter selection, no pip:

```bash
curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash
curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash --with-service
```

Build the tarball on an aarch64 linux machine (python 3.11-3.13, network,
~15 GB free disk; a few minutes for the vLLM pull):

```bash
scripts/build-release.sh        # → dist/nvllm-0.4.0-linux-aarch64.tar.gz
```

then attach it to a GitHub Release (`gh release create 0.4.0 dist/… --title
"nvllm 0.4.0"` or upload via the web UI). Knobs: `NVLLM_VERSION`,
`NVLLM_RELEASE_URL`, `NVLLM_HOME`; extra args (e.g. `--with-service`) pass
through to the installer.

### Development install

Work from the repo with an editable venv instead:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install "vllm==0.31.0"          # the engine (validated baseline)
python -m pytest -q                 # 143 tests, no GPU needed
```

## Quickstart

```bash
nvllm serve          # default bind is 0.0.0.0:8004
```

Then, with the CLI or plain HTTP:

```bash
nvllm list
nvllm run <org>/<Model> "hello"

curl -s localhost:8004/api/tags | jq
curl -s localhost:8004/v1/models | jq

# Ollama chat
curl -s localhost:8004/api/chat -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": false
}' | jq

# OpenAI chat
curl -s localhost:8004/v1/chat/completions -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}]
}' | jq
```

## Configuration

Precedence: **CLI flag > environment variable > default**.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--host` | `NVLLM_HOST` | `0.0.0.0` | bind host |
| `--port` | `NVLLM_PORT` | `8004` | bind port |
| `--hf-hub-root` | `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | HF cache to scan |
| `--keep-alive` | `NVLLM_KEEP_ALIVE` | `300` | idle seconds before unload (`-1` keep forever, `0` unload immediately) |
| `--max-models` | `NVLLM_MAX_MODELS` | `1` | max non-judge models resident at once (judge excluded) |
| `--max-model-len` | `NVLLM_MAX_MODEL_LEN` | (vLLM default) | passed to workers |
| `--gpu-memory-utilization` | `NVLLM_GPU_MEMORY_UTILIZATION` | `0.5` | passed to workers |
| `--router-judge` | `NVLLM_ROUTER_JUDGE` | — | model name to act as the Router judge |
| `--router-candidates` | `NVLLM_ROUTER_CANDIDATES` | all supported | comma-separated repos the Router may route to (keeps it from picking a model that won't fit in memory) |
| — | `NVLLM_STATE_DIR` | `~/.local/share/nvllm-serve` | state dir: created-models store (`created.json`) + `launch claude` settings.json |

Also honored by workers: `NVLLM_MAX_NUM_SEQS`, `NVLLM_TRUST_REMOTE_CODE`,
`NVLLM_WORKER_MAX_JOBS`, and `NVLLM_WORKER_EXTRA_ARGS` (raw args appended to every
`vllm serve`, e.g. `"--enable-auto-tool-choice --tool-call-parser hermes"`).

CLI client commands (the `nvllm` CLI) additionally honor `NVLLM_SERVER` —
the server URL (default `http://127.0.0.1:8004`); each command also accepts
`--server URL`.

> **DGX Spark (GB10):** the OS/desktop consumes a large share of the 128G
> unified memory, so vLLM's default `gpu-memory-utilization` of 0.92 fails at
> startup. `0.5` is a safe default there; raise it on a dedicated machine.

> **NVFP4 first load (DGX Spark):** the first request JIT-compiles the
> FlashInfer CUTLASS NVFP4 GEMM kernels into `~/.cache/flashinfer`. That is a
> large, memory-hungry one-time build — at the default parallelism it
> OOM-kills `nvcc` (SIGKILL / exit 137) on the Spark's shared 128G. Set
> `NVLLM_WORKER_MAX_JOBS=2` (or 3–4). The kernels are cached, so subsequent
> loads are fast.

## Multi-model concurrency

By default only **one** non-judge model is resident at a time (`--max-models 1`)
— safe on the Spark's shared 128G. Raise it to keep several models warm:

```bash
# keep two 8B-class models resident at once
NVLLM_MAX_MODELS=2 NVLLM_GPU_MEMORY_UTILIZATION=0.35 .venv/bin/python -m nvllm_service
```

- **Residency is concurrent, loading is serial.** Already-loaded models serve
  requests in parallel, but only one model loads at a time (two simultaneous
  `vllm serve` loads — each JIT-compiling kernels — can OOM the shared memory).
  A request for a not-yet-loaded model waits its turn.
- **Eviction is LRU, expired-first.** When a new model would exceed
  `--max-models`, the manager evicts expired residents first, then the
  least-recently-used. The judge is never evicted.
- **`keep-alive` now really unloads.** Idle workers past their keep-alive are
  reaped by the monitor; a worker mid-request is never killed.
- **Memory math (DGX Spark, 128G):** `gpu-memory-utilization` is per-worker, as
  a fraction of *total* device memory. So `N` workers × `U` must fit: at the
  default `U=0.5`, two workers already claim the whole 128G. For `N` models use
  roughly `U ≈ headroom / N` (e.g. `N=2 → U≈0.35`).

## Security

The service exposes model inference over HTTP and has **no built-in
authentication**. By default it binds `0.0.0.0` (all interfaces) on port `8004`,
so on a multi-host machine anyone on the network can use your GPUs.

- Keep it on a trusted network, or bind a specific interface:
  `--host 127.0.0.1` (local only) or `--host 192.168.1.10`.
- If you must expose it, put it behind a reverse proxy / VPN / firewall and add
  your own auth there. This service does not verify callers.

Workers (the `vllm serve` subprocesses) always bind `127.0.0.1` regardless of
`--host`, so only this service is directly reachable.

## Endpoints

| Method | Path | API |
|---|---|---|
| GET | `/healthz`, `/api/version` | meta |
| GET | `/api/tags` | Ollama — list models |
| GET | `/api/ps` | Ollama — loaded models |
| POST | `/api/show` | Ollama — model info |
| POST | `/api/create` | Ollama — create a virtual model (base + system + params, metadata only) |
| POST | `/api/rm` | Ollama — remove a virtual model (real models: 404 + guidance) |
| POST | `/api/unload` | Ollama — unload a loaded model (`nvllm stop`) |
| POST | `/api/chat`, `/api/generate` | Ollama — inference (stream + non-stream) |
| POST | `/api/embed`, `/api/embeddings` | Ollama — embeddings |
| GET | `/api/router` | routing status (judge + candidates) |
| GET | `/v1/models` | OpenAI — list models |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI — inference |
| POST | `/v1/messages` | Anthropic Messages API — for `nvllm launch claude` & other Anthropic clients (streaming SSE included) |
| POST/GET | `/v1/aliases` | model-name aliases (client-accepted name → real model), used by `nvllm launch claude` |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The suite (143 tests) uses a fake HF hub cache and a stub OpenAI-compatible
worker, so it runs without a GPU or a real vLLM install — it also runs in CI.

## Licensing

nvllm is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).

**Runtime dependency:** nvllm requires a **vLLM** installation
([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)). It launches
`vllm serve` as a **subprocess** and talks to it over HTTP — it does **not**
bundle or import vLLM's source code. The two are therefore independent programs
(analogous to Ollama ↔ llama.cpp); vLLM is a runtime prerequisite, not a code
dependency of this repository.

## Milestones

- **M1** — catalog discovery + metadata API + worker process management. ✅
- **M2** — inference proxy (Ollama + OpenAI, streaming). ✅
- **M3** — Router / auto-routing (`--router-judge` + `model: "auto"`). ✅
- **M4** — end-to-end NVFP4 smoke test on DGX Spark (GB10): `Llama-3.1-8B-Instruct-NVFP4`
  served over both `/api/chat` (Ollama) and `/v1/chat/completions` (OpenAI). ✅
- **M5** — multi-model concurrency: `--max-models` (co-located non-judge models,
  LRU/expired-first eviction), `keep-alive` expiry reaping, and in-flight protection
  so a request is never cut off. ✅
- **M6** — Router hardening (judge-residency fix, `--router-candidates` whitelist,
  short-name parsing) plus NVFP4 throughput/latency benchmark and a real-judge
  Router smoke test on DGX Spark (see [`bench/REPORT.md`](bench/REPORT.md)). ✅
- **M7** — deep code review (streaming in-flight protection fix, dead-config and
  version cleanup) and open-source packaging (LICENSE, security notes, docs). ✅
- **M8** — one-click packaging (`scripts/install.sh` / `update.sh` / `uninstall.sh`
  into `~/.nvllm/`, pinned vLLM baseline, optional systemd user service,
  `--version` + vLLM preflight) and open-source close-out (Apache-2.0, CI). ✅
- **M9** — Ollama-style CLI (`nvllm` subcommands: serve / list / run / ps / stop /
  show / version, one-shot + interactive REPL, `POST /api/unload`) while keeping
  NvLLM's traits (HF-cache model discovery, no model download — `pull`/`rm`/…
  print guidance instead). ✅
- **M10** — local agents (`nvllm launch claude/codex/qwen/gemini/opencode/aider`
  wired to the local service) + Anthropic Messages API at `/v1/messages`
  (streaming SSE, tools, system) + CLI unification (single `nvllm` entry; the
  `nvllm-serve` dual-entry is retired in 0.3.0 — a deliberate breaking change). ✅
- **M11 (0.4.0)** — purification pass: `nvllm create` virtual models
  (base + system + params, worker-shared, per-request injection) with a real
  `nvllm rm` for them; vision (`--image`) + tool-calls (`--tools` / REPL
  `/tools`, `⚙ tool:` display) in the CLI; `launch claude` solved (Claude Code
  gates `ANTHROPIC_MODEL` client-side but accepts any name in `settings.json`
  → `CLAUDE_CONFIG_DIR`); self-contained release (`build-release.sh` tarball
  with bundled python+vLLM, `bootstrap.sh` one-line install). ✅
