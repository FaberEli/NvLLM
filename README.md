# nvllm

Ollama-style inference suite backed by vLLM — the **NvLLM** model-backend
inference engine: one `nvllm` CLI, a resident model service, and local agents
(Claude Code, Codex, …) wired to your models.

> **0.3.0 breaking change:** the CLI is now a single `nvllm <command>` entry
> (Ollama-style). The old `nvllm-serve` binary is gone — `nvllm serve` replaces
> it (the installer retires the old symlink/unit automatically).
>
> **0.4.0:** `nvllm create` virtual models (base + system prompt + sampling
> params — metadata only, weights untouched), image & tool-call UX in `run`/REPL,
> and a self-contained one-line install (tarball bundles python + vLLM + nvllm).
>
> **0.5.0:** the Router judge is now a **decision model** (System
> One) — a small, always-resident model that picks the best backend model by
> *scoring the candidate options*, not by free text. Two judge backends
> (default: vLLM + logprob scoring; optional: in-process HuggingFace
> one-forward-pass readout), a `nvllm route` dry-run, and the license moves
> **Apache-2.0 → MIT**.
>
> **0.6.0 (this release):** routing is now an **opt-in, activatable
> capability**. `nvllm router on` scans your model cache for **System One
> decision models** (detected by a vendor-neutral `model_router.yaml`), lets you
> pick the judge — with an optional **recommendation** from an already-loaded
> local model — and persists the choice so that on the next `nvllm serve` the
> judge runs **resident** and a new **`AutoRouter`** option appears in the model
> list; selecting it routes each request to the best local model. `nvllm router
> off` / `nvllm router status` deactivate and inspect it.

Built to serve **NVFP4-quantized** models at full speed on **DGX Spark (GB10)**
unified memory. It discovers models from your local HuggingFace hub cache,
serves them on a single fixed port (default **8004**), and exposes the
**Ollama** (`/api/*`), **OpenAI** (`/v1/*`) and **Anthropic** (`/v1/messages`)
APIs. Each model runs in its own `vllm serve` subprocess so crashes are
isolated and the supervisor can reap and respawn.

> **Interactive demo report:** [`index.html`](index.html) — a single-file,
> dependency-free walkthrough of the features and usage, with a live
> connect-and-chat panel. Live at <https://FaberEli.github.io/NvLLM/>
> (GitHub Pages). For the live panel, open it on a machine where
> `nvllm serve` is running and the panel becomes a working client (the service
> sends permissive CORS headers for browser clients).

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
nvllm route "prove sqrt(2) is irrational"   # preview which model 'auto' would pick (dry-run)
nvllm router on                            # activate auto-routing: pick a decision-model judge (interactive)
nvllm router on --judge acme/DecisionLM    # …or name the judge directly (no prompt)
nvllm router status                        # show activation state + the decision models in cache
nvllm router off                           # deactivate auto-routing
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
| `nvllm router on\|off\|status` | — | NvLLM-only: **activate / deactivate / inspect** the decision-model Router (pick the judge, optional recommend) |
| `model: "AutoRouter"` | — | NvLLM-only: the resident decision-model judge routes this request to the best local model (appears once routing is activated) |
| `model: "auto"` | — | NvLLM-only: hidden back-compat alias that also triggers the Router |
| `nvllm route TEXT` | — | NvLLM-only: preview the routing decision + distribution (runs only the judge) |

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

## Routing (auto model selection with a decision model)

Set `--router-judge` to a **decision model** (a *System One* structured-decision
model — e.g. `startlux-models/StartLux-Decision-v1.1-4B-Instant` or
`internlm/Intern-Decision-4B`), not a chat LLM. It stays resident and is never
evicted. Then any inference request can use `model: "auto"`: the judge reads the
request plus the list of candidate models, **scores the candidates as options
(A/B/C, …) and picks the best one**, and the service forwards to it.

Routing is a "pick one of N" decision — exactly what these models are trained
for — so instead of parsing free text the Router reads the judge's
**probability distribution over the candidate options** and takes the argmax.
That distribution is exposed (see `/api/route` below), so you see the confidence,
not just the winner.

## Activating the Router (0.6.0, opt-in)

The Router is **off by default** and is a capability you **activate** with a
command — not a flag you remember at startup. Activation is Ollama-flavored:

```bash
nvllm router on        # interactive: scan cache for decision models → pick the judge
nvllm router status    # is it active? which judge? what decision models are in cache?
nvllm router off       # deactivate
```

What `nvllm router on` does:

1. **Scans your model cache for *System One* decision models.** A model is
   recognized as a decision model when its snapshot carries a
   **`model_router.yaml`** (a vendor-neutral marker — NvLLM does **not** hardcode
   any local model name). If **none** are found, it reports that activation
   failed and prints `hf download <org>/<decision-model>` guidance.
2. **Lists the decision models** you have, numbered, and lets you pick the judge.
   You can also pass the judge directly to skip the prompt:
   `nvllm router on --judge acme/DecisionLM`.
3. **Optional: ask a local model to recommend one.** An extra option (on by
   default) hands the decision models' metadata to an **already-loaded** local
   model (falling back to the largest servable chat model), which reads them and
   returns a recommendation; the list is shown again so you can accept or override
   it. Disable with `--no-recommend`.
4. **Persists the choice** (`state_dir()/router.json`). It takes effect on the
   **next `nvllm serve`** — at that startup the judge becomes a **resident**
   worker (never evicted) and a new **`AutoRouter`** option appears in
   `nvllm list`, `/api/tags`, and `/v1/models`. Selecting it routes each request
   through the resident judge to the best local model:

   ```bash
   nvllm run AutoRouter "prove that sqrt(2) is irrational"   # judge picks the backend
   ```

The hidden `model: "auto"` alias still works (0.5.0 back-compat), and the
`--router-judge` / `--router-candidates` / `--router-backend` /
`--router-top-logprobs` startup flags still work too — an **explicit startup flag
wins** over a persisted activation, and an activation wins over the off default.

### Judge backends

| `--router-backend` | How the judge runs | Notes |
|---|---|---|
| `vllm` (default) | the judge is a `vllm serve` worker; the choice is read from the response's `top_logprobs` over the option letters | zero new dependencies, reuses all worker lifecycle; works on DGX Spark with vLLM 0.31 |
| `transformers` | the judge is loaded **in-process** (HuggingFace + torch) and read with **one forward pass** — the native "softmax over the allowed option symbols" | more faithful; needs `transformers`+`torch` in the service env and a transformers build that knows the model's architecture |

Both backends share the same scoring core, so they agree on how a distribution
becomes a choice. If the judge can't be consulted or returns nothing usable, the
Router degrades to the **largest candidate** rather than failing the request.

### Dry-run: `nvllm route`

Preview the decision **without consuming a backend model** (only the judge runs) —
handy for exploring routing:

```bash
nvllm route "prove that sqrt(2) is irrational"
# route -> nvidia/Qwen3-14B-NVFP4:main    (backend: vllm-logprob, judge: …)
#   distribution:
#       0.3410  ##########  nvidia/Qwen3-14B-NVFP4:main
#       0.2827  ########  nvidia/Qwen3-32B-NVFP4:main
#       …
```

### Setup

**Recommended (0.6.0, opt-in):** activate the Router from the CLI, then restart
the service so the judge goes resident:

```bash
nvllm serve                                # start the service
nvllm router on                            # scan cache → pick the judge (or: --judge …)
nvllm router status                        # confirm: judge + candidates + decision models
# restart so the activation takes effect (the judge goes resident, AutoRouter appears):
nvllm serve
nvllm list                                 # note the new "AutoRouter" entry
curl -s localhost:8004/api/router | jq     # enabled + judge + candidates
nvllm route "what is 2+2?"                 # dry-run preview (runs only the judge)
curl -s localhost:8004/api/chat -d '{      # real routing via AutoRouter
  "model": "AutoRouter",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

**Alternative (0.5.0, explicit startup flags):** pass the judge at `serve` time —
this path wins over a persisted activation:

```bash
nvllm serve \
  --router-judge startlux-models/StartLux-Decision-v1.1-4B-Instant \
  --router-candidates "nvidia/Qwen3-8B-NVFP4,nvidia/Qwen3-14B-NVFP4,nvidia/Qwen3-32B-NVFP4,nvidia/Phi-4-reasoning-plus-NVFP4" \
  --router-top-logprobs 10 \
  --gpu-memory-utilization 0.35     # see the DGX Spark note below
```

> **Practical notes (validated on DGX Spark GB10):**
> - **Constrain candidates** with `--router-candidates`. With dozens of models in
>   the cache the option list is long; keep it to the few models you actually
>   route between. `--router-top-logprobs` must be ≥ the number of candidates so
>   every option letter is captured.
> - **Use a decision model, and prefer the 4B class.** Measured with the same
>   prompts: a **4B** decision model picks the light model for trivial tasks and
>   the vision model for image requests (clear signal), while the **0.8B** model
>   was near a coin-flip and occasionally wrong-direction. When the candidates are
>   all similar general LLMs the distribution is naturally flatter.
> - **Licensing of the judge:** `StartLux-Decision` is CC-BY-NC-4.0 (non-commercial),
>   `Intern-Decision` / `SII-Decision` are Apache-2.0. Pick per your use.
> - **`--gpu-memory-utilization`**: the judge is a worker too. On the Spark the
>   OS/desktop hold much of the 128G, so the default `0.5` can fail to start
>   (it requests ~61G of the 122G device). Use a lower value (e.g. `0.35`) for the
>   resident judge; the decision model is small and needs little.

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

then attach it to a GitHub Release. **GitHub caps a single release asset at
2 GB** — a bigger tarball must be split and uploaded as parts, which
`bootstrap.sh` reassembles transparently:

```bash
# tarball > 2 GB: split into ≤1.9 GB parts and upload each
split -b 1900M -d -a 2 nvllm-0.4.0-linux-aarch64.tar.gz nvllm-0.4.0-linux-aarch64.tar.gz.part-
gh release upload 0.4.0 nvllm-0.4.0-linux-aarch64.tar.gz.part-00 --clobber
gh release upload 0.4.0 nvllm-0.4.0-linux-aarch64.tar.gz.part-01 --clobber
# (≤ 2 GB releases: upload the single tarball directly — bootstrap.sh handles both)
```

Knobs: `NVLLM_VERSION`, `NVLLM_RELEASE_URL`, `NVLLM_HOME`; extra args
(e.g. `--with-service`) pass through to the installer.

### Development install

Work from the repo with an editable venv instead:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install "vllm==0.31.0"          # the engine (validated baseline)
python -m pytest -q                 # 182 tests, no GPU needed
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
| `--router-judge` | `NVLLM_ROUTER_JUDGE` | — | a **decision model** to act as the Router judge |
| `--router-candidates` | `NVLLM_ROUTER_CANDIDATES` | all supported | comma-separated repos the Router may route to (keep it to the models you route between) |
| `--router-backend` | `NVLLM_ROUTER_BACKEND` | `vllm` | judge backend: `vllm` (serve + logprob scoring) or `transformers` (in-process one-forward readout) |
| `--router-top-logprobs` | `NVLLM_ROUTER_TOP_LOGPROBS` | `32` | top-k logprobs to request from the judge for option scoring (≥ number of candidates) |
| — | `NVLLM_STATE_DIR` | `~/.local/share/nvllm-serve` | state dir: created-models store (`created.json`), Router activation (`router.json`), and `launch claude` settings.json |

> The Router activation command `nvllm router on` mirrors these flags —
> `--judge`, `--candidates`, `--backend {vllm,transformers}`, `--top-logprobs`,
> plus `--no-recommend` — and stores the choice in `state_dir()/router.json`
> (an explicit `--router-judge` at `serve` time still wins over it).

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
- Browser (cross-origin) access is allowed by design — `Access-Control-Allow-Origin: *`
  is sent on every response so web clients and the `index.html` demo panel work
  from any origin. Same trust posture as the open bind; don't expose it
  unauthenticated to the internet.

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
| GET | `/api/router` | routing status: `enabled` (live) + `configured` (persisted) + judge + candidates + the `decision_models` in cache |
| POST | `/api/router` | activate (`on`) / deactivate (`off`) the Router — persists to `state_dir()/router.json` |
| POST | `/api/router/recommend` | have an already-loaded model read the decision models' metadata and recommend a judge |
| POST | `/api/route` | dry-run: run only the judge, return the choice + distribution (`nvllm route`) |
| GET | `/v1/models` | OpenAI — list models |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI — inference |
| POST | `/v1/messages` | Anthropic Messages API — for `nvllm launch claude` & other Anthropic clients (streaming SSE included) |
| POST/GET | `/v1/aliases` | model-name aliases (client-accepted name → real model), used by `nvllm launch claude` |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The suite (182 tests) uses a fake HF hub cache and a stub OpenAI-compatible
worker (including a logprobs-returning stub for the Router), so it runs without a
GPU or a real vLLM install — it also runs in CI.

## What NvLLM is, borrows, and pulls

**Identity.** NvLLM is an **Ollama-style, resident inference backend for models in
your local HuggingFace cache**, driven by **vLLM**. It benchmarks Ollama's API and
feel (subcommand CLI, `/api/*`, `run`/`repl`, `ps`/`stop`, `create`) and adds its
own traits: models are **discovered from the local HF cache, never downloaded**,
and it is tuned to get full speed from **NVFP4-quantized models on DGX Spark (GB10)**
unified memory (default port **8004**).

**This repository contains only NvLLM's own code.** It does **not** include the
upstream vLLM engine, the reference Ollama, or any other referenced project's
source. (The sibling `ollama/` and `vllm/` checkouts, if present, are local
references and are not part of this repo.)

**What it borrows**
- **Ollama** — the CLI shape, `/api/*` endpoints, and the "resident runner" UX.
  NvLLM speaks those protocols so Ollama-style clients work unchanged.
- **vLLM** — the inference engine. NvLLM **launches `vllm serve` as a subprocess**
  and talks to it over HTTP; it does **not** bundle or `import` vLLM's code. They
  are independent programs (analogous to Ollama ↔ llama.cpp). vLLM is a *runtime
  prerequisite*, not a code dependency of this repo.

**What it pulls at install / runtime** (none of it is in this repo):
- **Install:** a pinned **`vllm==0.31.0`** (the validated DGX Spark baseline) +
  its matching torch/FlashInfer, into an **isolated venv at `~/.nvllm/venv`** —
  so it does **not** interfere with any vLLM you deploy yourself. Override with
  `--vllm-version V` / `NVLLM_VLLM_VERSION`.
- **Runtime:** your **HuggingFace model cache** (you supply the weights with e.g.
  `hf download <org>/<name>`; NvLLM never downloads them) and, for the optional
  `transformers` judge backend, `transformers` + `torch`.

**What it builds (original):** HF-cache model discovery & capability detection;
the resident worker supervisor (spawn / health / LRU-evict / in-flight protection);
the Ollama + OpenAI + Anthropic tri-protocol proxy with streaming translation;
the **opt-in decision-model Router** (System One judge, logprob / one-forward
readout, `nvllm router on` activation + `AutoRouter` option + local-model
recommendation, `nvllm route` dry-run); `nvllm launch` local agents; and
metadata-only virtual models.

## Licensing

nvllm is licensed under the **MIT License** (see [LICENSE](LICENSE)).

The decision models used as Router judges are third-party checkpoints with their
own licenses — e.g. `Intern-Decision` / `SII-Decision` (Apache-2.0),
`StartLux-Decision` (CC-BY-NC-4.0, non-commercial). They are **not** part of this
repo and are governed by their own terms; choose a judge whose license fits your
use.

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
- **M13 (0.5.0)** — the Router judge becomes a **decision model (System One)**:
  candidates are scored as options and the choice is read from the judge's
  probability distribution over the option letters (default **vLLM + logprob**
  backend, or in-process **transformers** one-forward readout); graceful
  fallback chain (logprob → letter/text → largest); `nvllm route` dry-run +
  `/api/route` exposing the distribution; deep review pass; **license moved
  Apache-2.0 → MIT**; README documents what NvLLM is / borrows / original / pulls.
  Validated end-to-end on DGX Spark against real 0.8B & 4B decision models. ✅
- **M14 (0.6.0)** — routing becomes an **opt-in, activatable capability**:
  `nvllm router on` scans the model cache for **System One decision models**
  (vendor-neutral `model_router.yaml` marker, never hardcoded to local models),
  lets you pick the judge, and — optionally — has an **already-loaded local
  model read their metadata and recommend one**; the choice is **persisted**
  (`state_dir()/router.json`) and, on the next `nvllm serve`, the judge runs
  **resident** and a new **`AutoRouter`** model option appears (`nvllm list`,
  `/api/tags`, `/v1/models`) that routes each request through the resident
  judge; `nvllm router off` / `nvllm router status` deactivate and inspect;
  `POST /api/router` + `POST /api/router/recommend` endpoints. The 0.5.0
  explicit-flag path (`--router-judge`) and hidden `model: "auto"` alias remain
  (flag wins over persisted activation). ✅
