# CLAUDE.md — nvllm-serve

Ollama-style resident inference service backed by vLLM. `README.md` is the
user-facing doc; this file is for working on the code.

## Running

- Tests (no GPU, no real vLLM needed): `.venv/bin/python -m pytest -q`
- Dev server (default bind `0.0.0.0:8004`): `.venv/bin/python -m nvllm_service --hf-hub-root <cache>`
- Entry point: `nvllm_service.cli:main` (console script `nvllm-serve`).

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
  `PATH`). Only one non-judge model is resident at a time; the Router judge is
  never evicted and never evicts. `_wait_ready` polls `/health`; the monitor
  reaps crashed workers every 2s.
- `main.py` — FastAPI assembly; `create_app(settings)` is what tests target.
  `_resolve_model(body)` resolves the request's model — the `"auto"` alias
  triggers the Router (if `--router-judge` is set); otherwise it's `find_model`.
  Endpoints then `ensure()` a worker and delegate to `forward`.
- `forward.py` — Ollama↔OpenAI translation + streaming (OpenAI SSE in, Ollama
  NDJSON out). Keep it worker-agnostic: it only needs a `Worker` exposing
  `base_url` and `name`.
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
- `Settings` precedence is **CLI > env > default**; add new options to both
  `config.py` and `cli.py`.
- Workers bind `127.0.0.1` and speak OpenAI natively; the proxy rewrites only
  the `model` field on `/v1/*` passthrough.
- Streaming helpers are async so connect-failures / worker 4xx surface as proper
  HTTP errors *before* the response stream starts.
