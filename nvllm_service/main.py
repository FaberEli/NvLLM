"""FastAPI application assembly.

M1: model catalog + metadata endpoints (Ollama /api/*, OpenAI /v1/models).
M2: inference proxy — every request resolves the model from the catalog,
    ensures a worker (`vllm serve`) is resident, and forwards the request.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import __version__, anthropic, forward
from .catalog import Capability, ModelInfo
from .config import Settings
from .created import CreatedModel, CreatedStore
from .discovery import by_name, discover
from .router import ROUTE_ALIAS, Router
from .workers import Worker, WorkerManager

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    catalog: list[ModelInfo] = discover(settings.hub_root())
    log.info("discovered %d model(s) under %s", len(catalog), settings.hub_root())
    manager = WorkerManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start_monitor()
        try:
            yield
        finally:
            await manager.stop_all()

    app = FastAPI(title="nvllm-serve", version=__version__, lifespan=lifespan)
    # The browser demo report (index.html) and any web client may live on an
    # arbitrary origin; the service is unauthenticated and LAN-oriented, so
    # cross-origin calls are allowed (same posture as the default 0.0.0.0 bind).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.manager = manager
    # Anthropic-model-name aliases -> real catalog model. Lets agents that
    # client-side-gate model names (Claude Code) point at local models: the
    # agent sends a recognized "claude-*" name, the server maps it.
    app.state.model_aliases: dict[str, str] = {}
    # `nvllm create` virtual models (name -> base + system + options), persisted
    # in the state dir so they survive restarts.
    app.state.created = CreatedStore(settings.state_dir() / "created.json")

    # M3: Router — only active when a judge model is configured
    router: Router | None = None
    if settings.router_judge:
        judge = by_name(catalog, settings.router_judge)
        if judge is None or not judge.supported:
            raise RuntimeError(f"router judge '{settings.router_judge}' is not a servable catalog model")
        candidates = [m for m in catalog if m.supported]
        if settings.router_candidates:
            wanted = set(settings.router_candidates)
            candidates = [m for m in candidates if m.repo in wanted]
        router = Router(judge=judge, candidates=candidates)
        log.info("router enabled: judge=%s, %d candidate(s)", judge.name, len(router.candidates))
    app.state.router = router

    @app.exception_handler(HTTPException)
    async def _ollama_style_errors(request: Request, exc: HTTPException) -> JSONResponse:
        # Ollama convention: every error body is {"error": "..."}
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    def find_model(name: str) -> ModelInfo:
        m = by_name(catalog, name)
        if m is None:
            raise HTTPException(status_code=404, detail=f"model '{name}' not found")
        if not m.supported:
            raise HTTPException(status_code=400, detail=f"model '{m.name}' is not servable: {m.note}")
        return m

    def require(info: ModelInfo, cap: Capability) -> None:
        if cap not in info.capabilities:
            raise HTTPException(status_code=400, detail=f"model '{info.name}' does not support {cap.value}")

    async def ensure(info: ModelInfo, keep_alive: int | None) -> Worker:
        # the configured judge model is never evicted once loaded. Match by repo, not
        # name: info.name is "repo:tag" while router_judge is given as "repo" (or a
        # different tag), so a name compare would miss it and load the judge as an
        # ordinary (evictable) worker.
        is_judge = router is not None and info.repo == router.judge.repo
        try:
            return await manager.ensure(info, keep_alive=keep_alive, is_judge=is_judge)
        except (TimeoutError, RuntimeError, OSError) as e:
            raise HTTPException(status_code=502, detail=f"failed to load model: {e}")

    async def _json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object expected")
        return body

    def _request_text(body: dict) -> str:
        """Extract the user's text for the router: last user message, else prompt/input."""
        for m in reversed(body.get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(
                        p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                    )
        p = body.get("prompt")
        if isinstance(p, str):
            return p
        i = body.get("input")
        if isinstance(i, str):
            return i
        return ""

    async def _resolve_model(body: dict) -> ModelInfo:
        """Resolve the request's model.

        Registered aliases (POST /v1/aliases, e.g. Claude Code's gated
        'claude-*' name) are mapped back to the real catalog model first;
        the 'auto' alias then triggers the Router (if enabled).
        """
        name = str(body.get("model") or body.get("name") or "")
        target = app.state.model_aliases.get(name)
        if target:
            body["model"] = target
            name = target
        if name == ROUTE_ALIAS:
            if router is None:
                raise HTTPException(
                    status_code=400,
                    detail="routing not configured; set --router-judge to enable model='auto'",
                )
            judge_worker = await ensure(router.judge, keep_alive=-1)
            chosen = await router.pick(judge_worker, _request_text(body))
            log.info("routed 'auto' -> %s", chosen.name)
            return chosen
        return find_model(name)

    def _created(name: str) -> CreatedModel | None:
        """The virtual-model wrapper for `name`, if one was created (`nvllm create`)."""
        return app.state.created.get(name)

    async def _resolve_with_spec(body: dict) -> tuple[ModelInfo, str, CreatedModel | None]:
        """Resolve the request's model; a created (virtual) name resolves to its
        base model (shared worker) while its wrapper spec is returned for
        request-level injection. Returns (base_info, requested_name, spec)."""
        requested = str(body.get("model") or body.get("name") or "")
        cm = _created(requested)
        if cm is not None:
            base = by_name(catalog, cm.base)
            if base is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"created model '{requested}' points at base '{cm.base}', which is no longer in the HuggingFace cache",
                )
            body["model"] = cm.base  # catalog resolution happens against the base
        info = await _resolve_model(body)
        return info, requested, cm

    # ---- created-model (virtual) spec injection, shared across protocols ----
    def _apply_system_to_messages(msgs: list, system: str) -> None:
        """Merge a virtual model's system prompt into an Ollama/OpenAI-style
        message list: append to an existing system message, or prepend one."""
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                c = m.get("content")
                m["content"] = f"{c}\n\n{system}" if isinstance(c, str) and c else system
                return
        msgs.insert(0, {"role": "system", "content": system})

    def _apply_sampling(body: dict, options: dict) -> None:
        """Merge sampling options (explicit request params win; num_predict -> max_tokens)."""
        for k in ("temperature", "top_p"):
            if k in options and k not in body:
                body[k] = options[k]
        if "num_predict" in options and "max_tokens" not in body:
            body["max_tokens"] = int(options["num_predict"])

    def _inject_ollama_spec(body: dict, cm: CreatedModel | None) -> None:
        """Ollama bodies: system prompt as a message, options merged (request wins)."""
        if cm is None:
            return
        if cm.system and isinstance(body.get("messages"), list):
            _apply_system_to_messages(body["messages"], cm.system)
        if cm.options:
            body["options"] = {**cm.options, **(body.get("options") or {})}

    def _inject_openai_spec(body: dict, cm: CreatedModel | None) -> None:
        """OpenAI-style bodies: system message + top-level params (request wins)."""
        if cm is None:
            return
        if cm.system and isinstance(body.get("messages"), list):
            _apply_system_to_messages(body["messages"], cm.system)
        if cm.options:
            _apply_sampling(body, cm.options)

    def _inject_anthropic_spec(body: dict, cm: CreatedModel | None) -> None:
        """Anthropic bodies: system prompt into the `system` field; params top-level."""
        if cm is None:
            return
        if cm.system:
            existing = body.get("system")
            if isinstance(existing, str) and existing:
                body["system"] = f"{existing}\n\n{cm.system}"
            elif not existing:
                body["system"] = cm.system
        if cm.options:
            _apply_sampling(body, cm.options)

    # ---- misc ---------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> PlainTextResponse:
        return PlainTextResponse("nvllm-serve is running")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    # ---- Ollama-compatible -------------------------------------------
    @app.get("/api/version")
    async def api_version() -> dict[str, str]:
        return {"version": settings.version}

    def _created_entries_ollama() -> list[dict[str, Any]]:
        out = []
        for cm in app.state.created.list():
            base = next((m for m in catalog if m.name == cm.base), None)
            out.append(
                {
                    "name": cm.name,
                    "model": cm.name,
                    "modified_at": cm.created_at,
                    "size": base.size if base else 0,
                    "digest": "",
                    "details": {"parent_model": cm.base, "format": "virtual"},
                    "capabilities": sorted(c.value for c in (base.capabilities if base else ())),
                }
            )
        return out

    @app.get("/api/tags")
    async def api_tags() -> dict[str, Any]:
        return {"models": [m.to_ollama_list() for m in catalog] + _created_entries_ollama()}

    @app.get("/api/ps")
    async def api_ps() -> dict[str, Any]:
        return {"models": await manager.loaded_models()}

    @app.post("/api/show")
    async def api_show(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        name = body.get("model") or body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail='"model" is required')
        name = str(name)
        cm = _created(name)
        if cm is not None:
            base = find_model(cm.base)
            out: dict[str, Any] = base.to_show()
            out["details"]["parent_model"] = cm.base
            if cm.system:
                out["system"] = cm.system
            if cm.options:
                out["options"] = cm.options
            return out
        return find_model(name).to_show()

    @app.post("/api/unload")
    async def api_unload(request: Request) -> dict[str, Any]:
        """Unload a loaded model (the `nvllm stop` counterpart of Ollama's stop).

        A created (virtual) name resolves to its base: the virtual model has no
        worker of its own — it shares the base's, so stopping it stops the base.
        """
        body = await _json_body(request)
        name = str(body.get("model") or body.get("name") or "")
        if not name:
            raise HTTPException(status_code=400, detail='"model" is required')
        cm = _created(name)
        if cm is not None:
            name = cm.base
        info = by_name(catalog, name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"model '{name}' not found")
        if not await manager.unload(info.name):
            raise HTTPException(status_code=404, detail=f"model '{info.name}' is not loaded")
        return {"model": info.name}

    @app.post("/api/create")
    async def api_create(request: Request) -> dict[str, Any]:
        """Create a virtual model: `nvllm create NAME --from BASE [--system ...]`.

        Pure metadata over a catalog model (name + system prompt + sampling
        options) — no weights involved; serving it uses the base's worker.
        """
        body = await _json_body(request)
        name = str(body.get("name") or "").strip()
        base = str(body.get("model") or "").strip()
        system = str(body.get("system") or "")
        options = body.get("options")
        options = dict(options) if isinstance(options, dict) else {}
        if not name:
            raise HTTPException(status_code=400, detail='"name" is required')
        if not base:
            raise HTTPException(status_code=400, detail='"model" (the base model) is required')
        info = by_name(catalog, base)
        if info is None:
            raise HTTPException(status_code=404, detail=f"base model '{base}' not found")
        if not info.supported:
            raise HTTPException(status_code=400, detail=f"base model '{info.name}' is not servable: {info.note}")
        if by_name(catalog, name) is not None:
            raise HTTPException(status_code=400, detail=f"name '{name}' collides with a catalog model")
        cm = CreatedModel(name=name, base=info.name, system=system, options=options)
        try:
            app.state.created.create(cm)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        log.info("created model: %s (base=%s, system=%dB, options=%s)", name, info.name, len(system), options)
        return cm.to_api()

    @app.post("/api/rm")
    async def api_rm(request: Request) -> dict[str, Any]:
        """Delete a *created* (virtual) model. Real catalog models are never
        deleted by the server (NvLLM trait: they are files in the HF cache)."""
        body = await _json_body(request)
        name = str(body.get("name") or body.get("model") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail='"name" is required')
        if not app.state.created.delete(name):
            raise HTTPException(status_code=404, detail=f"'{name}' is not a created model")
        log.info("deleted created model: %s", name)
        return {"name": name}

    @app.get("/api/router")
    async def api_router() -> dict[str, Any]:
        if router is None:
            return {"enabled": False, "alias": ROUTE_ALIAS}
        return {
            "enabled": True,
            "alias": ROUTE_ALIAS,
            "judge": router.judge.name,
            "candidates": [m.name for m in router.candidates],
        }

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await _json_body(request)
        info, requested, spec = await _resolve_with_spec(body)
        require(info, Capability.COMPLETION)
        _inject_ollama_spec(body, spec)
        shown = requested if spec else info.name  # virtual name back to the client
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            oai = forward.openai_chat_body(info.name, body)
            t0 = time.time()
            if not oai.get("stream"):
                resp = await forward.forward_json(worker, "/v1/chat/completions", oai)
                return forward.ollama_chat_response(shown, resp, t0)
            return await forward.ollama_chat_stream(worker, shown, oai, t0)

    @app.post("/api/generate")
    async def api_generate(request: Request):
        body = await _json_body(request)
        info = await _resolve_model(body)
        require(info, Capability.COMPLETION)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            oai = forward.openai_completions_body(info.name, body)
            t0 = time.time()
            if not oai.get("stream"):
                resp = await forward.forward_json(worker, "/v1/completions", oai)
                return forward.ollama_generate_response(info.name, resp, t0)
            return await forward.ollama_generate_stream(worker, info.name, oai, t0)

    @app.post("/api/embed")
    async def api_embed(request: Request):
        body = await _json_body(request)
        info = await _resolve_model(body)
        require(info, Capability.EMBEDDING)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            inp = body.get("input")
            if inp is None:
                raise HTTPException(status_code=400, detail='"input" is required')
            if isinstance(inp, str):
                inp = [inp]
            resp = await forward.forward_json(worker, "/v1/embeddings", {"model": info.name, "input": list(inp)})
            rows = sorted(resp.get("data") or [], key=lambda r: r.get("index", 0))
            return {"model": info.name, "embeddings": [r.get("embedding") for r in rows]}

    @app.post("/api/embeddings")
    async def api_embeddings(request: Request):
        body = await _json_body(request)
        info = await _resolve_model(body)
        require(info, Capability.EMBEDDING)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            prompt = body.get("prompt")
            if prompt is None:
                raise HTTPException(status_code=400, detail='"prompt" is required')
            resp = await forward.forward_json(worker, "/v1/embeddings", {"model": info.name, "input": prompt})
            data = resp.get("data") or []
            if not data:
                raise HTTPException(status_code=502, detail="worker returned no embedding")
            return {"embedding": data[0].get("embedding"), "model": info.name}

    # ---- OpenAI-compatible -------------------------------------------
    @app.get("/v1/models")
    async def v1_models() -> dict[str, Any]:
        data = [m.to_openai_model() for m in catalog if m.supported]
        data += [
            {"id": cm.name, "object": "model", "created": 0, "owned_by": "nvllm"}
            for cm in app.state.created.list()
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Any:
        body = await _json_body(request)
        info, requested, spec = await _resolve_with_spec(body)
        _inject_openai_spec(body, spec)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            if spec is not None:
                # 虚拟模型：响应的 `model` 字段回显请求名（与 /api/chat、/v1/messages 一致）
                if not body.get("stream"):
                    resp = await forward.forward_json(worker, "/v1/chat/completions", body)
                    if isinstance(resp, dict):
                        resp["model"] = requested
                    return resp
                return await forward.openai_passthrough(worker, "/v1/chat/completions", body, as_model=requested)
            return await forward.openai_passthrough(worker, "/v1/chat/completions", body)

    @app.post("/v1/completions")
    async def v1_completions(request: Request) -> Any:
        body = await _json_body(request)
        info = await _resolve_model(body)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            return await forward.openai_passthrough(worker, "/v1/completions", body)

    @app.post("/v1/embeddings")
    async def v1_embeddings(request: Request) -> Any:
        body = await _json_body(request)
        info = await _resolve_model(body)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            return await forward.openai_passthrough(worker, "/v1/embeddings", body)

    # ---- Anthropic-compatible (for `nvllm launch claude` & other Anthropic clients) ----
    def _anthropic_error(e: HTTPException) -> JSONResponse:
        code, payload = anthropic.error_response(e.status_code, str(e.detail))
        return JSONResponse(status_code=code, content=payload)

    async def _anthropic_stream(worker: Worker, model: str, oai: dict) -> StreamingResponse:
        """OpenAI SSE in -> Anthropic SSE out; holds the worker's in-flight ref
        for the stream's lifetime (same pattern as forward.openai_passthrough)."""
        resp, client = await forward._openai_stream(worker, "/v1/chat/completions", oai)
        acc = anthropic.AnthropicStream(model)
        worker.inflight += 1

        async def gen():
            try:
                for ev in acc.start():
                    yield ev
                async for evt in forward._iter_sse_events(resp):
                    for ev in acc.apply(evt):
                        yield ev
                for ev in acc.close():
                    yield ev
            finally:
                await resp.aclose()
                await client.aclose()
                worker.inflight = max(0, worker.inflight - 1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/aliases")
    async def v1_aliases(request: Request) -> dict[str, Any]:
        """Register an alias name -> real model, e.g.
        {"name": "claude-sonnet-4-5", "model": "nvidia/Qwen3-8B-NVFP4"}."""
        body = await _json_body(request)
        alias = str(body.get("name") or "").strip()
        name = str(body.get("model") or "").strip()
        if not alias or not name:
            raise HTTPException(status_code=400, detail='"name" and "model" are required')
        info = by_name(catalog, name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"model '{name}' not found")
        if not info.supported:
            raise HTTPException(status_code=400, detail=f"model '{info.name}' is not servable: {info.note}")
        app.state.model_aliases[alias] = info.name
        log.info("model alias: %s -> %s", alias, info.name)
        return {"name": alias, "model": info.name}

    @app.get("/v1/aliases")
    async def v1_aliases_list() -> dict[str, Any]:
        return {"aliases": dict(app.state.model_aliases)}

    @app.post("/v1/messages")
    async def v1_messages(request: Request):
        body = await _json_body(request)
        try:
            info, requested, spec = await _resolve_with_spec(body)  # aliases + 'auto' + created
            require(info, Capability.COMPLETION)  # same guard as /api/chat
            _inject_anthropic_spec(body, spec)
            shown = requested if spec else info.name
            worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
            oai = anthropic.anthropic_chat_body(info.name, body)
            if not oai.get("stream"):
                async with manager.track(worker):
                    resp = await forward.forward_json(worker, "/v1/chat/completions", oai)
                return anthropic.anthropic_message(shown, resp)
            return await _anthropic_stream(worker, shown, oai)
        except HTTPException as e:
            return _anthropic_error(e)

    return app
