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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import __version__, forward
from .catalog import Capability, ModelInfo
from .config import Settings
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
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.manager = manager

    # M3: Router — only active when a judge model is configured
    router: Router | None = None
    if settings.router_judge:
        judge = by_name(catalog, settings.router_judge)
        if judge is None or not judge.supported:
            raise RuntimeError(f"router judge '{settings.router_judge}' is not a servable catalog model")
        router = Router(judge=judge, candidates=[m for m in catalog if m.supported])
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
        # the configured judge model is never evicted once loaded
        is_judge = bool(settings.router_judge) and info.name == settings.router_judge
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
        """Resolve the request's model; the 'auto' alias triggers the Router (if enabled)."""
        name = str(body.get("model") or body.get("name") or "")
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

    @app.get("/api/tags")
    async def api_tags() -> dict[str, Any]:
        return {"models": [m.to_ollama_list() for m in catalog]}

    @app.get("/api/ps")
    async def api_ps() -> dict[str, Any]:
        return {"models": await manager.loaded_models()}

    @app.post("/api/show")
    async def api_show(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        name = body.get("model") or body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail='"model" is required')
        return find_model(str(name)).to_show()

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
        info = await _resolve_model(body)
        require(info, Capability.COMPLETION)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
            oai = forward.openai_chat_body(info.name, body)
            t0 = time.time()
            if not oai.get("stream"):
                resp = await forward.forward_json(worker, "/v1/chat/completions", oai)
                return forward.ollama_chat_response(info.name, resp, t0)
            return await forward.ollama_chat_stream(worker, info.name, oai, t0)

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
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request) -> Any:
        body = await _json_body(request)
        info = await _resolve_model(body)
        worker = await ensure(info, forward.parse_keep_alive(body.get("keep_alive")))
        async with manager.track(worker):
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

    return app
