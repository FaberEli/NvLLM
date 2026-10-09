"""FastAPI application assembly (M1: catalog + metadata endpoints)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .catalog import ModelInfo
from .config import Settings
from .discovery import by_name, discover

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    catalog: list[ModelInfo] = discover(settings.hub_root())
    log.info("discovered %d model(s) under %s", len(catalog), settings.hub_root())

    app = FastAPI(title="nvllm-serve", version=__version__)
    app.state.settings = settings
    app.state.catalog = catalog
    # M2+: app.state.scheduler, app.state.workers, app.state.router

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

    app.state.find_model = find_model

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
        sched = getattr(app.state, "scheduler", None)
        if sched is None:
            return {"models": []}
        return {"models": await sched.loaded_models()}

    @app.post("/api/show")
    async def api_show(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        name = body.get("model") or body.get("name")
        if not name:
            raise HTTPException(status_code=400, detail='"model" is required')
        return find_model(str(name)).to_show()

    # ---- OpenAI-compatible -------------------------------------------
    @app.get("/v1/models")
    async def v1_models() -> dict[str, Any]:
        data = [m.to_openai_model() for m in catalog if m.supported]
        return {"object": "list", "data": data}

    # ---- helpers -------------------------------------------------------
    async def _json_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON object expected")
        return body

    return app


def _raise_ollama_error(e: Exception) -> JSONResponse:  # pragma: no cover - helper
    """Translate HTTPException into Ollama's {"error": "..."} shape."""
    status = getattr(e, "status_code", 500)
    detail = getattr(e, "detail", str(e))
    return JSONResponse(status_code=status, content={"error": str(detail)})
