"""FastAPI application assembly.

M1: model catalog + metadata endpoints (Ollama /api/*, OpenAI /v1/models).
M2: inference proxy — every request resolves the model from the catalog,
    ensures a worker (`vllm serve`) is resident, and forwards the request.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import __version__, anthropic, forward
from .catalog import Capability, ModelInfo, read_decision_info
from .config import Settings
from .created import CreatedModel, CreatedStore
from .discovery import by_name, discover
from .router import AUTOROUTER_NAME, ROUTE_ALIAS, ROUTE_TRIGGERS, Router, RouterDecision
from .router_config import RouterConfig, RouterConfigStore
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

    # ---- Router (opt-in activation, 0.6.0) -------------------------------
    # The router is a resident capability the user activates once (`nvllm router on`):
    # the choice is persisted in ``router.json`` and read here at startup, so the
    # judge *decision model* becomes resident and the ``AutoRouter`` option appears
    # in the model list. An explicit ``--router-judge`` CLI/env flag still wins
    # (0.5.0 back-compat); the persisted store is the default source.
    app.state.router_config = RouterConfigStore(settings.state_dir() / "router.json")
    router_cfg: RouterConfig | None
    if settings.router_judge:  # explicit flag/env beats the persisted activation
        router_cfg = RouterConfig(
            enabled=True,
            judge=settings.router_judge,
            candidates=list(settings.router_candidates),
            backend=settings.router_backend,
            top_logprobs=settings.router_top_logprobs,
        )
    else:
        router_cfg = app.state.router_config.load()

    router: Router | None = None
    if router_cfg is not None and router_cfg.enabled and router_cfg.judge:
        judge = by_name(catalog, router_cfg.judge)
        if judge is None or not judge.supported:
            raise RuntimeError(
                f"router judge '{router_cfg.judge}' is not a servable catalog model — "
                "run `nvllm router off` to deactivate, or point it at a cached model"
            )
        # routing targets = servable models, minus decision models (they are judges,
        # not general chat targets) and minus the judge itself (Router excludes it).
        candidates = [m for m in catalog if m.supported and not m.is_decision_model]
        if router_cfg.candidates:
            wanted = set(router_cfg.candidates)
            candidates = [m for m in candidates if m.repo in wanted]
        router = Router(
            judge=judge,
            candidates=candidates,
            backend=router_cfg.backend,
            top_logprobs=router_cfg.top_logprobs,
        )
        log.info("router enabled: judge=%s, backend=%s, %d candidate(s)", judge.name, router.backend, len(router.candidates))
    app.state.router = router
    # "transformers" judge backend: one in-process HuggingFace reader, resident and
    # shared across requests (created lazily so the "vllm" backend never imports it).
    app.state.transformer_judge = None
    if router is not None and router.backend == "transformers":
        from . import transformer_router

        app.state.transformer_judge = transformer_router.TransformerJudge(router)

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

    async def _route(request_text: str) -> "RouterDecision":
        """Run the router for `request_text` and return the decision.

        Dispatches on the configured judge backend: "vllm" (default) serves the
        judge via WorkerManager and scores its logprobs; "transformers" reads the
        in-process decision model. Never raises on a bad judge answer — the router
        itself degrades to a fallback candidate.
        """
        if router is None:
            raise HTTPException(
                status_code=400,
                detail="routing not configured — activate it with `nvllm router on`, "
                "or start the service with `--router-judge <model>`",
            )
        if router.backend == "transformers":
            return await app.state.transformer_judge.decide(request_text)
        judge_worker = await ensure(router.judge, keep_alive=-1)
        return await router.decide(judge_worker, request_text)

    async def _resolve_model(body: dict) -> ModelInfo:
        """Resolve the request's model.

        Registered aliases (POST /v1/aliases, e.g. Claude Code's gated
        'claude-*' name) are mapped back to the real catalog model first; the
        router triggers — ``AutoRouter`` (the visible option) or the hidden
        ``auto`` alias (0.5.0 back-compat) — then route through the judge.
        """
        name = str(body.get("model") or body.get("name") or "")
        target = app.state.model_aliases.get(name)
        if target:
            body["model"] = target
            name = target
        if name in ROUTE_TRIGGERS:
            decision = await _route(_request_text(body))
            log.info("routed '%s' -> %s (via %s, dist=%s)", name, decision.chosen.name, decision.backend, decision.distribution)
            return decision.chosen
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

    def _autorouter_entry_ollama() -> dict[str, Any] | None:
        """The visible ``AutoRouter`` option — present only while the router is
        active. It is a built-in virtual option (like a created model) that, when
        selected, routes the request through the resident decision-model judge."""
        if app.state.router is None:
            return None
        judge = app.state.router.judge
        return {
            "name": AUTOROUTER_NAME,
            "model": AUTOROUTER_NAME,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "size": 0,
            "digest": "",
            "details": {"format": "router", "judge": judge.name, "decision_model": True},
            "capabilities": [Capability.COMPLETION.value],
        }

    @app.get("/api/tags")
    async def api_tags() -> dict[str, Any]:
        models = [m.to_ollama_list() for m in catalog] + _created_entries_ollama()
        ar = _autorouter_entry_ollama()
        if ar is not None:
            models.append(ar)
        return {"models": models}

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
        if name == AUTOROUTER_NAME:
            if app.state.router is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"model '{AUTOROUTER_NAME}' not found — router not activated (run `nvllm router on`)",
                )
            judge = app.state.router.judge
            return {
                "name": AUTOROUTER_NAME,
                "details": {"format": "router", "judge": judge.name, "decision_model": True},
                "capabilities": [Capability.COMPLETION.value],
                "system": "AutoRouter routes each request to the best local model, chosen by a resident decision-model judge.",
            }
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
        """Router status + the decision models found in the cache (for `nvllm router on`)."""
        decision_models = []
        for m in catalog:
            if m.is_decision_model:
                info = read_decision_info(m.snapshot_dir, m.repo)
                decision_models.append(
                    {
                        "name": m.name,
                        "supported": m.supported,
                        "base_model": info.get("base_model") or m.model_type or "",
                        "description": info.get("description") or "",
                        "parameter_size": info.get("parameter_size") or "",
                        "source": info.get("source") or "decision model",
                    }
                )
        # report both the LIVE router (built this session) and the PERSISTED
        # activation (router.json — active from the next `nvllm serve` onward).
        persisted = app.state.router_config.load()
        out: dict[str, Any] = {
            "enabled": router is not None,
            "configured": persisted is not None or router is not None,
            "alias": ROUTE_ALIAS,
            "autorouter": AUTOROUTER_NAME,
            "decision_models": decision_models,
        }
        if router is not None:
            out.update(
                {
                    "backend": router.backend,
                    "judge": router.judge.name,
                    "candidates": [m.name for m in router.candidates],
                }
            )
        elif persisted is not None:
            out.update(
                {
                    "judge": persisted.judge,
                    "backend": persisted.backend,
                    "candidates": persisted.candidates,
                    "note": "configured — active on the next `nvllm serve`",
                }
            )
        return out

    @app.post("/api/router")
    async def api_router_set(request: Request) -> dict[str, Any]:
        """Activate / deactivate the router (persisted to ``router.json``).

        Body: ``{"action":"on","judge":"<catalog name>","candidates":[...],
        "backend":"vllm","top_logprobs":32}`` or ``{"action":"off"}``.
        Takes effect on the next ``nvllm serve`` startup (the judge then becomes
        resident and ``AutoRouter`` appears in the model list).
        """
        body = await _json_body(request)
        action = str(body.get("action") or "").strip().lower()
        if action == "off":
            app.state.router_config.clear()
            return {"enabled": False, "note": "router deactivated — takes effect on the next `nvllm serve`"}
        if action != "on":
            raise HTTPException(status_code=400, detail="'action' must be 'on' or 'off'")
        judge_name = str(body.get("judge") or "").strip()
        if not judge_name:
            raise HTTPException(status_code=400, detail="'judge' is required (a decision model from the cache)")
        judge = by_name(catalog, judge_name)
        if judge is None:
            raise HTTPException(status_code=404, detail=f"model '{judge_name}' not found in the cache")
        if not judge.supported:
            raise HTTPException(status_code=400, detail=f"judge '{judge.name}' is not servable: {judge.note}")
        candidates = [c for c in (body.get("candidates") or []) if isinstance(c, str) and c.strip()]
        # there must be at least one non-decision model to route to (judges aren't chat targets)
        targets = [m for m in catalog if m.supported and not m.is_decision_model and m.name != judge.name]
        if candidates:
            wanted = set(candidates)
            targets = [m for m in targets if m.repo in wanted]
        if not targets:
            raise HTTPException(status_code=400, detail="no routing targets — need at least one servable non-decision model to route to")
        backend = str(body.get("backend") or "vllm")
        top_logprobs = int(body.get("top_logprobs") or 32)
        cfg = RouterConfigStore.activate(judge.name, candidates, backend=backend, top_logprobs=top_logprobs)
        app.state.router_config.save(cfg)
        log.info("router activated: judge=%s, backend=%s, %d candidate(s)", judge.name, backend, len(candidates))
        return {
            "enabled": True,
            "judge": judge.name,
            "backend": backend,
            "candidates": candidates,
            "top_logprobs": top_logprobs,
            "note": "router activated — the judge becomes resident and AutoRouter appears on the next `nvllm serve`",
        }

    def _match_suggested(text: str, dms: list) -> str:
        """Best-effort: which decision-model name appears in a recommendation text."""
        if not text:
            return ""
        low = text.lower()
        best, best_len = "", 0
        for d in dms:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "")
            for key in (name, name.split("/", 1)[-1]):
                if key and key.lower() in low and len(key) > best_len:
                    best, best_len = name, len(key)
        return best

    @app.post("/api/router/recommend")
    async def api_router_recommend(request: Request) -> dict[str, Any]:
        """Ask an already-loaded local chat model to recommend a decision model.

        Body: ``{"recommender":"<catalog name>", "decision_models":[{name,description,base_model,...}]}``.
        Runs the recommender once (non-streaming) and returns its recommendation
        plus a best-effort ``suggested`` decision-model name parsed from it.
        """
        body = await _json_body(request)
        recommender_name = str(body.get("recommender") or "").strip()
        if not recommender_name:
            raise HTTPException(status_code=400, detail="'recommender' is required")
        rec = by_name(catalog, recommender_name)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"recommender '{recommender_name}' not found")
        if not rec.supported or Capability.COMPLETION not in rec.capabilities:
            raise HTTPException(status_code=400, detail=f"recommender '{rec.name}' is not a servable chat model")
        dms = [d for d in (body.get("decision_models") or []) if isinstance(d, dict)]
        if not dms:
            raise HTTPException(status_code=400, detail="need at least one decision model to evaluate")
        listing = "\n".join(
            f"- {d.get('name')}: {d.get('description') or '(no description)'} "
            f"(base {d.get('base_model') or '?'}, {d.get('parameter_size') or '?'})"
            for d in dms
        )
        prompt = (
            "I am choosing a 'System One' decision model to act as a router judge that picks "
            "the best local model for each request. Based only on the info below, recommend the "
            "single best decision model for this role and justify it in 1-2 sentences. "
            "Start your reply with the recommended model's name.\n\nDecision models:\n" + listing
        )
        worker = await ensure(rec, forward.parse_keep_alive(None))
        async with manager.track(worker):
            resp = await forward.forward_json(
                worker,
                "/v1/chat/completions",
                {"model": rec.name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0},
            )
        content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return {"recommender": rec.name, "recommendation": content, "suggested": _match_suggested(content, dms)}

    @app.post("/api/route")
    async def api_route(request: Request) -> dict[str, Any]:
        """Preview the routing decision WITHOUT consuming a backend model.

        Body: ``{"text": "..."}`` (or the usual ``messages``/``prompt``). Returns
        the chosen model, the judge's probability distribution over the
        candidates, and which backend produced it. This is the ``nvllm route``
        dry-run counterpart — it only runs the judge, not the routed model.
        """
        body = await _json_body(request)
        if router is None:
            raise HTTPException(status_code=400, detail="routing not configured; set --router-judge to enable model='auto'")
        text = str(body.get("text") or "").strip() or _request_text(body)
        decision = await _route(text)
        return {
            "alias": ROUTE_ALIAS,
            "backend": decision.backend,
            "judge": router.judge.name,
            "chosen": decision.chosen.name,
            "distribution": decision.distribution,
            "raw": decision.raw,
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
        if app.state.router is not None:  # AutoRouter appears only when the router is active
            data.append({"id": AUTOROUTER_NAME, "object": "model", "created": 0, "owned_by": "nvllm"})
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
