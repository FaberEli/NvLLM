"""Router activation (0.6.0): decision-model detection in the API, on/off
persistence, the recommend step, and the `AutoRouter` option taking effect on
(re)build. Uses the shared fixtures; GPU-free.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---- GET /api/router ----------------------------------------------------------


def test_api_router_lists_decision_models(client):
    body = client.get("/api/router").json()
    assert body["enabled"] is False  # not activated in this session
    assert body["configured"] is False
    assert body["autorouter"] == "AutoRouter"
    names = [d["name"] for d in body["decision_models"]]
    assert "acme/DecisionLM:main" in names
    # a decision model's yaml metadata is surfaced
    dm = next(d for d in body["decision_models"] if d["name"] == "acme/DecisionLM:main")
    assert dm["base_model"] == "Qwen3-4B"
    assert "routing" in dm["description"]


def test_api_router_activate_reports_configured(client):
    r = client.post("/api/router", json={"action": "on", "judge": "acme/DecisionLM", "candidates": ["acme/TinyLM"]})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["judge"].startswith("acme/DecisionLM")
    # activation is persisted, so status reports it as configured (live-activation
    # happens on the next `nvllm serve`, hence `enabled` is still False here)
    st = client.get("/api/router").json()
    assert st["configured"] is True
    assert st["enabled"] is False
    assert st["judge"].startswith("acme/DecisionLM")


def test_api_router_off(client):
    client.post("/api/router", json={"action": "on", "judge": "acme/DecisionLM"})
    assert client.get("/api/router").json()["configured"] is True
    r = client.post("/api/router", json={"action": "off"})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert client.get("/api/router").json()["configured"] is False


def test_api_router_activate_unknown_judge_404(client):
    r = client.post("/api/router", json={"action": "on", "judge": "nope/Nope"})
    assert r.status_code == 404


def test_api_router_bad_action_400(client):
    assert client.post("/api/router", json={"action": "maybe"}).status_code == 400


# ---- recommend step -----------------------------------------------------------


def test_router_recommend(client):
    dms = [{"name": "acme/DecisionLM:main", "description": "a 4B decision model", "base_model": "Qwen3-4B", "parameter_size": "4B"}]
    r = client.post("/api/router/recommend", json={"recommender": "acme/TinyLM", "decision_models": dms})
    assert r.status_code == 200
    body = r.json()
    assert body["recommender"] == "acme/TinyLM:main"
    assert isinstance(body["recommendation"], str) and body["recommendation"]
    assert "suggested" in body


def test_router_recommend_requires_recommender(client):
    assert client.post("/api/router/recommend", json={"decision_models": [{"name": "x"}]}).status_code == 400


# ---- AutoRouter taking effect on (re)build -----------------------------------


def _build_activated_app(settings, stub_server, monkeypatch, candidates=None):
    """Build the app AFTER persisting an activation, so the router is live —
    simulating a `nvllm serve` that reads router.json at startup."""
    from nvllm_service.main import create_app
    from nvllm_service.router_config import RouterConfigStore
    from nvllm_service.workers import STATE_READY, Worker

    store = RouterConfigStore(settings.state_dir() / "router.json")
    store.save(RouterConfigStore.activate("acme/DecisionLM:main", candidates or ["acme/TinyLM"]))

    app = create_app(settings)
    assert app.state.router is not None, "router should be built from the persisted activation"
    assert app.state.router.judge.name == "acme/DecisionLM:main"
    manager = app.state.manager

    async def fake_ensure(info, keep_alive=None, is_judge=False):
        w = manager.workers.get(info.name)
        if w is None or w.state != STATE_READY:
            w = Worker(info=info, port=stub_server, state=STATE_READY, pid=9999)
            manager.workers[info.name] = w
        return w

    monkeypatch.setattr(manager, "ensure", fake_ensure)
    return TestClient(app)


def test_autorouter_in_model_lists(settings, stub_server, stub_captured, monkeypatch):
    c = _build_activated_app(settings, stub_server, monkeypatch)
    tags = [m["name"] for m in c.get("/api/tags").json()["models"]]
    assert "AutoRouter" in tags
    assert "acme/DecisionLM:main" in tags  # the judge itself is still listed
    v1 = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "AutoRouter" in v1


def test_autorouter_routes_to_candidate(settings, stub_server, stub_captured, monkeypatch):
    c = _build_activated_app(settings, stub_server, monkeypatch, candidates=["acme/TinyLM"])
    stub_captured.clear()
    r = c.post("/api/chat", json={"model": "AutoRouter", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200
    assert r.json()["message"]["content"] == "Hello world"
    # the judge (decision model) was consulted first, then the target served the reply
    models_called = [b.get("model") for b in stub_captured]
    assert "acme/DecisionLM:main" in models_called
    assert "acme/TinyLM:main" in models_called
    # the routed target is a non-decision model
    assert r.json()["model"] != "acme/DecisionLM:main"


def test_auto_alias_still_routes(settings, stub_server, stub_captured, monkeypatch):
    """0.5.0 back-compat: the hidden `auto` alias still triggers routing."""
    c = _build_activated_app(settings, stub_server, monkeypatch, candidates=["acme/TinyLM"])
    r = c.post("/api/chat", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200


def test_autorouter_absent_when_router_off(settings, stub_server, stub_captured, monkeypatch):
    """Without any activation, AutoRouter is NOT in the model list."""
    from nvllm_service.main import create_app
    from nvllm_service.workers import STATE_READY, Worker

    app = create_app(settings)
    manager = app.state.manager

    async def fake_ensure(info, keep_alive=None, is_judge=False):
        w = manager.workers.get(info.name)
        if w is None or w.state != STATE_READY:
            w = Worker(info=info, port=stub_server, state=STATE_READY, pid=9999)
            manager.workers[info.name] = w
        return w

    monkeypatch.setattr(manager, "ensure", fake_ensure)
    with TestClient(app) as c:
        tags = [m["name"] for m in c.get("/api/tags").json()["models"]]
        assert "AutoRouter" not in tags
        # requesting AutoRouter without a router is a clear 400
        r = c.post("/api/chat", json={"model": "AutoRouter", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert r.status_code == 400
        assert "nvllm router on" in r.json()["error"]
