"""`GET /api/doctor` — environment + service health snapshot (drives `nvllm doctor`)."""
from __future__ import annotations


def test_doctor_shape(client, settings):
    r = client.get("/api/doctor")
    assert r.status_code == 200
    d = r.json()
    assert d["version"]  # package version
    s = d["settings"]
    assert s["port"] == 8004
    assert s["logs_dir"] == str(settings.logs_dir())
    assert s["state_dir"] == str(settings.state_dir())
    assert s["hub_root"] == str(settings.hub_root())
    m = d["models"]
    assert m["total"] == 4  # fake hub: TinyLM, Embedder, WanVideo, DecisionLM
    assert m["supported"] == 3  # WanVideo is unsupported
    assert m["decision"] == 1  # DecisionLM carries model_router.yaml
    assert m["loaded"] == []
    assert d["workers"] == []
    # no judge configured in this app
    assert d["router"]["enabled"] is False
    assert d["router"]["configured"] is False
    # GPU is box-dependent (present on DGX Spark, absent in CI). On GB10 the
    # memory fields are [N/A] -> None; only the name is guaranteed when present.
    assert d["gpu"] is None or d["gpu"]["name"]


def test_doctor_reports_loaded_worker(client):
    name = client.get("/api/tags").json()["models"][0]["name"]
    r = client.post("/api/chat", json={"model": name, "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200
    d = client.get("/api/doctor").json()
    assert d["models"]["loaded"] == [name]
    w = d["workers"][0]
    assert w["state"] == "ready"
    assert w["pid"] == 9999  # the stub worker from conftest
    assert w["inflight"] == 0


def test_doctor_router_enabled_when_judge_set(client):
    # activation is per-startup ("next serve"): build a second app on the same
    # settings with an explicit judge flag (the 0.5.0 path)
    from fastapi.testclient import TestClient

    from nvllm_service.main import create_app

    s = client.app.state.settings
    s.router_judge = "acme/DecisionLM:main"
    with TestClient(create_app(s)) as c2:
        d = c2.get("/api/doctor").json()
    assert d["router"]["enabled"] is True
    assert d["router"]["judge"] == "acme/DecisionLM:main"
    assert d["router"]["configured"] is True
    # decision models are never routing candidates (they are judges)
    assert "acme/DecisionLM" not in d["router"]["candidates"]
