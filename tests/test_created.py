"""Virtual (created) models: the store, /api/create, /api/rm, and request injection."""
from __future__ import annotations

import pytest

from nvllm_service.created import CreatedModel, CreatedStore


def test_store_crud(tmp_path):
    store = CreatedStore(tmp_path / "created.json")
    assert store.list() == []
    cm = CreatedModel(name="assistant", base="acme/TinyLM:main", system="be brief", options={"temperature": 0.3})
    store.create(cm)
    assert [m.name for m in store.list()] == ["assistant"]
    got = store.get("assistant")
    assert got.base == "acme/TinyLM:main"
    assert got.system == "be brief"
    assert got.options == {"temperature": 0.3}
    assert store.delete("assistant") is True
    assert store.list() == []
    assert store.delete("nope") is False


def test_store_rejects_bad_names_and_duplicates(tmp_path):
    store = CreatedStore(tmp_path / "created.json")
    with pytest.raises(ValueError):
        store.create(CreatedModel(name="bad/name", base="acme/TinyLM"))
    store.create(CreatedModel(name="ok", base="acme/TinyLM"))
    with pytest.raises(ValueError, match="already exists"):
        store.create(CreatedModel(name="ok", base="acme/TinyLM"))
    with pytest.raises(ValueError, match="base"):
        store.create(CreatedModel(name="no-base", base=""))


def test_store_persists_across_instances(tmp_path):
    p = tmp_path / "created.json"
    CreatedStore(p).create(CreatedModel(name="a", base="x/y"))
    assert CreatedStore(p).get("a") is not None


# ---- API ---------------------------------------------------------------------


def _create(client, name="assistant", base="acme/TinyLM", **kw):
    body = {"name": name, "model": base}
    body.update(kw)
    return client.post("/api/create", json=body)


def test_api_create_and_listing(client):
    r = _create(client, system="be brief", options={"temperature": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "assistant"
    assert body["base"] == "acme/TinyLM:main"
    # `nvllm list` / `nvllm run` see it
    tags = client.get("/api/tags").json()["models"]
    entry = next(m for m in tags if m["name"] == "assistant")
    assert entry["details"]["parent_model"] == "acme/TinyLM:main"
    assert "completion" in entry["capabilities"]  # inherited from the base
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "assistant" in ids


def test_api_create_validation(client):
    assert client.post("/api/create", json={"name": "x"}).status_code == 400  # missing base
    assert client.post("/api/create", json={"name": "x", "model": "nope/Nope"}).status_code == 404
    # unservable base (video model in the fake hub)
    assert client.post("/api/create", json={"name": "x", "model": "acme/WanVideo"}).status_code == 400
    # collides with a catalog (bare) name
    assert client.post("/api/create", json={"name": "TinyLM", "model": "acme/TinyLM"}).status_code == 400
    _create(client)
    assert _create(client).status_code == 400  # duplicate name


def test_api_show_created(client):
    _create(client, system="be brief", options={"temperature": 0.3})
    r = client.post("/api/show", json={"model": "assistant"})
    assert r.status_code == 200
    body = r.json()
    assert body["details"]["parent_model"] == "acme/TinyLM:main"
    assert body["system"] == "be brief"
    assert body["options"] == {"temperature": 0.3}


def test_api_chat_injects_system_and_options(client, stub_captured):
    _create(client, system="be brief", options={"temperature": 0.3, "num_predict": 40})
    r = client.post(
        "/api/chat", json={"model": "assistant", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    )
    assert r.status_code == 200
    assert r.json()["model"] == "assistant"  # virtual name back to the client
    oai = stub_captured[-1]
    assert oai["model"] == "acme/TinyLM:main"  # the worker gets the base
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["temperature"] == 0.3
    assert oai["max_tokens"] == 40  # num_predict mapped


def test_api_chat_request_options_win(client, stub_captured):
    _create(client, system="be brief", options={"temperature": 0.3})
    client.post(
        "/api/chat",
        json={
            "model": "assistant",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.9},
            "stream": False,
        },
    )
    assert stub_captured[-1]["temperature"] == 0.9  # explicit request beats the wrapper


def test_v1_chat_injects_created_spec(client, stub_captured):
    _create(client, system="be brief", options={"temperature": 0.3})
    r = client.post("/v1/chat/completions", json={"model": "assistant", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["model"] == "assistant"  # virtual name echoed, not the base
    oai = stub_captured[-1]
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["temperature"] == 0.3


def test_v1_messages_injects_created_system(client, stub_captured):
    _create(client, system="be brief", options={"num_predict": 25})
    r = client.post(
        "/v1/messages",
        json={"model": "assistant", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "assistant"
    oai = stub_captured[-1]
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["max_tokens"] == 100  # explicit request param wins over the wrapper's


def test_created_models_share_one_worker(client):
    _create(client, name="a1", base="acme/TinyLM", system="one")
    _create(client, name="a2", base="acme/TinyLM", system="two")
    for name in ("a1", "a2"):
        r = client.post("/api/chat", json={"model": name, "messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert r.status_code == 200
    loaded = [m["name"] for m in client.get("/api/ps").json()["models"]]
    assert loaded == ["acme/TinyLM:main"]  # one base worker serves both virtuals


def test_chat_created_model_with_missing_base_404(client):
    """A virtual model whose base left the cache gives an actionable 404,
    not a bare 'model not found' for the base name."""
    from nvllm_service.created import CreatedModel

    client.app.state.created.create(CreatedModel(name="orphan", base="gone/Ghost:main"))
    r = client.post("/api/chat", json={"model": "orphan", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert "no longer in the HuggingFace cache" in r.json()["error"]


def test_api_rm_only_created(client):
    _create(client)
    r = client.post("/api/rm", json={"name": "assistant"})
    assert r.status_code == 200 and r.json()["name"] == "assistant"
    assert client.post("/api/rm", json={"name": "assistant"}).status_code == 404
    # real catalog models are never deleted by the server
    assert client.post("/api/rm", json={"name": "acme/TinyLM"}).status_code == 404
    names = [m["name"] for m in client.get("/api/tags").json()["models"]]
    assert "assistant" not in names
    assert "acme/TinyLM:main" in names
