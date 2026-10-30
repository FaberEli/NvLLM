"""Router activation state (router_config.py) — persistence round-trip.

Pure file I/O in a tmp dir; no server, no GPU. Mirrors the CreatedStore test style.
"""
from __future__ import annotations

from pathlib import Path

from nvllm_service.router_config import RouterConfig, RouterConfigStore


def test_save_load_roundtrip(tmp_path: Path):
    store = RouterConfigStore(tmp_path / "router.json")
    assert store.load() is None  # nothing written yet
    cfg = RouterConfigStore.activate("acme/DecisionLM:main", ["acme/TinyLM"], backend="vllm", top_logprobs=10)
    store.save(cfg)
    loaded = store.load()
    assert loaded is not None
    assert loaded.enabled is True
    assert loaded.judge == "acme/DecisionLM:main"
    assert loaded.candidates == ["acme/TinyLM"]
    assert loaded.backend == "vllm"
    assert loaded.top_logprobs == 10


def test_load_missing_or_corrupt_returns_none(tmp_path: Path):
    store = RouterConfigStore(tmp_path / "router.json")
    assert store.load() is None
    # corrupt JSON
    (tmp_path / "router.json").write_text("{not json", encoding="utf-8")
    assert store.load() is None
    # enabled=false is treated as "not activated"
    (tmp_path / "router.json").write_text('{"enabled": false, "judge": "x"}', encoding="utf-8")
    assert store.load() is None


def test_clear_removes(tmp_path: Path):
    store = RouterConfigStore(tmp_path / "router.json")
    store.save(RouterConfigStore.activate("acme/DecisionLM:main"))
    assert store.load() is not None
    assert store.clear() is True
    assert store.load() is None
    assert store.clear() is False  # already gone


def test_save_requires_judge(tmp_path: Path):
    store = RouterConfigStore(tmp_path / "router.json")
    import pytest

    with pytest.raises(ValueError):
        store.save(RouterConfig(enabled=True, judge=""))


def test_save_rejects_bad_backend(tmp_path: Path):
    store = RouterConfigStore(tmp_path / "router.json")
    import pytest

    with pytest.raises(ValueError):
        store.save(RouterConfig(enabled=True, judge="acme/X", backend="bogus"))


def test_load_tolerates_corrupt_top_logprobs(tmp_path: Path):
    # A non-numeric top_logprobs in router.json must not crash the loader
    # (previously int() raised ValueError and took the whole server down);
    # it falls back to the default of 32.
    store = RouterConfigStore(tmp_path / "router.json")
    (tmp_path / "router.json").write_text(
        '{"enabled": true, "judge": "acme/DecisionLM:main", "top_logprobs": "lots"}',
        encoding="utf-8",
    )
    loaded = store.load()
    assert loaded is not None
    assert loaded.top_logprobs == 32
