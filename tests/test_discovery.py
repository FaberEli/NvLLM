"""Catalog discovery and name resolution over a fake HF hub cache."""
from nvllm_service.catalog import Capability
from nvllm_service.discovery import by_name, discover


def _catalog(fake_hub):
    return discover(fake_hub)


def test_catalog_entries(fake_hub):
    catalog = _catalog(fake_hub)
    names = sorted(m.name for m in catalog)
    assert names == ["acme/DecisionLM:main", "acme/Embedder:main", "acme/TinyLM:main", "acme/WanVideo:main"]


def test_decision_model_flagged(fake_hub):
    """A model that ships model_router.yaml is flagged as a decision model; ordinary
    chat/embedding models are not."""
    catalog = _catalog(fake_hub)
    dm = next(m for m in catalog if m.repo == "acme/DecisionLM")
    assert dm.is_decision_model is True
    # a decision model is still a servable completion model (it's the router judge)
    assert Capability.COMPLETION in dm.capabilities and dm.supported
    for repo in ("acme/TinyLM", "acme/Embedder"):
        assert next(m for m in catalog if m.repo == repo).is_decision_model is False


def test_read_decision_info_parses_yaml(fake_hub):
    from nvllm_service.catalog import read_decision_info

    dm = next(m for m in _catalog(fake_hub) if m.repo == "acme/DecisionLM")
    info = read_decision_info(dm.snapshot_dir)
    assert info.get("decision_model") is True
    assert info.get("base_model") == "Qwen3-4B"
    assert "decision model for routing" in info.get("description", "")
    # a non-decision model yields no decision info
    tiny = next(m for m in _catalog(fake_hub) if m.repo == "acme/TinyLM")
    assert read_decision_info(tiny.snapshot_dir) == {}


def test_capabilities_detected(fake_hub):
    catalog = _catalog(fake_hub)
    tiny = next(m for m in catalog if m.repo == "acme/TinyLM")
    assert Capability.COMPLETION in tiny.capabilities
    assert Capability.TOOLS in tiny.capabilities  # ships a chat template
    emb = next(m for m in catalog if m.repo == "acme/Embedder")
    assert Capability.EMBEDDING in emb.capabilities
    assert Capability.COMPLETION not in emb.capabilities


def test_symlinked_weights_report_blob_size(fake_hub):
    """Real HF caches store snapshots as symlinks into blobs/ — the size must
    follow the link (the real blob), not measure the ~50-byte link path."""
    from nvllm_service.discovery import discover

    sha = "b" * 40
    mdir = fake_hub / "models--acme--Linked"
    (mdir / "refs").mkdir(parents=True)
    (mdir / "refs" / "main").write_text(sha)
    blob = mdir / "blobs" / "blob0"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\0" * 5_000_000)  # 5 MB "weights"
    snap_dir = mdir / "snapshots" / sha
    snap_dir.mkdir(parents=True)
    (snap_dir / "config.json").write_text(
        '{"architectures": ["LinkedForCausalLM"], "model_type": "linked", "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 16}'
    )
    (snap_dir / "model.safetensors").symlink_to(blob)

    m = next(x for x in discover(fake_hub) if x.repo == "acme/Linked")
    assert m.size == 5_000_000


def test_pooling_chat_model_keeps_embedding_capability(fake_hub):
    """A chat decoder that also ships `pooling` (an embedding endpoint) keeps
    BOTH capabilities — pins the explicit-precedence embedding heuristic."""
    sha = "c" * 40
    mdir = fake_hub / "models--acme--Hybrid"
    (mdir / "refs").mkdir(parents=True)
    (mdir / "refs" / "main").write_text(sha)
    snap = mdir / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        '{"architectures": ["HybridForCausalLM"], "model_type": "hybrid", "pooling": "last", '
        '"hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 16}'
    )
    (snap / "model.safetensors").write_bytes(b"\0" * 512)
    m = next(x for x in discover(fake_hub) if x.repo == "acme/Hybrid")
    assert Capability.COMPLETION in m.capabilities
    assert Capability.EMBEDDING in m.capabilities


def test_embed_substring_not_trusted_for_chat_decoders(fake_hub):
    """The loose 'embed' substring heuristic must not fire on a ForCausalLM
    chat decoder (the guard binds to that branch alone)."""
    sha = "d" * 40
    mdir = fake_hub / "models--acme--EmbedLite"
    (mdir / "refs").mkdir(parents=True)
    (mdir / "refs" / "main").write_text(sha)
    snap = mdir / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(
        '{"architectures": ["EmbedLiteForCausalLM"], "model_type": "embd", '
        '"hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 16}'
    )
    (snap / "model.safetensors").write_bytes(b"\0" * 512)
    m = next(x for x in discover(fake_hub) if x.repo == "acme/EmbedLite")
    assert Capability.COMPLETION in m.capabilities
    assert Capability.EMBEDDING not in m.capabilities


def test_unsupported_model_flagged(fake_hub):
    catalog = _catalog(fake_hub)
    wan = next(m for m in catalog if m.repo == "acme/WanVideo")
    assert wan.supported is False
    assert "wan" in wan.note


def test_by_name_variants(fake_hub):
    catalog = _catalog(fake_hub)
    # full name with tag
    assert by_name(catalog, "acme/TinyLM:main").repo == "acme/TinyLM"
    # full name without tag (resolves to the 'main' ref)
    assert by_name(catalog, "acme/TinyLM").repo == "acme/TinyLM"
    # unique bare name
    assert by_name(catalog, "TinyLM").repo == "acme/TinyLM"
    # unknown
    assert by_name(catalog, "nope/Nope") is None
