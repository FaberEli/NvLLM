"""Catalog discovery and name resolution over a fake HF hub cache."""
from nvllm_service.catalog import Capability
from nvllm_service.discovery import by_name, discover


def _catalog(fake_hub):
    return discover(fake_hub)


def test_catalog_entries(fake_hub):
    catalog = _catalog(fake_hub)
    names = sorted(m.name for m in catalog)
    assert names == ["acme/Embedder:main", "acme/TinyLM:main", "acme/WanVideo:main"]


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
