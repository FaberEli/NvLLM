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
