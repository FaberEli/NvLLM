"""Scan the HuggingFace hub cache and build a model catalog.

Layout scanned (HF hub standard)::

    <hub_root>/models--<org>--<name>/
        refs/            one file per ref (main, tags/...); file content = commit sha
        snapshots/<sha>/ config.json, tokenizer_config.json, *.safetensors, ...
        blobs/           actual weight blobs (snapshots entries are symlinks)

Only the standard library is used here, so discovery can run without vLLM
installed.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .catalog import Capability, ModelInfo

log = logging.getLogger(__name__)

# model_type / architecture fragments that vLLM cannot (or should not) serve as a
# text/embedding engine -> listed but marked unsupported.
_UNSUPPORTED_TYPES = ("wan", "diffusers", "stable_diffusion", "cogvideox", "lumina")
_UNSUPPORTED_ARCHS = ("WanModel", "WanForTextToVideo", "Diffusion", "CogVideoX")

# heuristic: these model_type families are known embedding/pooling architectures
_EMBED_HINT_TYPES = ("nllb", "bert", "xlm-roberta", "gte", "bge", "ministral3", "jina")


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _refs_of(model_dir: Path) -> dict[str, str]:
    """ref name -> commit sha. 'main' maps to tag 'latest'."""
    refs: dict[str, str] = {}
    refs_dir = model_dir / "refs"
    if not refs_dir.is_dir():
        return refs
    for p in sorted(refs_dir.rglob("*")):
        if not p.is_file():
            continue
        try:
            sha = p.read_text().strip()
        except OSError:
            continue
        if not re.fullmatch(r"[0-9a-f]{7,64}", sha):
            continue
        ref = p.relative_to(refs_dir).as_posix()
        refs["main" if ref == "main" else ref.lstrip("tags/")] = sha
    return refs


def _weight_size(snapshot: Path) -> int:
    total = 0
    for ext in ("*.safetensors", "*.bin", "*.gguf", "*.pt", "*.pth"):
        for p in snapshot.glob(ext):
            if p.is_symlink() or p.is_file():
                try:
                    total += p.lstat().st_size
                except OSError:
                    pass
    return total


def _estimate_parameters(cfg: dict) -> int:
    """Rough parameter count for display (Ollama 'parameter_size' equivalent)."""
    if isinstance(cfg.get("num_parameters"), int):
        return int(cfg["num_parameters"])
    h = cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size")
    L = cfg.get("num_hidden_layers") or (cfg.get("text_config") or {}).get("num_hidden_layers")
    if not h or not L:
        return 0
    n = 12 * h * h * L  # standard dense transformer estimate
    moe = cfg.get("num_experts") or (cfg.get("text_config") or {}).get("num_experts")
    moe_int = cfg.get("moe_intermediate_size") or (cfg.get("text_config") or {}).get("moe_intermediate_size")
    if moe and moe_int:
        n += 3 * h * moe_int * moe * L
    vocab = cfg.get("vocab_size")
    if vocab:
        n += vocab * h
    return int(n)


def _quantization(cfg: dict) -> str:
    q = cfg.get("quantization_config") or {}
    algo = q.get("quant_algo") or q.get("quant_method") or ""
    return str(algo).lower()


def _detect_capabilities(cfg: dict, snapshot: Path) -> set[Capability]:
    caps: set[Capability] = set()
    archs = [str(a) for a in cfg.get("architectures", [])]
    mtype = str(cfg.get("model_type", "")).lower()
    any_arch = " ".join(archs).lower()

    if any(a.endswith("ForCausalLM") or a.endswith("ForConditionalGeneration") for a in archs):
        caps.add(Capability.COMPLETION)

    # pooling / embedding model?
    if (
        cfg.get("pooling")
        or cfg.get("is_causal") is False
        or mtype in _EMBED_HINT_TYPES
        or ("embed" in mtype or "embed" in any_arch)
        and not any(a.endswith("ForCausalLM") for a in archs)
    ):
        caps.add(Capability.EMBEDDING)

    if cfg.get("vision_config") or "vision" in any_arch or "_vl" in mtype or mtype.endswith("_vl"):
        caps.add(Capability.VISION)
    if cfg.get("audio_config") or "audio" in any_arch or "whisper" in any_arch:
        caps.add(Capability.AUDIO)

    # tools: model ships a chat template (heuristic)
    tok_cfg = _load_json(snapshot / "tokenizer_config.json") or {}
    if tok_cfg.get("chat_template") or (tok_cfg.get("chat_template_jinja") is not None):
        if Capability.COMPLETION in caps:
            caps.add(Capability.TOOLS)

    # thinking: name/type heuristic
    if "reason" in mtype or "thinking" in mtype or "reason" in any_arch:
        caps.add(Capability.THINKING)

    return caps


def _is_unsupported(cfg: dict) -> str | None:
    mtype = str(cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    for t in _UNSUPPORTED_TYPES:
        if t in mtype:
            return f"model_type '{mtype}' is not servable by vLLM"
    for a in _UNSUPPORTED_ARCHS:
        if any(a.lower() in x for x in archs):
            return f"architecture '{a}' is not servable by vLLM"
    # diffusion / image-gen checkpoints usually lack a tokenizer
    return None


def _scan_one(model_dir: Path, hub_root: Path) -> list[ModelInfo]:
    m = model_dir.name
    if not m.startswith("models--"):
        return []
    inner = m[len("models--"):]
    if "--" not in inner:
        return []
    org, repo_name = inner.split("--", 1)
    repo = f"{org}/{repo_name}"

    refs = _refs_of(model_dir)
    if not refs:
        # no refs -> try the snapshots dir directly
        snap_root = model_dir / "snapshots"
        if not snap_root.is_dir():
            return []
        refs = {
            "latest": s.name
            for s in sorted(snap_root.iterdir())
            if s.is_dir() and re.fullmatch(r"[0-9a-f]{7,64}", s.name)
        }
    if not refs:
        return []

    results: list[ModelInfo] = []
    for tag, sha in refs.items():
        snapshot = model_dir / "snapshots" / sha
        if not snapshot.is_dir():
            continue
        cfg = _load_json(snapshot / "config.json") or {}
        text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        hidden = cfg.get("hidden_size") or text_cfg.get("hidden_size") or 0
        ctx = cfg.get("max_position_embeddings") or text_cfg.get("max_position_embeddings") or 0

        caps = _detect_capabilities(cfg, snapshot)
        unsupported = _is_unsupported(cfg)
        if unsupported is None and not caps:
            unsupported = "could not determine any capability (no recognizable architecture)"
        if unsupported is None and not cfg.get("architectures"):
            unsupported = "no architectures in config.json (not a vLLM-servable model)"

        modified: datetime | None = None
        try:
            st = snapshot.stat()
            modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        except OSError:
            pass

        info = ModelInfo(
            repo=repo,
            tag=tag,
            commit=sha,
            snapshot_dir=str(snapshot),
            size=_weight_size(snapshot),
            modified_at=modified,
            architectures=[str(a) for a in cfg.get("architectures", [])],
            model_type=str(cfg.get("model_type", "")),
            context_length=int(ctx),
            hidden_size=int(hidden),
            vocab_size=int(cfg.get("vocab_size") or text_cfg.get("vocab_size") or 0),
            parameter_count=_estimate_parameters({**text_cfg, **{k: v for k, v in cfg.items() if k not in text_cfg}}),
            quantization=_quantization(cfg) or _quantization(text_cfg),
            embedding_length=int(hidden) if Capability.EMBEDDING in caps else None,
            capabilities=caps,
            supported=unsupported is None,
            note=unsupported or "",
        )
        results.append(info)
    return results


def discover(hub_root: Path) -> list[ModelInfo]:
    """Scan <hub_root>/models--* and return the catalog, newest first."""
    hub_root = Path(hub_root).expanduser()
    models: list[ModelInfo] = []
    if not hub_root.is_dir():
        log.warning("HF hub root not found: %s", hub_root)
        return models
    for d in sorted(hub_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("models--"):
            continue
        try:
            models.extend(_scan_one(d, hub_root))
        except Exception:
            log.exception("failed to scan %s", d)
    models.sort(key=lambda m: (m.modified_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return models


def by_name(catalog: list[ModelInfo], name: str) -> ModelInfo | None:
    """Resolve a name like 'org/name', 'org/name:tag' or 'name' (unique suffix match).

    Without a tag (or with the conventional 'latest' tag) any ref of the repo
    matches — HF caches usually only have 'main' — preferring main/latest.
    """
    name = name.strip()
    repo, _, tag = name.partition(":")
    if not tag:
        tag = "latest"

    def repo_match(m: ModelInfo) -> bool:
        # bare-name queries (e.g. "Llama-3.3-70B") match on the repo's last path segment
        return m.repo == repo or m.repo.split("/", 1)[-1] == repo

    pool = [m for m in catalog if repo_match(m) and (m.tag == tag or tag == "latest")]
    if pool:
        if tag == "latest":
            for pref in ("main", "latest"):
                for m in pool:
                    if m.tag == pref:
                        return m
            return pool[0]
        return pool[0]
    # case-insensitive fallback
    low = name.lower()
    ci = [m for m in catalog if m.name.lower() == low]
    if len(ci) == 1:
        return ci[0]
    return None
