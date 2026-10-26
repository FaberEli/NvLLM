"""Model metadata: one entry per discovered HuggingFace cache model."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# the on-disk file that marks a snapshot as a "System One decision model"
# (a small model trained to score options A/B/C and emit a distribution).
# It is a *file convention*, not a vendor name — so detecting on it stays
# generic and is never tied to one model family. Both StartLux-Decision and
# SII-Decision ship it; ordinary chat models do not (they ship lora_config.yaml).
DECISION_MODEL_YAML = "model_router.yaml"


class Capability(str, Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"
    TOOLS = "tools"
    THINKING = "thinking"


def _fmt_param_size(n: int) -> str:
    if n <= 0:
        return "unknown"
    b = n / 1e9
    if b >= 100:
        return f"{round(b)}B"
    if b >= 10:
        return f"{round(b)}B"
    return f"{b:.1f}B"


@dataclass
class ModelInfo:
    # identity
    repo: str  # "org/name"
    tag: str  # "latest" or ref name
    commit: str
    snapshot_dir: str  # absolute path handed to `vllm serve`
    # metadata
    size: int = 0  # bytes of weight files
    modified_at: datetime | None = None
    architectures: list[str] = field(default_factory=list)
    model_type: str = ""
    context_length: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    parameter_count: int = 0
    quantization: str = ""  # e.g. "nvfp4", "fp8"
    embedding_length: int | None = None
    capabilities: set[Capability] = field(default_factory=set)
    supported: bool = True
    note: str = ""  # why unsupported, if any
    # True if this snapshot is a System One *decision model* (ships a
    # model_router.yaml). Such models are candidates for the Router judge, not
    # for being routed to as a general chat model.
    is_decision_model: bool = False

    @property
    def name(self) -> str:
        return f"{self.repo}:{self.tag}"

    # ---- serialization -------------------------------------------------
    def _details(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "parent_model": "",
            "format": "safetensors" if self.supported else "unknown",
            "family": self.model_type,
            "families": [self.model_type] if self.model_type else [],
            "parameter_size": _fmt_param_size(self.parameter_count),
            "quantization_level": self.quantization,
            "decision_model": self.is_decision_model,
        }
        if self.context_length:
            d["context_length"] = self.context_length
        if self.embedding_length:
            d["embedding_length"] = self.embedding_length
        return d

    def to_ollama_list(self) -> dict[str, Any]:
        m = {
            "name": self.name,
            "model": self.name,
            "modified_at": (self.modified_at or datetime.now(timezone.utc)).isoformat(),
            "size": self.size,
            "digest": self.commit,
            "details": self._details(),
            "capabilities": sorted(c.value for c in self.capabilities),
        }
        if self.note:
            m["details"]["note"] = self.note
        return m

    def to_openai_model(self) -> dict[str, Any]:
        created = int((self.modified_at or datetime.now(timezone.utc)).timestamp())
        org = self.repo.split("/", 1)[0]
        return {
            "id": self.name,
            "object": "model",
            "created": created,
            "owned_by": org,
        }

    def to_show(self) -> dict[str, Any]:
        return {
            "details": self._details(),
            "capabilities": sorted(c.value for c in self.capabilities),
            "model_info": {
                "general.architecture": (self.architectures[0] if self.architectures else ""),
                "general.parameter_count": self.parameter_count,
                "general.size": f"{_fmt_param_size(self.parameter_count)} {self.quantization or ''}".strip(),
                f"{self.model_type or 'model'}.context_length": self.context_length,
                f"{self.model_type or 'model'}.embedding_length": self.embedding_length or 0,
            },
            "modified_at": (self.modified_at or datetime.now(timezone.utc)).isoformat(),
        }


def is_decision_snapshot(snapshot_dir: Path | str) -> bool:
    """A snapshot is a decision model iff it ships ``model_router.yaml``.

    This is the generic, vendor-neutral signal (see ``DECISION_MODEL_YAML``) —
    it never depends on a specific model name, so it holds for any decision
    model following that convention and stays a clean negative for ordinary
    chat/embedding models.
    """
    return (Path(snapshot_dir) / DECISION_MODEL_YAML).is_file()


def read_decision_info(snapshot_dir: Path | str) -> dict:
    """Read the routing-relevant fields from a decision model's
    ``model_router.yaml`` (name / base_model / description / parameter_size).

    Never raises: a missing file, a missing ``pyyaml`` install, or an unparseable
    file all degrade to a minimal ``{"decision_model": True, ...}`` (or ``{}``
    when the file is absent) — discovery and the router must not die on an odd
    checkpoint. ``pyyaml`` is imported lazily so the default path stays free of it.
    """
    snap = Path(snapshot_dir)
    yaml_path = snap / DECISION_MODEL_YAML
    if not yaml_path.is_file():
        return {}
    base = {"decision_model": True, "source": DECISION_MODEL_YAML}
    try:
        import yaml  # local: keep pyyaml out of the module import path
    except ImportError:
        return base
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return base
    if not isinstance(doc, dict):
        return base
    # the useful fields live either at the top level or under a ``model_router:``
    # key (vendors differ); try the nested one first, then the top level.
    inner = doc.get(DECISION_MODEL_YAML.rstrip(".yaml"))
    if not isinstance(inner, dict):
        inner = doc
    info = {
        "name": str(inner.get("name") or inner.get("model_name") or ""),
        "base_model": str(inner.get("base_model") or inner.get("base") or ""),
        "description": str(inner.get("description") or inner.get("model_description") or ""),
        "parameter_size": str(inner.get("parameter_size") or ""),
    }
    base.update({k: v for k, v in info.items() if v})
    return base
