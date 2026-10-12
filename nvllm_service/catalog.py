"""Model metadata: one entry per discovered HuggingFace cache model."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


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
                f"{self.model_type}.context_length": self.context_length,
                f"{self.model_type}.embedding_length": self.embedding_length or 0,
            },
            "modified_at": (self.modified_at or datetime.now(timezone.utc)).isoformat(),
        }
