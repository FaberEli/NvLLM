"""Model metadata: one entry per discovered HuggingFace cache model."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Signals that mark a snapshot as a "System One decision model" — a small model
# trained to score options A/B/C and emit a probability distribution (the Router
# judge). Detection is deliberately **generic** and never tied to one specific
# local model, so it holds for decision models from any vendor and stays a clean
# negative for ordinary chat/embedding models. Two signals (either suffices):
#   1. a decision-marker file at the snapshot root — vendor file conventions:
#      StartLux-Decision ships ``decision_config.json``; some families ship
#      ``model_router.yaml``. Checked first (strongest, explicit).
#   2. fallback: the repo name carries the ``decision`` class keyword (StartLux /
#      Intern / SII all name them ``*-Decision``). Matching the *class keyword* —
#      not a specific repo — keeps it generic while catching families that ship
#      no marker file. ``nvllm router on --judge`` always overrides either.
DECISION_MARKER_FILES = ("decision_config.json", "model_router.yaml")
DECISION_NAME_KEYWORD = "decision"


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


def is_decision_snapshot(snapshot_dir: Path | str, repo: str | None = None) -> bool:
    """A snapshot is a decision model iff it ships a decision-marker file, or its
    repo name carries the ``decision`` class keyword (see module constants).

    Both signals are generic (a file convention / a semantic class keyword), so
    detection never depends on one specific local model and stays a clean negative
    for ordinary chat/embedding models.
    """
    snap = Path(snapshot_dir)
    if any((snap / marker).is_file() for marker in DECISION_MARKER_FILES):
        return True
    if repo and DECISION_NAME_KEYWORD in repo.lower():
        return True
    return False


def read_decision_info(snapshot_dir: Path | str, repo: str | None = None) -> dict:
    """Read routing-relevant metadata for a decision model.

    Tries the decision-marker files in order (``decision_config.json`` then
    ``model_router.yaml``) and degrades gracefully: a missing file, a missing
    ``pyyaml`` install, or an unparseable file all yield a minimal dict rather
    than raising — discovery and the router must not die on an odd checkpoint.
    Returns ``{}`` only when nothing marks the snapshot as a decision model.
    """
    snap = Path(snapshot_dir)

    # 1) JSON marker (StartLux "lodestar-letter-readout" decision models)
    json_path = snap / "decision_config.json"
    if json_path.is_file():
        base: dict = {"decision_model": True, "source": "decision_config.json"}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            doc = None
        if isinstance(doc, dict):
            if doc.get("format"):
                base["description"] = str(doc["format"])
            if doc.get("max_options_per_pass"):
                base["parameter_size"] = f"≤{doc['max_options_per_pass']} options/pass"
        return base

    # 2) YAML marker (other vendors)
    yaml_path = snap / "model_router.yaml"
    if yaml_path.is_file():
        base = {"decision_model": True, "source": "model_router.yaml"}
        try:
            import yaml  # local: keep pyyaml out of the module import path
        except ImportError:
            return base
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return base
        if isinstance(doc, dict):
            # the useful fields live either at the top level or under a
            # ``model_router:`` key (vendors differ); try the nested one first.
            inner = doc.get("model_router")
            if not isinstance(inner, dict):
                inner = doc
            base.update(
                {
                    k: str(v)
                    for k, v in {
                        "name": inner.get("name") or inner.get("model_name"),
                        "base_model": inner.get("base_model") or inner.get("base"),
                        "description": inner.get("description") or inner.get("model_description"),
                        "parameter_size": inner.get("parameter_size"),
                    }.items()
                    if v
                }
            )
        return base

    # 3) detected by repo-name keyword only — no marker-file metadata to read
    if repo and DECISION_NAME_KEYWORD in repo.lower():
        return {"decision_model": True, "source": f"name (contains '{DECISION_NAME_KEYWORD}')"}
    return {}
