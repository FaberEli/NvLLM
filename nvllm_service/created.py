"""Virtual (created) models — the Ollama `ollama create` counterpart.

A created model is pure metadata layered over a catalog model: a name, a base
model, an optional system prompt, and sampling options. It never touches
weights — serving it resolves to the base model's worker, and the wrapper
(system prompt / options) is injected into each request by the API layer.

Persisted as a single JSON file (``created.json``) in the state dir, so it
survives restarts and is trivially inspectable. Stdlib only, no I/O beyond the
JSON file, so it is unit-testable without a server.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# short human names (Ollama-style): letters/digits/dot/dash/underscore
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CreatedModel:
    name: str                      # e.g. "assistant"
    base: str                      # catalog name, e.g. "nvidia/Qwen3-8B-NVFP4:main"
    system: str = ""
    options: dict = field(default_factory=dict)  # temperature / top_p / num_predict
    created_at: str = field(default_factory=_now_iso)

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "base": self.base,
            "system": self.system,
            "options": dict(self.options),
            "created_at": self.created_at,
        }


class CreatedStore:
    """File-backed registry of created models (one JSON document, atomic writes)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ---- reads (per-call, so external edits are picked up) ----------------
    def _read(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        models = doc.get("models") if isinstance(doc, dict) else None
        return [m for m in (models or []) if isinstance(m, dict) and m.get("name")]

    def list(self) -> list[CreatedModel]:
        return [self._from_doc(m) for m in self._read()]

    def get(self, name: str) -> CreatedModel | None:
        for m in self._read():
            if m["name"] == name:
                return self._from_doc(m)
        return None

    def names(self) -> set[str]:
        return {m["name"] for m in self._read()}

    # ---- writes ------------------------------------------------------------
    def create(self, cm: CreatedModel) -> None:
        if not _NAME_RE.match(cm.name):
            raise ValueError(
                f"invalid name '{cm.name}' — use letters, digits, '.', '-', '_' (max 64 chars)"
            )
        if not cm.base:
            raise ValueError("a created model needs a base model (the catalog model it wraps)")
        if cm.name in self.names():
            raise ValueError(f"created model '{cm.name}' already exists")
        self._write(self._read() + [asdict(cm)])

    def delete(self, name: str) -> bool:
        models = self._read()
        kept = [m for m in models if m["name"] != name]
        if len(kept) == len(models):
            return False
        self._write(kept)
        return True

    def _write(self, models: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"models": models, "updated_at": _now_iso()}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".created-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, self.path)  # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _from_doc(m: dict) -> CreatedModel:
        opts = m.get("options")
        return CreatedModel(
            name=str(m["name"]),
            base=str(m.get("base") or ""),
            system=str(m.get("system") or ""),
            options=dict(opts) if isinstance(opts, dict) else {},
            created_at=str(m.get("created_at") or ""),
        )
