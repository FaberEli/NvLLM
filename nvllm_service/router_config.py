"""Router activation state — the persisted "router on/off + chosen judge".

The router is an *opt-in, resident* capability: the user activates it once
(``nvllm router on``), the choice is written to a single JSON file (``router.json``)
in the state dir, and every later ``nvllm serve`` reads it at startup — the judge
decision model then becomes resident and the ``AutoRouter`` option appears. This
mirrors :class:`nvllm_service.created.CreatedStore` (one JSON document, atomic
tmp+rename writes, per-call reads) so it is stdlib-only and unit-testable
without a server.

Precedence at startup (see ``main.create_app``): an explicit ``--router-judge``
CLI/env flag wins (0.5.0 back-compat), then this persisted store (when
``enabled``), then the router is off.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .router import BACKEND_VLLM


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = True
    judge: str = ""  # catalog name, e.g. "startlux/StartLux-Decision-v1.1-4B-Instant:main"
    candidates: list[str] = field(default_factory=list)  # optional whitelist of repos; empty = all
    backend: str = BACKEND_VLLM  # "vllm" | "transformers"
    top_logprobs: int = 32
    activated_at: str = field(default_factory=_now_iso)

    def to_api(self) -> dict:
        return {
            "enabled": self.enabled,
            "judge": self.judge,
            "candidates": list(self.candidates),
            "backend": self.backend,
            "top_logprobs": self.top_logprobs,
            "activated_at": self.activated_at,
        }


class RouterConfigStore:
    """File-backed router-activation state (one JSON document, atomic writes)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ---- reads (per-call, so an external `nvllm router` edit is picked up) --
    def load(self) -> RouterConfig | None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(doc, dict) or not doc.get("enabled"):
            return None
        judge = str(doc.get("judge") or "")
        if not judge:
            return None
        candidates = doc.get("candidates")
        try:
            top_logprobs = int(doc.get("top_logprobs") or 32)
        except (TypeError, ValueError):
            top_logprobs = 32  # a corrupt value must not take the server down
        return RouterConfig(
            enabled=True,
            judge=judge,
            candidates=[c for c in (candidates or []) if isinstance(c, str) and c.strip()],
            backend=str(doc.get("backend") or BACKEND_VLLM),
            top_logprobs=max(1, top_logprobs),
            activated_at=str(doc.get("activated_at") or ""),
        )

    # ---- writes -----------------------------------------------------------
    def save(self, cfg: RouterConfig) -> None:
        if not cfg.judge:
            raise ValueError("router activation needs a judge model (the decision model)")
        if cfg.backend not in ("vllm", "transformers"):
            raise ValueError(f"unknown router backend '{cfg.backend}' (expected 'vllm' or 'transformers')")
        self._write(cfg.to_api())

    def clear(self) -> bool:
        """Deactivate: remove the file. Returns True if it was present."""
        try:
            os.remove(self.path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def _write(self, doc: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".router-", suffix=".tmp")
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

    # convenience: build a config from a catalog name + optional narrowing
    @staticmethod
    def activate(judge: str, candidates: list[str] | None = None, backend: str = BACKEND_VLLM, top_logprobs: int = 32) -> RouterConfig:
        return RouterConfig(
            enabled=True,
            judge=judge,
            candidates=[c.strip() for c in (candidates or []) if c and c.strip()],
            backend=backend,
            top_logprobs=max(1, int(top_logprobs)),
        )
