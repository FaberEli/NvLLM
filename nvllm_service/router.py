"""Router (M3): auto-select the best model from the catalog for a request.

A small, always-resident "judge" model reads the request plus a listing of the
servable candidate models and returns the name of the one that best fits. The
service then forwards the request to that model. The judge is exempt from
eviction (see workers.WorkerManager), so it stays warm across requests.

Routing is triggered by a request whose model is the well-known alias
(`ROUTE_ALIAS`, "auto"). If the judge can't be consulted or returns something
unrecognizable, the router falls back to the largest candidate rather than
failing the request.
"""
from __future__ import annotations

import logging
import re

from .catalog import ModelInfo
from .forward import forward_json
from .workers import Worker

log = logging.getLogger(__name__)

# model name that triggers routing, e.g. {"model": "auto"}
ROUTE_ALIAS = "auto"

# system prompt for the judge; the "model router" phrasing is also what tests
# look for to recognize a routing call on the stub worker.
ROUTER_SYSTEM = (
    "You are a model router. From the candidate models listed, pick the ONE that "
    "best fits the request. Reply with only the exact model name and nothing else."
)


class Router:
    def __init__(self, judge: ModelInfo, candidates: list[ModelInfo]) -> None:
        self.judge = judge
        # the judge is the router, not a routing target
        self.candidates = [m for m in candidates if m.name != judge.name]
        if not self.candidates:
            raise ValueError("router has no candidate models to route to")

    # ---- candidate listing -------------------------------------------
    @staticmethod
    def _describe(m: ModelInfo) -> str:
        parts = [m.repo]
        if m.parameter_count:
            parts.append(f"{m.parameter_count / 1e9:.0f}B")
        if m.quantization:
            parts.append(m.quantization)
        caps = sorted(c.value for c in m.capabilities)
        if caps:
            parts.append("+".join(caps))
        return f"{m.name} ({', '.join(parts)})"

    def build_route_prompt(self, request_text: str) -> list[dict]:
        listing = "\n".join(f"- {self._describe(m)}" for m in self.candidates)
        return [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"Candidate models:\n{listing}\n\nRequest:\n{request_text}"},
        ]

    # ---- choice parsing ----------------------------------------------
    def parse_choice(self, text: str) -> ModelInfo | None:
        """Resolve the judge's free-text reply to a candidate, or None.

        Prefers a verbatim `org/name[:tag]` (specific enough to substring-match
        safely); falls back to a bare model name matched as a whole token so a
        short name never false-matches inside an unrelated word.
        """
        if not text:
            return None
        t = text.strip()
        low = t.lower()
        # exact whole-string (name / repo / bare name)
        for m in self.candidates:
            if t in (m.name, m.repo, m.repo.split("/", 1)[-1]):
                return m
        # a candidate's full name/repo appearing anywhere in the reply (longest wins)
        best, best_len = None, 0
        for m in self.candidates:
            for key in (m.name, m.repo):
                if key.lower() in low and len(key) > best_len:
                    best, best_len = m, len(key)
        if best is not None:
            return best
        # bare model name as a whole token (word boundary)
        for m in self.candidates:
            bare = m.repo.split("/", 1)[-1].lower()
            if re.search(r"(?<![a-z0-9])" + re.escape(bare) + r"(?![a-z0-9])", low):
                return m
        return None

    def fallback(self) -> ModelInfo:
        # no valid judge answer -> route to the largest (most capable) model
        return max(self.candidates, key=lambda m: m.size or m.parameter_count or 0)

    # ---- the actual routing call -------------------------------------
    async def pick(self, judge_worker: Worker, request_text: str) -> ModelInfo:
        try:
            resp = await forward_json(
                judge_worker,
                "/v1/chat/completions",
                {
                    "model": self.judge.name,
                    "messages": self.build_route_prompt(request_text),
                    "max_tokens": 48,
                    "temperature": 0,
                },
            )
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception as e:  # judge unreachable / errored -> fall back, don't fail the request
            log.warning("router judge call failed (%s); using fallback model", e)
            return self.fallback()
        chosen = self.parse_choice(content)
        if chosen is None:
            log.warning("router judge returned unrecognized model %r; using fallback", content)
            return self.fallback()
        return chosen
