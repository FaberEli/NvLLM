"""Router (M3, reworked M13): auto-select the best model for a request.

A small, always-resident *decision* model (the "judge") reads the request plus a
listing of the servable candidate models and chooses the one that best fits. The
service then forwards the request to that model. The judge is exempt from
eviction (see workers.WorkerManager), so it stays warm across requests.

Routing is triggered by a request whose model is the well-known alias
(`ROUTE_ALIAS`, "auto").

Two judge backends (selected by ``Settings.router_backend``):

- ``"vllm"`` (default): the judge is a ``vllm serve`` worker. The router asks it
  to answer a *choice* question (candidates listed as options A/B/C, …) and reads
  the model's probability distribution over the candidates from the response's
  ``top_logprobs`` — scoring the option symbols instead of parsing free text.
  This is the decision model's native "softmax over the allowed option symbols".
  It degrades gracefully: logprob scoring → parsing the generated token (a letter
  or a name) → the largest candidate, so a routing hiccup never fails the request.

- ``"transformers"``: the judge is loaded in-process via HuggingFace and read
  with one forward pass (see ``transformer_router``). Same scoring core, a more
  faithful (and heavier) inference path.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .catalog import ModelInfo
from .forward import forward_json
from .workers import Worker

log = logging.getLogger(__name__)

# model name that triggers routing, e.g. {"model": "auto"}
ROUTE_ALIAS = "auto"

# backend names (keep in sync with Settings.router_backend)
BACKEND_VLLM = "vllm"
BACKEND_TRANSFORMERS = "transformers"

# trailing quantization tags a model name may end with; stripped to get a base name
_QUANT_TAGS = {"NVFP4", "FP8", "FP16", "BF16", "FP4", "FP6", "AWQ", "GPTQ", "INT4", "INT8", "QAD", "MIXED"}

# system prompt for the judge; the "model router" phrasing is also what tests
# look for to recognize a routing call on the stub worker.
ROUTER_SYSTEM = (
    "You are a model router. From the candidate models listed, pick the ONE that "
    "best fits the request. Reply with only the exact model name and nothing else."
)

# choice-mode system prompt (decision models are trained to "pick one option")
CHOICE_SYSTEM = (
    "You are a model router. You are given a request and a list of candidate "
    "models, each on its own line prefixed by a single option letter (A, B, C, …). "
    "Choose the ONE model that best fits the request and reply with only that "
    "model's option letter — a single letter and nothing else."
)


def option_symbol(i: int) -> str:
    """0->'A', 1->'B', … 25->'Z', 26->'a', … 51->'z', 52->'0', … (up to 90 options)."""
    if i < 26:
        return chr(ord("A") + i)
    if i < 52:
        return chr(ord("a") + i - 26)
    return str(i - 52)


def softmax_topk(pairs: Sequence[tuple[int, float]]) -> tuple[int, list[float]]:
    """Numerically-stable softmax over a subset of log-scale values.

    ``pairs`` is a sequence of ``(candidate_index, log_value)`` — the log_value
    may be a raw logit (transformers backend) or an already-normalized logprob
    (vLLM backend); only relative differences matter, so both read the same way.
    Returns ``(best_index, probs)`` where ``probs`` is the distribution renormalized
    to sum to 1 over exactly the supplied pairs, in their input order. This is the
    shared "softmax over the allowed option symbols" the decision models are
    trained with, and it is pure so the scoring core is unit-testable without a GPU
    or torch.
    """
    vals = [float(v) for _, v in pairs]
    peak = max(vals)
    exps = [math.exp(v - peak) for v in vals]
    z = sum(exps)
    probs = [e / z for e in exps]
    best_pos = max(range(len(probs)), key=lambda i: probs[i])
    return pairs[best_pos][0], probs


@dataclass
class RouterDecision:
    """The outcome of a routing decision, including the model's distribution.

    ``distribution`` maps candidate name -> probability (a renormalized slice of
    the judge's softmax over the option symbols), so clients can see the
    confidence, not just the winner.
    """

    chosen: ModelInfo
    distribution: dict[str, float] = field(default_factory=dict)
    backend: str = ""  # "vllm-logprob" | "vllm-text" | "transformers" | "fallback-largest"
    raw: str = ""  # the judge's raw answer, for debugging / transparency


class Router:
    def __init__(
        self,
        judge: ModelInfo,
        candidates: list[ModelInfo],
        backend: str = BACKEND_VLLM,
        top_logprobs: int = 32,
    ) -> None:
        self.judge = judge
        # the judge is the router, not a routing target
        self.candidates = [m for m in candidates if m.name != judge.name]
        if not self.candidates:
            raise ValueError("router has no candidate models to route to")
        self.backend = backend
        # always cover every candidate's option letter (config value is a floor),
        # so a large candidate set does not silently drop options from the read
        self.top_logprobs = max(1, int(top_logprobs), len(self.candidates))

    # ---- prompt building -------------------------------------------------
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
        """Free-text prompt (kept for the ``vllm-text`` fallback path)."""
        listing = "\n".join(f"- {self._describe(m)}" for m in self.candidates)
        return [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"Candidate models:\n{listing}\n\nRequest:\n{request_text}"},
        ]

    def build_choice_prompt(self, request_text: str) -> list[dict]:
        """Choice-mode prompt: candidates as A/B/C options; the judge replies with
        a single option letter (its native "pick one" task)."""
        options = "\n".join(f"{option_symbol(i)}. {self._describe(m)}" for i, m in enumerate(self.candidates))
        allowed = ", ".join(option_symbol(i) for i in range(len(self.candidates)))
        user = (
            f"Candidate models (one per line, each prefixed by its option letter):\n"
            f"{options}\n\n"
            f"Request:\n{request_text}\n\n"
            f"Reply with exactly one of: {allowed} — the option letter of the best model."
        )
        return [
            {"role": "system", "content": CHOICE_SYSTEM},
            {"role": "user", "content": user},
        ]

    # ---- scoring ---------------------------------------------------------
    def _symbol_index(self) -> dict[str, int]:
        return {option_symbol(i): i for i in range(len(self.candidates))}

    def score_logprobs(self, top_logprobs: list[dict]) -> tuple[ModelInfo | None, dict[str, float]]:
        """Turn the judge's ``top_logprobs`` into ``(chosen, distribution)``.

        The returned values are already the model's normalized log-probabilities,
        so renormalizing them over the observed option letters reproduces the
        decision model's native "softmax over the allowed option symbols".
        Returns ``(None, {})`` when no candidate option letter is present, so the
        caller can fall back.
        """
        sym = self._symbol_index()
        logp: dict[int, float] = {}
        for entry in top_logprobs or []:
            tok = str((entry or {}).get("token", "")).strip()
            if tok in sym:
                v = (entry or {}).get("logprob")
                if isinstance(v, (int, float)):
                    logp[sym[tok]] = float(v)
        if not logp:
            return None, {}
        pairs = sorted(logp.items())  # (candidate_index, logprob), index-ordered
        best, probs = softmax_topk(pairs)
        dist = {self.candidates[pairs[i][0]].name: round(probs[i], 6) for i in range(len(pairs))}
        return self.candidates[best], dist

    def _parse_letter(self, text: str) -> ModelInfo | None:
        """Resolve a reply of the form 'A' / '(A)' / 'option A' to a candidate.

        Only a *whole token* that is exactly a valid option symbol counts, so a
        name like "acme/BigLM" is never mistaken for a letter.
        """
        if not text:
            return None
        sym = self._symbol_index()
        t = text.strip()
        if len(t) == 1 and t in sym:
            return self.candidates[sym[t]]
        for tok in re.split(r"\s+", t):
            clean = tok.strip("().,;: ")
            if clean in sym:
                return self.candidates[sym[clean]]
        return None

    def parse_choice_text(self, text: str) -> ModelInfo | None:
        """Free-text resolution: a bare option letter first, then a model name."""
        by_letter = self._parse_letter(text)
        if by_letter is not None:
            return by_letter
        return self.parse_choice(text)

    # ---- choice parsing (name-based) -------------------------------------
    def parse_choice(self, text: str) -> ModelInfo | None:
        """Resolve the judge's free-text reply to a candidate, or None.

        Preference order:
          1. exact whole string (name / repo / bare name);
          2. a candidate's full name or repo appearing anywhere (longest wins);
          3. a candidate's bare name as a whole token;
          4. a candidate's base name (bare name minus a trailing quant tag),
             matched only when exactly one candidate's base name appears — so a
             reply that mentions several models never mis-routes.
        """
        if not text:
            return None
        t = text.strip()
        low = t.lower()

        def word_hit(needle: str) -> bool:
            return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", low) is not None

        # 1) exact whole-string (name / repo / bare name)
        for m in self.candidates:
            if t in (m.name, m.repo, m.repo.split("/", 1)[-1]):
                return m
        # 2) a candidate's full name/repo appearing anywhere in the reply (longest wins)
        best, best_len = None, 0
        for m in self.candidates:
            for key in (m.name, m.repo):
                if key.lower() in low and len(key) > best_len:
                    best, best_len = m, len(key)
        if best is not None:
            return best
        # 3) bare model name as a whole token (word boundary)
        for m in self.candidates:
            if word_hit(m.repo.split("/", 1)[-1].lower()):
                return m
        # 4) base name (bare minus trailing quant tag) — only when unambiguous
        hits = [m for m in self.candidates if word_hit(self._base_name(m).lower())]
        if len(hits) == 1:
            return hits[0]
        return None

    @staticmethod
    def _base_name(m: ModelInfo) -> str:
        """Bare model name with a trailing quantization tag stripped, if present."""
        bare = m.repo.split("/", 1)[-1]
        parts = bare.rsplit("-", 1)
        if len(parts) == 2 and parts[1].upper() in _QUANT_TAGS:
            return parts[0]
        return bare

    def fallback(self) -> ModelInfo:
        # no valid judge answer -> route to the largest (most capable) model
        return max(self.candidates, key=lambda m: m.size or m.parameter_count or 0)

    # ---- the routing call (default vLLM backend) -------------------------
    async def decide(self, judge_worker: Worker, request_text: str) -> RouterDecision:
        """Route via the vLLM judge: choice prompt + logprob scoring, degrading to
        letter/name parsing, then to the largest candidate."""
        K = min(self.top_logprobs, 128)  # 128 option letters is far past any real candidate set
        try:
            resp = await forward_json(
                judge_worker,
                "/v1/chat/completions",
                {
                    "model": self.judge.name,
                    "messages": self.build_choice_prompt(request_text),
                    "max_tokens": 1,  # we only need the single decision token
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": K,
                },
            )
            choice = (resp.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content")) or ""
            content_lps = ((choice.get("logprobs") or {}).get("content")) or []
            top = (content_lps[0] or {}).get("top_logprobs") if content_lps else None
            if top:
                chosen, dist = self.score_logprobs(top)
                if chosen is not None:
                    return RouterDecision(chosen=chosen, distribution=dist, backend="vllm-logprob", raw=content)
            # fallback: the generated token itself (a letter, or a model name)
            chosen = self.parse_choice_text(content)
            if chosen is not None:
                return RouterDecision(chosen=chosen, backend="vllm-text", raw=content)
            log.warning("router judge (vllm) answer %r not resolvable; using fallback", content)
        except Exception as e:  # judge unreachable / errored -> don't fail the request
            log.warning("router judge (vllm) call failed (%s); using fallback model", e)
        return RouterDecision(chosen=self.fallback(), backend="fallback-largest")

    # back-compat alias (returns the chosen model, as the old `pick` did)
    async def pick(self, judge_worker: Worker, request_text: str) -> ModelInfo:
        return (await self.decide(judge_worker, request_text)).chosen
