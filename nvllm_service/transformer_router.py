"""In-process judge backend: HuggingFace one-forward-pass decision readout (M13).

The "transformers" router backend loads the decision model directly (via
``transformers`` + ``torch``) and reads its choice with a SINGLE forward pass —
the native "softmax over the allowed option symbols" the decision models are
trained with — instead of token generation. It is a more faithful (and heavier)
path than the default "vllm" backend, and is OPTIONAL: it is only imported and
used when ``Settings.router_backend == "transformers"``.

The scoring semantics are shared with ``router.Router`` (the choice prompt lists
candidates as option letters A/B/C, …; the distribution is the renormalized
softmax over those option symbols), so both backends agree on how a distribution
becomes a choice.

Notes / trade-offs
------------------
- ``transformers`` and ``torch`` are imported lazily (inside methods), so the
  "vllm" backend — and the whole service's thin-orchestrator footprint — never
  pulls them in.
- The readout reads the logits at the last prompt position (the position that
  predicts the first completion token, i.e. the option letter) and softmaxes over
  the option symbols' ids. This is the general form of the decision models'
  native readout; the exact per-vendor scaffolding (e.g. a ``<decision>``
  placeholder or an empty thinking block) is model-specific and, where present,
  is honoured by using the checkpoint's own ``apply_chat_template``.
- This backend is best on causal decision variants (Intern-Decision, StartLux
  *-Instant). It needs a transformers build that understands the model's
  architecture (qwen3_5); the default "vllm" backend is the more broadly
  supported path.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from .router import Router, RouterDecision, option_symbol, softmax_topk

log = logging.getLogger(__name__)


class TransformerJudge:
    """Loads the decision model once (lazy, thread-safe) and scores the choice
    per request with one forward pass."""

    def __init__(self, router: Router) -> None:
        self.router = router
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device: str = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # recheck under the lock
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "router backend 'transformers' requires the 'transformers' and 'torch' packages, "
                    f"which are not installed: {e}\n"
                    "  install them in the service environment, or use the default backend:\n"
                    "      nvllm serve --router-backend vllm   # judge served via vllm serve + logprob scoring"
                ) from e

            info = self.router.judge
            log.info("loading decision-model judge in-process: %s (snapshot=%s)", info.name, info.snapshot_dir)
            tokenizer = AutoTokenizer.from_pretrained(info.snapshot_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                info.snapshot_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            log.info("decision-model judge ready on %s", device)

    def _option_token_ids(self) -> list[int]:
        """Token id for each candidate's option symbol (A, B, C, …).

        Option symbols are single tokens in Qwen-family tokenizers; if one is not,
        we fall back to its vocab id and warn rather than silently mis-scoring.
        """
        enc = self._tokenizer
        ids: list[int] = []
        for i in range(len(self.router.candidates)):
            sym = option_symbol(i)
            t = enc(sym, add_special_tokens=False)
            input_ids = t.get("input_ids") or []
            if len(input_ids) == 1:
                ids.append(int(input_ids[0]))
            else:
                vid = enc.convert_tokens_to_ids(sym)
                if i == 0:
                    log.warning("option symbol %r is not a single token; using vocab id %s (scoring may be off)", sym, vid)
                ids.append(int(vid))
        return ids

    def _decide_sync(self, request_text: str) -> RouterDecision:
        import torch  # local: keep torch out of the module import path

        router = self.router
        messages = router.build_choice_prompt(request_text)
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        opt_ids = self._option_token_ids()

        with torch.inference_mode():
            logits = self._model(**inputs).logits[0, -1, :]  # position predicting the decision token

        # softmax over the allowed option symbols only (the native readout)
        lps = logits[opt_ids].float()
        pairs = list(enumerate(lps.tolist()))  # (candidate_index, logit), all candidates in order
        best, probs = softmax_topk(pairs)
        dist = {router.candidates[i].name: round(p, 6) for i, p in enumerate(probs)}
        return RouterDecision(chosen=router.candidates[best], distribution=dist, backend="transformers")

    async def decide(self, request_text: str) -> RouterDecision:
        """Score the choice; blocking load/forward run in a worker thread so the
        event loop stays free for other requests."""
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._decide_sync, request_text)
