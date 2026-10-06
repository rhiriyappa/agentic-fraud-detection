"""Thin wrapper around a local Mistral 7B model served by Ollama.

Ollama (https://ollama.com) is the simplest open-source way to run Mistral 7B
locally with no API key and no cloud dependency:

    ollama pull mistral
    ollama serve      # usually already running as a background service

If Ollama isn't reachable, every caller falls back to a deterministic
template explanation so the rest of the demo still runs end-to-end offline.

Observability: every call is counted in process-wide `LLMStats` (see
`get_stats()`) and logged via the "fraud.llm" logger. Successful responses are
cached in SQLite keyed by a hash of the exact request, so repeating a run
costs zero Mistral calls.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict

import requests

from src import db

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("FRAUD_AGENT_MODEL", "mistral")
TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT", "20"))
TEMPERATURE = 0.2

logger = logging.getLogger("fraud.llm")


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMStats:
    requests: int = 0          # HTTP calls actually sent to Ollama
    ok: int = 0
    errors: int = 0
    cache_hits: int = 0        # answered from SQLite, no request sent
    fallbacks: int = 0         # explanation fell back to the template
    total_latency_s: float = 0.0

    def minus(self, earlier: "LLMStats") -> "LLMStats":
        return LLMStats(**{k: v - getattr(earlier, k) for k, v in asdict(self).items()})

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.ok if self.ok else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["avg_latency_s"] = self.avg_latency_s
        return d


_stats = LLMStats()
_stats_lock = threading.Lock()


def get_stats() -> LLMStats:
    with _stats_lock:
        return LLMStats(**asdict(_stats))


def reset_stats() -> None:
    global _stats
    with _stats_lock:
        _stats = LLMStats()


def is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _cache_key(prompt: str, system: str | None, max_tokens: int) -> str:
    raw = "\x1f".join([MODEL_NAME, str(TEMPERATURE), str(max_tokens), system or "", prompt])
    return hashlib.sha256(raw.encode()).hexdigest()


def generate(prompt: str, system: str | None = None, max_tokens: int = 200,
             use_cache: bool = True) -> str:
    """Single-shot completion from the local Mistral model.

    Raises LLMUnavailable if Ollama can't be reached; callers should catch
    this and fall back to a rule-based explanation.
    """
    key = _cache_key(prompt, system, max_tokens)
    if use_cache:
        cached = db.get_llm_cache(key)
        if cached is not None:
            with _stats_lock:
                _stats.cache_hits += 1
            logger.info("llm cache hit key=%s", key[:8])
            return cached

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": TEMPERATURE},
    }
    if system:
        payload["system"] = system

    with _stats_lock:
        _stats.requests += 1
    start = time.perf_counter()
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except requests.RequestException as exc:
        with _stats_lock:
            _stats.errors += 1
        logger.warning("llm request failed after %.2fs: %s", time.perf_counter() - start, exc)
        raise LLMUnavailable(f"Could not reach Ollama/Mistral at {OLLAMA_HOST}: {exc}") from exc

    latency = time.perf_counter() - start
    with _stats_lock:
        _stats.ok += 1
        _stats.total_latency_s += latency
    logger.info("llm request ok model=%s latency=%.2fs prompt_chars=%d",
                MODEL_NAME, latency, len(prompt))

    if use_cache and text:
        db.put_llm_cache(key, MODEL_NAME, text)
    return text


def explain_decision(transaction, action: str, risk_score: float, reasons: list[str],
                     use_cache: bool = True) -> str:
    """Ask Mistral for a short analyst-style narrative; fall back to a template."""
    reasons_str = "; ".join(reasons) if reasons else "no specific risk signals"
    system = (
        "You are a fraud-analyst assistant. Write a single, concise sentence "
        "(max 30 words) explaining a payment decision to a human analyst. "
        "Be factual and specific, no preamble."
    )
    prompt = (
        f"Transaction {transaction.transaction_id}: ${transaction.amount:.2f} "
        f"{transaction.currency} at {transaction.merchant} ({transaction.merchant_category}) "
        f"in {transaction.country}, channel={transaction.channel}. "
        f"Decision: {action}. Risk score: {risk_score:.2f}. "
        f"Signals: {reasons_str}. Explain why this decision makes sense."
    )
    try:
        text = generate(prompt, system=system, max_tokens=80, use_cache=use_cache)
        if text:
            return text
    except LLMUnavailable:
        pass
    with _stats_lock:
        _stats.fallbacks += 1
    return f"{action} (risk={risk_score:.2f}) based on: {reasons_str}."
