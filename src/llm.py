"""Thin wrapper around a local Mistral 7B model served by Ollama.

Ollama (https://ollama.com) is the simplest open-source way to run Mistral 7B
locally with no API key and no cloud dependency:

    ollama pull mistral
    ollama serve      # usually already running as a background service

If Ollama isn't reachable, every caller falls back to a deterministic
template explanation so the rest of the demo still runs end-to-end offline.
"""

from __future__ import annotations

import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("FRAUD_AGENT_MODEL", "mistral")
TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT", "20"))


class LLMUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def generate(prompt: str, system: str | None = None, max_tokens: int = 200) -> str:
    """Single-shot completion from the local Mistral model.

    Raises LLMUnavailable if Ollama can't be reached; callers should catch
    this and fall back to a rule-based explanation.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Could not reach Ollama/Mistral at {OLLAMA_HOST}: {exc}") from exc


def explain_decision(transaction, action: str, risk_score: float, reasons: list[str]) -> str:
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
        text = generate(prompt, system=system, max_tokens=80)
        if text:
            return text
    except LLMUnavailable:
        pass
    return f"{action} (risk={risk_score:.2f}) based on: {reasons_str}."
