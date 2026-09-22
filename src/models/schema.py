"""Core data structures shared by every agent."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class Action(str, Enum):
    APPROVE = "APPROVE"
    DECLINE = "DECLINE"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str  # ISO-8601
    user_id: str
    card_id: str
    amount: float
    currency: str
    merchant: str
    merchant_category: str
    country: str
    user_home_country: str
    device_id: str
    ip_address: str
    channel: str  # "online" | "pos" | "atm"
    is_fraud: bool  # ground-truth label, used only for evaluation/training

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    agent_name: str
    transaction_id: str
    action: Action
    risk_score: float  # 0.0 - 1.0
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""  # optional natural-language narrative (LLM-generated)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d
