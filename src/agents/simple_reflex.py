"""Simple Reflex Agent
======================
Use case: point-of-sale / card present hard limit gatekeeper.

A simple reflex agent maps the *current percept only* to an action via
condition action rules. It has no memory of past transactions and does not
model the world - exactly the kind of instant "swipe the card" decision a
terminal must make in milliseconds, before any history lookup is even
possible.

Rules are intentionally simple and fully explainable: hard amount ceilings,
a merchant category blocklist, and a country blocklist.
"""

from __future__ import annotations

from src.models import Transaction, Decision, Action
from src.agents.base import Agent

HARD_DECLINE_AMOUNT = 5000.00
STEP_UP_AMOUNT = 1500.00

HIGH_RISK_CATEGORIES = {"crypto_exchange", "gambling", "money_transfer", "gift_cards"}
BLOCKED_COUNTRIES = {"KP", "IR", "SY"}  # sanctioned-country demo blocklist


class SimpleReflexAgent(Agent):
    name = "simple_reflex"

    def perceive(self, transaction: Transaction) -> dict:
        # No history, no state -- the percept IS the transaction.
        return {"transaction": transaction}

    def decide(self, percept: dict) -> Decision:
        txn: Transaction = percept["transaction"]
        reasons: list[str] = []
        action = Action.APPROVE
        risk_score = 0.05

        if txn.country in BLOCKED_COUNTRIES:
            reasons.append(f"country '{txn.country}' is on the sanctions blocklist")
            action = Action.DECLINE
            risk_score = 0.99

        elif txn.amount >= HARD_DECLINE_AMOUNT:
            reasons.append(f"amount {txn.amount:.2f} >= hard ceiling {HARD_DECLINE_AMOUNT:.2f}")
            action = Action.DECLINE
            risk_score = 0.9

        elif txn.merchant_category in HIGH_RISK_CATEGORIES:
            reasons.append(f"merchant category '{txn.merchant_category}' is high-risk")
            action = Action.STEP_UP_AUTH
            risk_score = 0.6

        elif txn.amount >= STEP_UP_AMOUNT:
            reasons.append(f"amount {txn.amount:.2f} >= step-up threshold {STEP_UP_AMOUNT:.2f}")
            action = Action.STEP_UP_AUTH
            risk_score = 0.4

        else:
            reasons.append("no rule triggered")

        return Decision(
            agent_name=self.name,
            transaction_id=txn.transaction_id,
            action=action,
            risk_score=risk_score,
            reasons=reasons,
        )
