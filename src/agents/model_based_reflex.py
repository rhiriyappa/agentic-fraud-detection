"""Model-Based Reflex Agent
============================
Use case: card-velocity & impossible-travel detection.

Unlike the simple reflex agent, this agent keeps an internal *model of the
world* -- here, each card's recent transaction history pulled from SQLite --
because the current transaction alone can't reveal a burst of rapid-fire
purchases or a purchase that "teleported" across countries in minutes.
The percept is therefore (current transaction + reconstructed state), and
the condition-action rules fire against that reconstructed state.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.models import Transaction, Decision, Action
from src.agents.base import Agent
from src import db

VELOCITY_WINDOW_MINUTES = 10
VELOCITY_COUNT_THRESHOLD = 3
SPEND_SPIKE_MULTIPLIER = 4.0
IMPOSSIBLE_TRAVEL_WINDOW_MINUTES = 60


class ModelBasedReflexAgent(Agent):
    name = "model_based_reflex"

    def perceive(self, transaction: Transaction) -> dict:
        history = db.fetch_user_history(transaction.user_id, transaction.timestamp, limit=20)
        return {"transaction": transaction, "history": history}

    def decide(self, percept: dict) -> Decision:
        txn: Transaction = percept["transaction"]
        history: list[Transaction] = percept["history"]
        reasons: list[str] = []
        risk_score = 0.05
        action = Action.APPROVE

        now = datetime.fromisoformat(txn.timestamp)

        # --- update internal model: recent window & rolling average spend ---
        recent = [h for h in history if now - datetime.fromisoformat(h.timestamp)
                  <= timedelta(minutes=VELOCITY_WINDOW_MINUTES)]
        rolling_avg = (sum(h.amount for h in history) / len(history)) if history else txn.amount
        last_txn = history[0] if history else None

        # --- rule 1: velocity ---
        if len(recent) >= VELOCITY_COUNT_THRESHOLD:
            reasons.append(
                f"{len(recent)} transactions in the last {VELOCITY_WINDOW_MINUTES} min "
                f"(threshold {VELOCITY_COUNT_THRESHOLD})"
            )
            risk_score = max(risk_score, 0.75)
            action = Action.MANUAL_REVIEW

        # --- rule 2: spend spike vs. this user's modeled baseline ---
        if history and txn.amount >= rolling_avg * SPEND_SPIKE_MULTIPLIER:
            reasons.append(
                f"amount {txn.amount:.2f} is {txn.amount / rolling_avg:.1f}x this "
                f"card's rolling average ({rolling_avg:.2f})"
            )
            risk_score = max(risk_score, 0.65)
            if action == Action.APPROVE:
                action = Action.STEP_UP_AUTH

        # --- rule 3: impossible travel (country changed too fast) ---
        if last_txn and last_txn.country != txn.country:
            minutes_since_last = (now - datetime.fromisoformat(last_txn.timestamp)).total_seconds() / 60
            if minutes_since_last <= IMPOSSIBLE_TRAVEL_WINDOW_MINUTES:
                reasons.append(
                    f"country changed {last_txn.country} -> {txn.country} in "
                    f"{minutes_since_last:.0f} min (impossible travel)"
                )
                risk_score = max(risk_score, 0.9)
                action = Action.DECLINE

        if not reasons:
            reasons.append("current transaction is consistent with modeled history")

        return Decision(
            agent_name=self.name,
            transaction_id=txn.transaction_id,
            action=action,
            risk_score=risk_score,
            reasons=reasons,
        )
