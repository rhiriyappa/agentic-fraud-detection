"""Utility-Based Agent
======================
Use case: expected-value checkout optimizer for a payments platform that
cares about *both* fraud losses and revenue/customer-experience.

Rather than testing actions against a binary goal, this agent assigns a
monetary expected utility to every candidate action and picks the argmax.
The utility function explicitly trades off:
  - merchant margin captured on an approved, legitimate sale
  - chargeback loss + fee on an approved fraudulent sale
  - goodwill/CLV cost of wrongly declining a legitimate customer
  - cart-abandonment cost + fraud-deterrence benefit of step-up auth
  - fixed operational cost of routing a transaction to manual review

This is the natural fit for the classic "maximize expected utility" agent
definition, and demonstrates a business trade-off a pure goal/rule agent
can't express: sometimes it is worth *tolerating* residual risk because the
expected revenue outweighs the expected loss.
"""

from __future__ import annotations

import math

from src.models import Transaction, Decision, Action
from src.agents.base import Agent

# --- tunable business parameters (the "utility function") -----------------
MARGIN_RATE = 0.02            # merchant margin as a fraction of transaction amount
CHARGEBACK_FEE = 25.0          # flat fee incurred on a fraud chargeback
GOODWILL_COST_DECLINE = 15.0    # CLV/trust cost of wrongly declining a legit customer
REVIEW_OP_COST = 4.0             # analyst labor cost of a manual review
REVIEW_DELAY_GOODWILL_COST = 2.0  # minor friction cost even for legit reviewed customers
REVIEW_ACCURACY = 0.90             # manual review isn't infallible either
STEP_UP_ABANDON_RATE = 0.15       # fraction of legit customers who abandon at 2FA
STEP_UP_FRICTION_COST = 15.0        # blended goodwill/conversion cost when a legit
                                      # customer is challenged with step-up auth
STEP_UP_FRAUD_DETERRENCE = 0.80    # fraction of fraud attempts stopped by 2FA

HIGH_RISK_CATEGORIES = {"crypto_exchange", "gambling", "money_transfer", "gift_cards"}


class UtilityBasedAgent(Agent):
    name = "utility_based"

    def perceive(self, transaction: Transaction) -> dict:
        p_fraud = self._estimate_fraud_probability(transaction)
        return {"transaction": transaction, "p_fraud": p_fraud}

    def decide(self, percept: dict) -> Decision:
        txn: Transaction = percept["transaction"]
        p_fraud = percept["p_fraud"]
        p_legit = 1.0 - p_fraud
        amount = txn.amount

        utilities = {
            Action.APPROVE: (
                p_legit * (MARGIN_RATE * amount)
                - p_fraud * (amount + CHARGEBACK_FEE)
            ),
            Action.DECLINE: (
                -p_legit * GOODWILL_COST_DECLINE
            ),
            Action.STEP_UP_AUTH: (
                p_legit * (1 - STEP_UP_ABANDON_RATE) * (MARGIN_RATE * amount)
                - p_legit * STEP_UP_ABANDON_RATE * STEP_UP_FRICTION_COST
                - p_fraud * (1 - STEP_UP_FRAUD_DETERRENCE) * (amount + CHARGEBACK_FEE)
            ),
            Action.MANUAL_REVIEW: (
                p_legit * (
                    REVIEW_ACCURACY * (MARGIN_RATE * amount)
                    - (1 - REVIEW_ACCURACY) * GOODWILL_COST_DECLINE
                    - REVIEW_DELAY_GOODWILL_COST
                )
                - p_fraud * (1 - REVIEW_ACCURACY) * (amount + CHARGEBACK_FEE)
                - REVIEW_OP_COST
            ),
        }

        best_action = max(utilities, key=utilities.get)
        ranked = sorted(utilities.items(), key=lambda kv: kv[1], reverse=True)
        reasons = [f"P(fraud)={p_fraud:.2f}"] + [
            f"EU({a.value})=${u:.2f}" for a, u in ranked
        ]

        return Decision(
            agent_name=self.name,
            transaction_id=txn.transaction_id,
            action=best_action,
            risk_score=p_fraud,
            reasons=reasons,
        )

    @staticmethod
    def _estimate_fraud_probability(txn: Transaction) -> float:
        """Self-contained heuristic risk scorer (logistic combination of
        simple features). Deliberately independent of the other agents so
        this agent's utility trade-off is easy to reason about in isolation.
        """
        z = -4.5
        z += 1.2 * (txn.amount / 1000.0)
        z += 1.3 if txn.country != txn.user_home_country else 0.0
        z += 1.8 if txn.merchant_category in HIGH_RISK_CATEGORIES else 0.0
        z += 0.5 if txn.channel == "online" else 0.0
        return 1.0 / (1.0 + math.exp(-z))
