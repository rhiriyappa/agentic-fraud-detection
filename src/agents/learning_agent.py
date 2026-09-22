"""Learning Agent
=================
Use case: adaptive fraud-scoring model that improves as analysts label
outcomes -- the "chargeback feedback loop" every real payments platform runs.

This is the only agent whose decision policy is not hand-written: it fits a
small online classifier (scikit-learn's SGDClassifier with log loss, i.e.
online logistic regression) on the labeled sample transactions, then keeps
adapting via `learn_one()` as new ground truth arrives (e.g. a chargeback
lands, or an analyst clears a false positive). That mirrors AIMA's learning
agent: a performance element (predict_proba -> action), and a learning
element that updates the model from the critic's feedback (the true label).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight

from src.models import Transaction, Decision, Action
from src.agents.base import Agent

HIGH_RISK_CATEGORIES = {"crypto_exchange", "gambling", "money_transfer", "gift_cards"}

APPROVE_CEILING = 0.30
STEP_UP_CEILING = 0.60
REVIEW_CEILING = 0.85
# risk_score >= REVIEW_CEILING -> DECLINE


def featurize(txn: Transaction) -> list[float]:
    hour = datetime.fromisoformat(txn.timestamp).hour
    return [
        txn.amount,
        float(hour),
        1.0 if txn.country != txn.user_home_country else 0.0,
        1.0 if txn.merchant_category in HIGH_RISK_CATEGORIES else 0.0,
        1.0 if txn.channel == "online" else 0.0,
    ]


class LearningAgent(Agent):
    name = "learning_agent"

    def __init__(self, epochs: int = 200):
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss="log_loss", random_state=42, warm_start=True, alpha=0.001,
        )
        self.epochs = epochs
        self._fitted = False

    # --- learning element -------------------------------------------------
    def fit(self, transactions: list[Transaction]) -> None:
        """Initial batch training on historical labeled transactions.

        SGDClassifier learns via online gradient updates, so a single pass
        under-fits a small dataset -- we repeat shuffled passes (epochs)
        over the same labeled data, same as any online learner's warm-up.
        `class_weight="balanced"` compensates for fraud being the minority
        class (9/50 in the sample data).
        """
        X = np.array([featurize(t) for t in transactions])
        y = np.array([int(t.is_fraud) for t in transactions])
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        weights = compute_sample_weight("balanced", y)
        for epoch in range(self.epochs):
            Xe, ye, we = shuffle(Xs, y, weights, random_state=epoch)
            if epoch == 0:
                self.model.partial_fit(Xe, ye, classes=np.array([0, 1]), sample_weight=we)
            else:
                self.model.partial_fit(Xe, ye, sample_weight=we)
        self._fitted = True

    def learn_one(self, transaction: Transaction, true_label: bool) -> None:
        """Online update from a single new piece of ground truth (critic
        feedback, e.g. a confirmed chargeback or a cleared false positive).
        """
        X = np.array([featurize(transaction)])
        Xs = self.scaler.transform(X)
        self.model.partial_fit(Xs, np.array([int(true_label)]), sample_weight=np.array([3.0]))

    # --- performance element -----------------------------------------------
    def perceive(self, transaction: Transaction) -> dict:
        if not self._fitted:
            raise RuntimeError("LearningAgent.fit() must be called before making decisions")
        X = np.array([featurize(transaction)])
        Xs = self.scaler.transform(X)
        p_fraud = float(self.model.predict_proba(Xs)[0][1])
        return {"transaction": transaction, "p_fraud": p_fraud}

    def decide(self, percept: dict) -> Decision:
        txn: Transaction = percept["transaction"]
        p_fraud = percept["p_fraud"]

        if p_fraud < APPROVE_CEILING:
            action = Action.APPROVE
        elif p_fraud < STEP_UP_CEILING:
            action = Action.STEP_UP_AUTH
        elif p_fraud < REVIEW_CEILING:
            action = Action.MANUAL_REVIEW
        else:
            action = Action.DECLINE

        reasons = [f"learned model P(fraud)={p_fraud:.2f}"]
        return Decision(
            agent_name=self.name,
            transaction_id=txn.transaction_id,
            action=action,
            risk_score=p_fraud,
            reasons=reasons,
        )
