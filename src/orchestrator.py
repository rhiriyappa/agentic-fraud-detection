"""Runs all five agents over the transaction set and scores them against the
ground-truth labels so the demo can show, side by side, how each agent type
behaves differently on the exact same data.
"""

from __future__ import annotations

from src.models import Transaction, Decision, Action
from src.agents import (
    SimpleReflexAgent,
    ModelBasedReflexAgent,
    GoalBasedAgent,
    UtilityBasedAgent,
    LearningAgent,
)
from src import db

# Actions considered a "block" of the transaction, for precision/recall scoring.
BLOCKING_ACTIONS = {Action.DECLINE, Action.MANUAL_REVIEW, Action.STEP_UP_AUTH}


def build_agents(use_llm: bool = True) -> dict:
    return {
        "simple_reflex": SimpleReflexAgent(),
        "model_based_reflex": ModelBasedReflexAgent(),
        "goal_based": GoalBasedAgent(use_llm=use_llm),
        "utility_based": UtilityBasedAgent(),
        "learning_agent": LearningAgent(),
    }


def run_all(transactions: list[Transaction] | None = None, use_llm: bool = True,
            persist: bool = True) -> dict[str, list[Decision]]:
    transactions = transactions or db.fetch_all_transactions()
    agents = build_agents(use_llm=use_llm)

    # The learning agent needs to be trained before it can decide anything.
    agents["learning_agent"].fit(transactions)

    results: dict[str, list[Decision]] = {name: [] for name in agents}
    for name, agent in agents.items():
        for txn in transactions:
            decision = agent.run(txn)
            results[name].append(decision)
            if persist:
                db.save_decision(decision)
    return results


def score(transactions: list[Transaction], decisions: list[Decision]) -> dict:
    """Simple precision/recall against ground truth, treating any
    non-APPROVE action as a 'flagged' prediction."""
    by_id = {t.transaction_id: t for t in transactions}
    tp = fp = tn = fn = 0
    for d in decisions:
        actual_fraud = by_id[d.transaction_id].is_fraud
        flagged = d.action in BLOCKING_ACTIONS
        if flagged and actual_fraud:
            tp += 1
        elif flagged and not actual_fraud:
            fp += 1
        elif not flagged and actual_fraud:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(decisions) if decisions else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }
