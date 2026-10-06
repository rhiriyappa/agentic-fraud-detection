"""Runs all five agents over the transaction set and scores them against the
ground-truth labels so the demo can show, side by side, how each agent type
behaves differently on the exact same data.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.models import Transaction, Decision, Action
from src.agents import (
    SimpleReflexAgent,
    ModelBasedReflexAgent,
    GoalBasedAgent,
    UtilityBasedAgent,
    LearningAgent,
)
from src import db, llm

# Actions considered a "block" of the transaction, for precision/recall scoring.
BLOCKING_ACTIONS = {Action.DECLINE, Action.MANUAL_REVIEW, Action.STEP_UP_AUTH}

# Ollama serves one request at a time by default, so extra workers only queue
# behind each other (and inflate latency toward the request timeout). Raise this
# together with OLLAMA_NUM_PARALLEL.
LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "1"))


@dataclass
class RunResult:
    run_id: str
    llm_mode: str
    decisions: dict[str, list[Decision]]
    llm_stats: llm.LLMStats


def build_agents(llm_mode: str = "flagged", defer_llm: bool = False, use_cache: bool = True) -> dict:
    return {
        "simple_reflex": SimpleReflexAgent(),
        "model_based_reflex": ModelBasedReflexAgent(),
        "goal_based": GoalBasedAgent(llm_mode=llm_mode, defer_llm=defer_llm, use_cache=use_cache),
        "utility_based": UtilityBasedAgent(),
        "learning_agent": LearningAgent(),
    }


def run_all(transactions: list[Transaction] | None = None, llm_mode: str = "flagged",
            persist: bool = True, use_cache: bool = True) -> RunResult:
    transactions = transactions or db.fetch_all_transactions()
    run_id = db.new_run_id()
    stats_before = llm.get_stats()

    # Decisions are computed first without the LLM; explanations are then
    # filled in concurrently and only for decisions that need one.
    agents = build_agents(llm_mode=llm_mode, defer_llm=True, use_cache=use_cache)

    # The learning agent needs to be trained before it can decide anything.
    agents["learning_agent"].fit(transactions)

    results: dict[str, list[Decision]] = {
        name: [agent.run(txn) for txn in transactions] for name, agent in agents.items()
    }

    goal: GoalBasedAgent = agents["goal_based"]
    by_id = {t.transaction_id: t for t in transactions}
    pending = [d for d in results["goal_based"] if goal.should_explain(d.action)]
    if pending:
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as pool:
            texts = pool.map(lambda d: goal.explain(d, by_id[d.transaction_id]), pending)
            for decision, text in zip(pending, texts):
                decision.explanation = text

    stats = llm.get_stats().minus(stats_before)
    if persist:
        db.save_decisions([d for ds in results.values() for d in ds], run_id)
        db.save_run(run_id, llm_mode, len(transactions), stats.as_dict())
    return RunResult(run_id=run_id, llm_mode=llm_mode, decisions=results, llm_stats=stats)


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
