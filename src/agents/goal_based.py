"""Goal-Based Agent
===================
Use case: adaptive checkout-flow orchestrator.

A goal-based agent doesn't just react -- it holds a goal ("resolve this
transaction into a SAFE state while minimizing customer friction") and
searches over the space of possible actions to find one that achieves it.

It reuses the simple-reflex and model-based-reflex agents as cheap "world
models" to estimate risk, then performs a small forward search over the
ordered action space [APPROVE < STEP_UP_AUTH < MANUAL_REVIEW < DECLINE],
picking the least-friction action whose predicted outcome still satisfies
the safety goal. This is the classic goal-based pattern: candidate actions
are tested against a goal condition rather than fired from fixed rules.

Mistral 7B (via Ollama) is used to narrate the reasoning trace for analysts.
"""

from __future__ import annotations

from src.models import Transaction, Decision, Action
from src.agents.base import Agent
from src.agents.simple_reflex import SimpleReflexAgent
from src.agents.model_based_reflex import ModelBasedReflexAgent
from src import llm

# Ordered from least to most friction on the customer.
ACTION_ORDER = [Action.APPROVE, Action.STEP_UP_AUTH, Action.MANUAL_REVIEW, Action.DECLINE]

# The maximum residual risk each action is allowed to "resolve" and still
# count as satisfying the safety goal.
GOAL_RISK_CEILING = {
    Action.APPROVE: 0.30,
    Action.STEP_UP_AUTH: 0.70,
    Action.MANUAL_REVIEW: 0.95,
    Action.DECLINE: 1.01,  # decline always trivially satisfies the goal
}


class GoalBasedAgent(Agent):
    name = "goal_based"

    def __init__(self, use_llm: bool = True):
        self._reflex = SimpleReflexAgent()
        self._model_based = ModelBasedReflexAgent()
        self.use_llm = use_llm

    def perceive(self, transaction: Transaction) -> dict:
        # Consult both lower-level agents as world models to estimate risk.
        reflex_decision = self._reflex.run(transaction)
        model_decision = self._model_based.run(transaction)
        estimated_risk = max(reflex_decision.risk_score, model_decision.risk_score)
        signals = list(dict.fromkeys(reflex_decision.reasons + model_decision.reasons))
        return {"transaction": transaction, "estimated_risk": estimated_risk, "signals": signals}

    def decide(self, percept: dict) -> Decision:
        txn: Transaction = percept["transaction"]
        risk = percept["estimated_risk"]
        signals: list[str] = percept["signals"]

        # Goal-directed search: try the least-friction action first, accept
        # the first one whose ceiling the estimated risk satisfies.
        chosen = Action.DECLINE
        search_trace = []
        for candidate in ACTION_ORDER:
            ceiling = GOAL_RISK_CEILING[candidate]
            satisfies = risk < ceiling
            search_trace.append(f"{candidate.value}(ceiling={ceiling:.2f}) -> {'OK' if satisfies else 'reject'}")
            if satisfies:
                chosen = candidate
                break

        reasons = signals + [f"goal search: {' | '.join(search_trace)}"]

        explanation = ""
        if self.use_llm:
            explanation = llm.explain_decision(txn, chosen.value, risk, signals or ["baseline risk"])

        return Decision(
            agent_name=self.name,
            transaction_id=txn.transaction_id,
            action=chosen,
            risk_score=risk,
            reasons=reasons,
            explanation=explanation,
        )
