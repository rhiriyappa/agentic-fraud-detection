"""FastAPI surface over the five agents, so the demo can also be driven over
HTTP (e.g. from a frontend, Postman, or curl) instead of only the CLI.

Run:
    uvicorn api.main:app --reload

Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import db, llm
from src.orchestrator import build_agents, run_all, score

app = FastAPI(
    title="Agentic Fraud Detection Demo",
    description="Five classic agent architectures (simple reflex, model-based "
                 "reflex, goal-based, utility-based, learning) applied to payment "
                 "fraud detection, each with a distinct use case.",
    version="0.1.0",
)

_agents_cache = None


def get_agents():
    global _agents_cache
    if _agents_cache is None:
        _agents_cache = build_agents(use_llm=llm.is_available())
        _agents_cache["learning_agent"].fit(db.fetch_all_transactions())
    return _agents_cache


class DecisionResponse(BaseModel):
    agent_name: str
    transaction_id: str
    action: str
    risk_score: float
    reasons: list[str]
    explanation: str = ""


class LearnRequest(BaseModel):
    true_label: bool


@app.get("/health")
def health():
    return {"status": "ok", "llm_available": llm.is_available()}


@app.get("/transactions")
def list_transactions():
    return [t.as_dict() for t in db.fetch_all_transactions()]


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    return txn.as_dict()


@app.get("/agents")
def list_agents():
    return {
        "simple_reflex": "Hard-limit gatekeeper: amount ceilings, blocklisted merchant categories/countries.",
        "model_based_reflex": "Card-velocity & impossible-travel detector using reconstructed transaction history.",
        "goal_based": "Checkout-flow orchestrator: searches the action space for the least-friction action that still satisfies a safety goal.",
        "utility_based": "Expected-value optimizer trading off fraud loss, merchant margin, and customer friction.",
        "learning_agent": "Online-learning fraud scorer (SGD/logistic regression) trained on labeled transactions, adaptable via feedback.",
    }


@app.post("/transactions/{transaction_id}/evaluate/{agent_name}", response_model=DecisionResponse)
def evaluate(transaction_id: str, agent_name: str):
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    agents = get_agents()
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"unknown agent '{agent_name}'")
    decision = agents[agent_name].run(txn)
    db.save_decision(decision)
    return DecisionResponse(**decision.as_dict())


@app.post("/transactions/{transaction_id}/evaluate-all")
def evaluate_all(transaction_id: str):
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    agents = get_agents()
    out = {}
    for name, agent in agents.items():
        decision = agent.run(txn)
        db.save_decision(decision)
        out[name] = decision.as_dict()
    return out


@app.post("/learning-agent/learn/{transaction_id}")
def learn_from_feedback(transaction_id: str, body: LearnRequest):
    """Feed the learning agent a confirmed outcome (e.g. a chargeback landed,
    or an analyst cleared a false positive) so it adapts online."""
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail="transaction not found")
    agents = get_agents()
    agents["learning_agent"].learn_one(txn, body.true_label)
    return {"status": "updated", "transaction_id": transaction_id, "true_label": body.true_label}


@app.get("/scoreboard")
def scoreboard():
    transactions = db.fetch_all_transactions()
    results = run_all(transactions, use_llm=False, persist=False)
    return {name: score(transactions, decisions) for name, decisions in results.items()}
