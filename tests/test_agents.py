import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.models import Transaction, Action
from src.agents.simple_reflex import SimpleReflexAgent
from src.agents.model_based_reflex import ModelBasedReflexAgent
from src.agents.goal_based import GoalBasedAgent
from src.agents.utility_based import UtilityBasedAgent
from src.agents.learning_agent import LearningAgent
from src import db


def make_txn(**overrides) -> Transaction:
    defaults = dict(
        transaction_id="t1",
        timestamp="2026-01-01T12:00:00",
        user_id="user_001",
        card_id="card_001",
        amount=50.0,
        currency="USD",
        merchant="Acme Corp",
        merchant_category="grocery",
        country="US",
        user_home_country="US",
        device_id="dev_1",
        ip_address="1.2.3.4",
        channel="pos",
        is_fraud=False,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


# --- Simple Reflex Agent ----------------------------------------------------

def test_simple_reflex_approves_ordinary_purchase():
    agent = SimpleReflexAgent()
    decision = agent.run(make_txn(amount=45.0))
    assert decision.action == Action.APPROVE


def test_simple_reflex_declines_blocked_country():
    agent = SimpleReflexAgent()
    decision = agent.run(make_txn(country="KP"))
    assert decision.action == Action.DECLINE


def test_simple_reflex_declines_over_hard_ceiling():
    agent = SimpleReflexAgent()
    decision = agent.run(make_txn(amount=6000.0))
    assert decision.action == Action.DECLINE


# --- Model-Based Reflex Agent ------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    test_db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db(reset=True)
    yield


def test_model_based_reflex_flags_velocity_burst():
    agent = ModelBasedReflexAgent()
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    history = [
        make_txn(transaction_id=f"h{i}", timestamp=(base_time + timedelta(minutes=i * 2)).isoformat())
        for i in range(3)
    ]
    db.insert_transactions(history)

    new_txn = make_txn(transaction_id="new", timestamp=(base_time + timedelta(minutes=7)).isoformat())
    decision = agent.run(new_txn)
    assert decision.action in (Action.MANUAL_REVIEW, Action.STEP_UP_AUTH, Action.DECLINE)
    assert any("transactions in the last" in r for r in decision.reasons)


def test_model_based_reflex_flags_impossible_travel():
    agent = ModelBasedReflexAgent()
    base_time = datetime(2026, 1, 1, 12, 0, 0)
    db.insert_transactions([
        make_txn(transaction_id="h1", timestamp=base_time.isoformat(), country="US"),
    ])
    new_txn = make_txn(
        transaction_id="new",
        timestamp=(base_time + timedelta(minutes=20)).isoformat(),
        country="FR",
    )
    decision = agent.run(new_txn)
    assert decision.action == Action.DECLINE
    assert any("impossible travel" in r for r in decision.reasons)


# --- Goal-Based Agent ---------------------------------------------------------

def test_goal_based_approves_low_risk():
    agent = GoalBasedAgent(use_llm=False)
    decision = agent.run(make_txn(amount=30.0))
    assert decision.action == Action.APPROVE


def test_goal_based_declines_high_risk():
    agent = GoalBasedAgent(use_llm=False)
    decision = agent.run(make_txn(country="IR"))
    assert decision.action == Action.DECLINE


# --- Utility-Based Agent -------------------------------------------------------

def test_utility_based_approves_cheap_low_risk_purchase():
    agent = UtilityBasedAgent()
    decision = agent.run(make_txn(amount=10.0))
    assert decision.action == Action.APPROVE


def test_utility_based_flags_high_risk_category_large_amount():
    agent = UtilityBasedAgent()
    decision = agent.run(make_txn(
        amount=4000.0, merchant_category="gift_cards", country="FR", user_home_country="US",
    ))
    assert decision.action in (Action.DECLINE, Action.MANUAL_REVIEW, Action.STEP_UP_AUTH)


# --- Learning Agent -------------------------------------------------------------

def test_learning_agent_requires_fit_before_deciding():
    agent = LearningAgent()
    with pytest.raises(RuntimeError):
        agent.run(make_txn())


def test_learning_agent_fits_and_decides():
    agent = LearningAgent(epochs=20)
    training = [make_txn(transaction_id=f"tr{i}", amount=20.0, is_fraud=False) for i in range(10)]
    training += [
        make_txn(transaction_id=f"fr{i}", amount=3000.0, merchant_category="crypto_exchange", is_fraud=True)
        for i in range(10)
    ]
    agent.fit(training)
    decision = agent.run(make_txn(amount=25.0))
    assert decision.action is not None
    assert 0.0 <= decision.risk_score <= 1.0


def test_learning_agent_learn_one_does_not_crash():
    agent = LearningAgent(epochs=5)
    training = [make_txn(transaction_id=f"tr{i}", amount=20.0, is_fraud=False) for i in range(5)]
    training += [make_txn(transaction_id=f"fr{i}", amount=3000.0, is_fraud=True) for i in range(5)]
    agent.fit(training)
    agent.learn_one(make_txn(transaction_id="fb1", amount=50.0), true_label=True)
    decision = agent.run(make_txn(amount=25.0))
    assert decision.action is not None
