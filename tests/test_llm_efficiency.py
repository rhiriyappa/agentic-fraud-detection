import sqlite3

import pytest
import requests

from src import db, llm
from src.agents.goal_based import GoalBasedAgent
from src.models import Action
from src.orchestrator import run_all
from tests.test_agents import make_txn


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "Mistral says hi."}


@pytest.fixture
def ollama_calls(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json)
        return FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls


def test_should_explain_gating():
    off, flagged, every = (GoalBasedAgent(llm_mode=m) for m in ("off", "flagged", "all"))
    assert [a.should_explain(Action.APPROVE) for a in (off, flagged, every)] == [False, False, True]
    assert [a.should_explain(Action.DECLINE) for a in (off, flagged, every)] == [False, True, True]


def test_invalid_llm_mode_rejected():
    with pytest.raises(ValueError):
        GoalBasedAgent(llm_mode="sometimes")


def test_flagged_mode_skips_llm_for_approved_transaction(ollama_calls):
    decision = GoalBasedAgent(llm_mode="flagged").run(make_txn(amount=30.0))
    assert decision.action == Action.APPROVE
    assert decision.explanation == ""
    assert ollama_calls == []


def test_flagged_mode_calls_llm_for_declined_transaction(ollama_calls):
    decision = GoalBasedAgent(llm_mode="flagged").run(make_txn(country="IR"))
    assert decision.action == Action.DECLINE
    assert decision.explanation == "Mistral says hi."
    assert len(ollama_calls) == 1


def test_generate_caches_identical_requests(ollama_calls):
    assert llm.generate("same prompt") == "Mistral says hi."
    assert llm.generate("same prompt") == "Mistral says hi."
    stats = llm.get_stats()
    assert (stats.requests, stats.ok, stats.cache_hits) == (1, 1, 1)
    assert len(ollama_calls) == 1


def test_use_cache_false_always_calls_ollama(ollama_calls):
    llm.generate("same prompt", use_cache=False)
    llm.generate("same prompt", use_cache=False)
    assert len(ollama_calls) == 2
    assert llm.get_stats().cache_hits == 0


def test_unreachable_ollama_counts_error_and_fallback(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("no ollama")

    monkeypatch.setattr(llm.requests, "post", boom)
    text = llm.explain_decision(make_txn(), "DECLINE", 0.9, ["blocked country"])
    assert text.startswith("DECLINE (risk=0.90)")
    stats = llm.get_stats()
    assert (stats.requests, stats.errors, stats.fallbacks) == (1, 1, 1)


def test_init_db_drops_pre_run_id_decisions_table(tmp_path, monkeypatch):
    old_db = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_PATH", old_db)
    conn = sqlite3.connect(old_db)
    conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, agent_name TEXT)")
    conn.execute("INSERT INTO decisions (agent_name) VALUES ('old')")
    conn.commit()
    conn.close()

    db.init_db()

    with db.get_conn() as c:
        columns = {r["name"] for r in c.execute("PRAGMA table_info(decisions)")}
        assert "run_id" in columns
        assert c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_run_all_tags_decisions_with_run_id_and_records_llm_usage(ollama_calls):
    txns = [
        make_txn(transaction_id="ok1", amount=20.0),
        make_txn(transaction_id="ok2", amount=25.0, timestamp="2026-01-01T13:00:00"),
        make_txn(transaction_id="bad", country="KP", timestamp="2026-01-01T14:00:00"),
    ]
    db.insert_transactions(txns)

    run = run_all(txns, llm_mode="flagged")

    assert len(ollama_calls) == 1  # only the declined transaction was narrated
    assert run.llm_stats.requests == 1
    rows = db.fetch_decisions(run_id=run.run_id)
    assert len(rows) == 5 * len(txns)
    with db.get_conn() as c:
        saved = c.execute("SELECT * FROM runs WHERE run_id = ?", (run.run_id,)).fetchone()
    assert (saved["llm_mode"], saved["llm_requests"], saved["n_transactions"]) == ("flagged", 1, 3)

    second = run_all(txns, llm_mode="flagged")
    assert second.llm_stats.requests == 0 and second.llm_stats.cache_hits == 1
    assert len(db.fetch_decisions(run_id=second.run_id)) == 5 * len(txns)
