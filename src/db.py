"""SQLite persistence layer.

Stripped-down on purpose: plain stdlib sqlite3, no ORM. Tables:
  transactions  - the 50 synthetic payment transactions (+ ground-truth label)
  decisions     - every decision each agent makes, tagged with the run_id that produced it
  runs          - one row per demo run: LLM mode plus LLM call/cache/latency counters
  llm_cache     - Mistral responses keyed by a hash of the exact prompt, so re-runs are free
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.models import Transaction, Decision

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud_agents.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      TEXT PRIMARY KEY,
    timestamp            TEXT NOT NULL,
    user_id               TEXT NOT NULL,
    card_id               TEXT NOT NULL,
    amount                REAL NOT NULL,
    currency               TEXT NOT NULL,
    merchant               TEXT NOT NULL,
    merchant_category      TEXT NOT NULL,
    country                 TEXT NOT NULL,
    user_home_country       TEXT NOT NULL,
    device_id               TEXT NOT NULL,
    ip_address               TEXT NOT NULL,
    channel                   TEXT NOT NULL,
    is_fraud                   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    agent_name       TEXT NOT NULL,
    transaction_id    TEXT NOT NULL REFERENCES transactions(transaction_id),
    action              TEXT NOT NULL,
    risk_score           REAL NOT NULL,
    reasons               TEXT NOT NULL,
    explanation            TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    llm_mode               TEXT NOT NULL,
    n_transactions          INTEGER NOT NULL,
    llm_requests             INTEGER NOT NULL,
    llm_ok                    INTEGER NOT NULL,
    llm_errors                 INTEGER NOT NULL,
    llm_cache_hits              INTEGER NOT NULL,
    llm_fallbacks                INTEGER NOT NULL,
    llm_avg_latency_s             REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     TEXT PRIMARY KEY,
    model          TEXT NOT NULL,
    response        TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_txn ON decisions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_txn_user_ts ON transactions(user_id, timestamp);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(reset: bool = False) -> None:
    """Create tables (idempotent). `reset=True` also wipes transactions,
    decisions and runs; the LLM cache is kept since it is keyed by prompt.

    A decisions table from before run_id existed is dropped: it only holds
    derived data that any demo run regenerates.
    """
    with get_conn() as conn:
        if reset:
            conn.executescript(
                "DROP TABLE IF EXISTS decisions; DROP TABLE IF EXISTS runs; "
                "DROP TABLE IF EXISTS transactions;"
            )
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(decisions)")}
        if columns and "run_id" not in columns:
            conn.execute("DROP TABLE decisions")
        conn.executescript(SCHEMA)


def insert_transactions(transactions: list[Transaction]) -> None:
    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO transactions
               (transaction_id, timestamp, user_id, card_id, amount, currency,
                merchant, merchant_category, country, user_home_country,
                device_id, ip_address, channel, is_fraud)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    t.transaction_id, t.timestamp, t.user_id, t.card_id, t.amount,
                    t.currency, t.merchant, t.merchant_category, t.country,
                    t.user_home_country, t.device_id, t.ip_address, t.channel,
                    int(t.is_fraud),
                )
                for t in transactions
            ],
        )


def fetch_all_transactions() -> list[Transaction]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM transactions ORDER BY timestamp").fetchall()
    return [_row_to_transaction(r) for r in rows]


def fetch_transaction(transaction_id: str) -> Transaction | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
    return _row_to_transaction(row) if row else None


def fetch_user_history(user_id: str, before_timestamp: str, limit: int = 20) -> list[Transaction]:
    """Used by the model-based reflex agent to build up internal state."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM transactions
               WHERE user_id = ? AND timestamp < ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, before_timestamp, limit),
        ).fetchall()
    return [_row_to_transaction(r) for r in rows]


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def save_decisions(decisions: list[Decision], run_id: str) -> None:
    with get_conn() as conn:
        conn.executemany(
            """INSERT INTO decisions
               (run_id, agent_name, transaction_id, action, risk_score, reasons, explanation)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    run_id,
                    d.agent_name,
                    d.transaction_id,
                    d.action.value,
                    d.risk_score,
                    "; ".join(d.reasons),
                    d.explanation,
                )
                for d in decisions
            ],
        )


def fetch_decisions(agent_name: str | None = None, run_id: str | None = None) -> list[sqlite3.Row]:
    query, params = "SELECT * FROM decisions WHERE 1=1", []
    if agent_name:
        query += " AND agent_name = ?"
        params.append(agent_name)
    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)
    with get_conn() as conn:
        return conn.execute(query + " ORDER BY id", params).fetchall()


def save_run(run_id: str, llm_mode: str, n_transactions: int, stats: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO runs
               (run_id, llm_mode, n_transactions, llm_requests, llm_ok, llm_errors,
                llm_cache_hits, llm_fallbacks, llm_avg_latency_s)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                run_id, llm_mode, n_transactions, stats["requests"], stats["ok"],
                stats["errors"], stats["cache_hits"], stats["fallbacks"],
                stats["avg_latency_s"],
            ),
        )


def get_llm_cache(cache_key: str) -> str | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
    return row["response"] if row else None


def put_llm_cache(cache_key: str, model: str, response: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, model, response) VALUES (?,?,?)",
            (cache_key, model, response),
        )


def _row_to_transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        transaction_id=row["transaction_id"],
        timestamp=row["timestamp"],
        user_id=row["user_id"],
        card_id=row["card_id"],
        amount=row["amount"],
        currency=row["currency"],
        merchant=row["merchant"],
        merchant_category=row["merchant_category"],
        country=row["country"],
        user_home_country=row["user_home_country"],
        device_id=row["device_id"],
        ip_address=row["ip_address"],
        channel=row["channel"],
        is_fraud=bool(row["is_fraud"]),
    )
