"""SQLite persistence layer.

Stripped-down on purpose: plain stdlib sqlite3, no ORM. Two tables:
  transactions  - the 50 synthetic payment transactions (+ ground-truth label)
  decisions     - every decision each agent makes, for later comparison/eval
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
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
    agent_name       TEXT NOT NULL,
    transaction_id    TEXT NOT NULL REFERENCES transactions(transaction_id),
    action              TEXT NOT NULL,
    risk_score           REAL NOT NULL,
    reasons               TEXT NOT NULL,
    explanation            TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

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
    with get_conn() as conn:
        if reset:
            conn.executescript("DROP TABLE IF EXISTS decisions; DROP TABLE IF EXISTS transactions;")
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


def save_decision(decision: Decision) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO decisions
               (agent_name, transaction_id, action, risk_score, reasons, explanation)
               VALUES (?,?,?,?,?,?)""",
            (
                decision.agent_name,
                decision.transaction_id,
                decision.action.value,
                decision.risk_score,
                "; ".join(decision.reasons),
                decision.explanation,
            ),
        )


def fetch_decisions(agent_name: str | None = None) -> list[sqlite3.Row]:
    with get_conn() as conn:
        if agent_name:
            return conn.execute(
                "SELECT * FROM decisions WHERE agent_name = ? ORDER BY id", (agent_name,)
            ).fetchall()
        return conn.execute("SELECT * FROM decisions ORDER BY id").fetchall()


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
