"""Generates 50 synthetic payment transactions covering the fraud patterns
each agent type is designed to catch:

  - plain everyday purchases (majority, legit)                 -> baseline
  - a card-testing / velocity burst                              -> model-based reflex
  - impossible travel (country hop within the hour)               -> model-based reflex
  - a spend spike vs. the card's own rolling average                -> model-based reflex
  - a hard-limit / sanctioned-country transaction                    -> simple reflex
  - high-risk-category purchases, some legit ("whale") some fraud    -> utility-based
  - a big legitimate purchase that *looks* risky on amount alone      -> utility-based

Deterministic (seeded) so the repo ships one canonical dataset, but you can
regenerate a fresh batch with `python data/generate_transactions.py --seed 7`.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import Transaction  # noqa: E402
from src import db  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "sample_transactions.csv"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "electronics", "travel", "subscription",
    "clothing", "home_goods", "pharmacy", "fuel",
]
HIGH_RISK_CATEGORIES = ["crypto_exchange", "gambling", "money_transfer", "gift_cards"]
BLOCKED_COUNTRIES = ["KP", "IR", "SY"]
NORMAL_COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "JP", "IN", "BR", "MX"]
CHANNELS = ["online", "pos", "atm"]


def _txn(fake, user_id, home_country, ts, amount, category, country=None,
          channel=None, is_fraud=False, card_id=None, device_id=None) -> Transaction:
    return Transaction(
        transaction_id=str(uuid.uuid4())[:8],
        timestamp=ts.isoformat(timespec="seconds"),
        user_id=user_id,
        card_id=card_id or f"card_{user_id[-4:]}",
        amount=round(amount, 2),
        currency="USD",
        merchant=fake.company(),
        merchant_category=category,
        country=country or home_country,
        user_home_country=home_country,
        device_id=device_id or f"dev_{fake.uuid4()[:6]}",
        ip_address=fake.ipv4_public(),
        channel=channel or random.choice(CHANNELS),
        is_fraud=is_fraud,
    )


def generate(seed: int = 42) -> list[Transaction]:
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    txns: list[Transaction] = []
    base_time = datetime(2026, 9, 1, 8, 0, 0)

    users = [
        {"user_id": f"user_{i:03d}", "home_country": random.choice(NORMAL_COUNTRIES)}
        for i in range(1, 16)
    ]

    # --- 1. baseline everyday purchases (28 legit transactions) -----------
    t = base_time
    for i in range(28):
        u = random.choice(users)
        t = t + timedelta(hours=random.uniform(1, 6))
        txns.append(_txn(
            fake, u["user_id"], u["home_country"], t,
            amount=random.uniform(8, 220),
            category=random.choice(MERCHANT_CATEGORIES),
            country=u["home_country"],
            is_fraud=False,
        ))

    # --- 2. card-testing / velocity burst (4 txns, same card, 2-4 min apart, one user) ---
    burst_user = users[0]
    t = base_time + timedelta(days=1, hours=3)
    for i in range(4):
        t = t + timedelta(minutes=random.uniform(2, 4))
        txns.append(_txn(
            fake, burst_user["user_id"], burst_user["home_country"], t,
            amount=random.uniform(1, 15),  # classic card-testing: tiny amounts
            category="subscription",
            country=burst_user["home_country"],
            channel="online",
            is_fraud=True,
        ))

    # --- 3. impossible travel: same user, two countries, 25 min apart -----
    travel_user = users[1]
    t = base_time + timedelta(days=1, hours=9)
    txns.append(_txn(
        fake, travel_user["user_id"], travel_user["home_country"], t,
        amount=random.uniform(40, 90), category="restaurant",
        country=travel_user["home_country"], channel="pos", is_fraud=False,
    ))
    t2 = t + timedelta(minutes=25)
    far_country = next(c for c in NORMAL_COUNTRIES if c != travel_user["home_country"])
    txns.append(_txn(
        fake, travel_user["user_id"], travel_user["home_country"], t2,
        amount=random.uniform(300, 600), category="electronics",
        country=far_country, channel="pos", is_fraud=True,
    ))

    # --- 4. spend spike vs. rolling average (user with steady small spend, then 1 big one) ---
    spike_user = users[2]
    t = base_time + timedelta(days=2, hours=1)
    for i in range(3):
        t = t + timedelta(hours=random.uniform(2, 5))
        txns.append(_txn(
            fake, spike_user["user_id"], spike_user["home_country"], t,
            amount=random.uniform(15, 40), category="grocery",
            country=spike_user["home_country"], is_fraud=False,
        ))
    t = t + timedelta(hours=3)
    txns.append(_txn(
        fake, spike_user["user_id"], spike_user["home_country"], t,
        amount=random.uniform(900, 1400), category="electronics",
        country=spike_user["home_country"], channel="online", is_fraud=True,
    ))

    # --- 5. hard-limit / sanctioned-country transactions -------------------
    for i in range(2):
        u = random.choice(users)
        t = t + timedelta(hours=random.uniform(1, 4))
        txns.append(_txn(
            fake, u["user_id"], u["home_country"], t,
            amount=random.uniform(2000, 6000), category="money_transfer",
            country=random.choice(BLOCKED_COUNTRIES), channel="online", is_fraud=True,
        ))

    # --- 6. high-risk category: legit "whale" collector vs. fraud attempt --
    whale_user = users[3]
    t = t + timedelta(hours=2)
    txns.append(_txn(
        fake, whale_user["user_id"], whale_user["home_country"], t,
        amount=random.uniform(3000, 4800), category="crypto_exchange",
        country=whale_user["home_country"], channel="online", is_fraud=False,
    ))
    fraud_user = users[4]
    t = t + timedelta(hours=1)
    txns.append(_txn(
        fake, fraud_user["user_id"], fraud_user["home_country"], t,
        amount=random.uniform(2500, 4800), category="gift_cards",
        country=fraud_user["home_country"], channel="online", is_fraud=True,
    ))

    # --- 7. big but legitimate purchase (tests utility agent's trade-off) --
    for i in range(2):
        u = random.choice(users)
        t = t + timedelta(hours=random.uniform(2, 6))
        txns.append(_txn(
            fake, u["user_id"], u["home_country"], t,
            amount=random.uniform(1600, 2800), category="travel",
            country=u["home_country"], channel="online", is_fraud=False,
        ))

    # --- fill remainder up to 50 with more everyday baseline txns ----------
    while len(txns) < 50:
        u = random.choice(users)
        t = t + timedelta(hours=random.uniform(1, 5))
        txns.append(_txn(
            fake, u["user_id"], u["home_country"], t,
            amount=random.uniform(8, 180),
            category=random.choice(MERCHANT_CATEGORIES),
            country=u["home_country"], is_fraud=False,
        ))

    txns.sort(key=lambda x: x.timestamp)
    return txns[:50]


def write_csv(txns: list[Transaction], path: Path = CSV_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(txns[0].as_dict().keys()))
        writer.writeheader()
        for t in txns:
            writer.writerow(t.as_dict())


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud-demo transactions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-db", action="store_true", help="drop & recreate tables")
    args = parser.parse_args()

    txns = generate(seed=args.seed)
    write_csv(txns)
    db.init_db(reset=args.reset_db)
    db.insert_transactions(txns)

    n_fraud = sum(t.is_fraud for t in txns)
    print(f"Generated {len(txns)} transactions ({n_fraud} labeled fraud) -> {CSV_PATH}")
    print(f"Loaded into SQLite -> {db.DB_PATH}")


if __name__ == "__main__":
    main()
