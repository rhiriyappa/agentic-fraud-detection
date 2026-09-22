# Agentic Fraud Detection Demo

A stripped-down, runnable demonstration of the five classic AI agent
architectures (Russell & Norvig's taxonomy), each applied to a **distinct**
payment-fraud use case:

| # | Agent type | Use case | What makes it that type |
|---|---|---|---|
| 1 | **Simple Reflex** | Point-of-sale hard-limit gatekeeper | Maps the *current transaction alone* to an action via fixed condition-action rules (amount ceilings, blocked countries/categories). No memory. |
| 2 | **Model-Based Reflex** | Card-velocity & impossible-travel detector | Reconstructs internal state (recent transaction history) from SQLite before applying rules -- catches patterns a single transaction can't reveal. |
| 3 | **Goal-Based** | Adaptive checkout-flow orchestrator | Holds the goal "resolve the transaction safely with minimal friction" and *searches* the action space `[APPROVE < STEP_UP_AUTH < MANUAL_REVIEW < DECLINE]` for the least-friction action that satisfies it. Narrates its reasoning via Mistral 7B. |
| 4 | **Utility-Based** | Expected-value checkout optimizer | Assigns a monetary expected utility to every action (fraud loss vs. merchant margin vs. customer-friction cost vs. review cost) and picks the argmax. |
| 5 | **Learning Agent** | Adaptive fraud-scoring model | An online logistic-regression classifier (scikit-learn `SGDClassifier`) trained on labeled transactions, with a `learn_one()` feedback hook for chargebacks / cleared false positives. |

All five agents run over the **same 50 synthetic transactions** so you can see,
side by side, how each type of "agentic-ness" changes the outcome.

## Reference architecture

```
                    ┌────────────────────────────────┐
                    │ data/generate_transactions.py  │
                    │   (Faker, seeded RNG)          │
                    └───────────────┬────────────────┘
                                    │ writes 50 transactions
                                    ▼
                ┌──────────────────────────────────────────┐
                │         SQLite -- data/fraud_agents.db   │
                │      tables: transactions, decisions     │
                └────────────────────┬─────────────────────┘
                                     │ perceive(): txn (+ rolling history for model-based)
                                     ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │                       src/orchestrator.py                         │
      │              run_all()  ->  { agent_name: [Decision, ...] }       │
      └───┬─────────────┬─────────────┬─────────────┬─────────────┬───────┘
          │             │             │             │             │
          ▼             ▼             ▼             ▼             ▼
      Simple        Model-Based    Goal-Based    Utility-Based   Learning
      Reflex          Reflex        Agent          Agent          Agent
      (condition-   (condition-    (searches      (expected-     (scikit-learn
      action on     action on      action space   utility        SGDClassifier,
      txn only,     txn + SQLite   for goal-      argmax over    trained on
      no memory)    history)       satisfying     weighted        labels, updates
                                    least-friction  cost/benefit   via learn_one())
                                    action)         table)
                                      │
                                      │ explain_decision(txn, action, risk, reasons)
                                      ▼
                          ┌────────────────────┐ HTTP/api/generate ┌───────────────────┐
                          │   src/llm.py       │──────────────────▶│ Ollama server     │
                          │ (client + offline  │ ──────────────────│ Mistral 7B (local)│
                          │  template fallback)│      completion   │ no API key needed │
                          └────────────────────┘                   └───────────────────┘

              │              │             │             │             │
              └──────────────┴────────┬────┴─────────────┴─────────────┘
                                      ▼
                        Decision{ action, risk_score, reasons, explanation }
                                      │ persisted
                                      ▼
                        SQLite `decisions` table  (per-agent audit trail)
                                      │
                          ┌───────────────────────┐
                          ▼                       ▼
            scripts/run_demo.py            api/main.py (FastAPI)
            Rich CLI: comparison table      HTTP endpoints + /docs:
          + precision/recall scoreboard    evaluate / evaluate-all /
                                                  learn / scoreboard
```

Only the **goal-based agent** calls out to Mistral (for its analyst-facing
narration); the other four agents are deliberately dependency-free rule/math/
ML logic so the differences between agent *types* stay easy to read in the
source. If Ollama isn't reachable, `src/llm.py` transparently falls back to a
templated explanation string, so the whole pipeline still runs offline.

## Tech stack (all open source)

- **Database**: SQLite (stdlib `sqlite3`, no ORM -- two tables: `transactions`, `decisions`)
- **LLM**: [Mistral 7B](https://mistral.ai) served locally via [Ollama](https://ollama.com) (no API key, fully offline-capable)
- **Data generation**: [Faker](https://faker.readthedocs.io/)
- **Learning agent**: [scikit-learn](https://scikit-learn.org/) (`SGDClassifier`, online logistic regression)
- **API**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **CLI output**: [Rich](https://github.com/Textualize/rich)
- **Tests**: [pytest](https://pytest.org/)

## Project layout

```
agentic_ai/
├── data/
│   ├── generate_transactions.py   # generates the 50 sample transactions (seeded)
│   ├── sample_transactions.csv     # committed snapshot of the generated data
│   └── fraud_agents.db              # SQLite db (generated, gitignored)
├── src/
│   ├── db.py                        # SQLite schema + helpers
│   ├── llm.py                       # Mistral 7B client (via Ollama) with offline fallback
│   ├── orchestrator.py               # runs all agents + scores them vs. ground truth
│   ├── models/schema.py               # Transaction / Decision / Action dataclasses
│   └── agents/
│       ├── base.py                     # perceive() -> decide() interface
│       ├── simple_reflex.py
│       ├── model_based_reflex.py
│       ├── goal_based.py
│       ├── utility_based.py
│       └── learning_agent.py
├── api/main.py                        # FastAPI app exposing every agent over HTTP
├── scripts/run_demo.py                # CLI: full comparison table + scoreboard
└── tests/test_agents.py                # unit tests per agent
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull the local LLM (skip this and everything still works via template fallback)
ollama pull mistral
ollama serve   # usually already running as a background service on macOS
```

## Generate the 50 sample transactions

```bash
python data/generate_transactions.py --reset-db
```

This seeds SQLite with 50 transactions (9 labeled fraud) covering: everyday
legit purchases, a card-testing velocity burst, an impossible-travel case, a
spend spike vs. a card's own baseline, sanctioned-country transactions,
a legit "whale" high-risk-category purchase vs. a fraudulent one, and large
but legitimate purchases (to stress-test the utility trade-off). Re-run with
`--seed <n>` for a different batch.

## Run the CLI demo

```bash
python scripts/run_demo.py                 # full comparison + scoreboard (uses Mistral if reachable)
python scripts/run_demo.py --no-llm         # skip LLM narration (faster, fully offline)
python scripts/run_demo.py --txn <txn_id>    # deep-dive one transaction across all 5 agents
```

## Run the API

```bash
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

Key endpoints:
- `GET /agents` -- descriptions of the five agents
- `GET /transactions` -- the 50 sample transactions
- `POST /transactions/{id}/evaluate/{agent_name}` -- run one agent on one transaction
- `POST /transactions/{id}/evaluate-all` -- run all five agents on one transaction
- `POST /learning-agent/learn/{id}` -- feed the learning agent a confirmed outcome (`{"true_label": true}`)
- `GET /scoreboard` -- precision/recall for every agent vs. ground truth

## Run the tests

```bash
pytest tests/ -q
```

## Why each agent behaves differently (the point of the demo)

Run `python scripts/run_demo.py` and look at the scoreboard: the simple
reflex agent has the fewest false positives but misses slow-building fraud
patterns (it has no memory); the model-based reflex agent catches velocity
and impossible-travel fraud the simple agent structurally cannot see; the
goal-based agent reuses both of them as "world models" and picks the
least-friction action that still satisfies a safety goal; the utility-based
agent will sometimes *approve* a moderately risky transaction because the
expected fraud loss is smaller than the expected cost of friction/lost
revenue -- a trade-off none of the rule-based agents can express; and the
learning agent is the only one whose policy comes from data rather than
hand-written logic, so it's also the only one that improves when you call
`learn_one()` with new ground truth.

## Notes

- This is intentionally a "stripped down" educational scaffold, not a
  production fraud stack: thresholds, utility weights, and the heuristic
  fraud-probability estimator are simplified for clarity and are meant to be
  tuned/replaced.
- `data/sample_transactions.csv` is committed so the repo is reproducible
  without regenerating data; `data/fraud_agents.db` is gitignored and
  rebuilt by `generate_transactions.py`.
