"""CLI demo: runs all five agent types over the 50 sample transactions and
prints a side-by-side comparison, plus precision/recall against the
ground-truth fraud labels.

Usage:
    python scripts/run_demo.py                 # uses Mistral via Ollama if reachable
    python scripts/run_demo.py --no-llm         # skip LLM explanations (faster, offline)
    python scripts/run_demo.py --txn <id>        # deep-dive a single transaction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src import db, llm
from src.orchestrator import run_all, score, build_agents

console = Console()

AGENT_ORDER = ["simple_reflex", "model_based_reflex", "goal_based", "utility_based", "learning_agent"]
AGENT_LABELS = {
    "simple_reflex": "1. Simple Reflex",
    "model_based_reflex": "2. Model-Based Reflex",
    "goal_based": "3. Goal-Based",
    "utility_based": "4. Utility-Based",
    "learning_agent": "5. Learning Agent",
}
ACTION_COLOR = {
    "APPROVE": "green",
    "STEP_UP_AUTH": "yellow",
    "MANUAL_REVIEW": "orange3",
    "DECLINE": "red",
}


def print_comparison_table(transactions, results):
    table = Table(title="Agent decisions across all 50 transactions", show_lines=False)
    table.add_column("Txn", style="dim")
    table.add_column("Amount", justify="right")
    table.add_column("Fraud?", justify="center")
    for name in AGENT_ORDER:
        table.add_column(AGENT_LABELS[name], justify="center")

    by_txn = {name: {d.transaction_id: d for d in decisions} for name, decisions in results.items()}
    for txn in transactions:
        row = [txn.transaction_id, f"${txn.amount:,.2f}", "[red]YES[/]" if txn.is_fraud else "no"]
        for name in AGENT_ORDER:
            d = by_txn[name][txn.transaction_id]
            color = ACTION_COLOR.get(d.action.value, "white")
            row.append(f"[{color}]{d.action.value}[/]")
        table.add_row(*row)
    console.print(table)


def print_scoreboard(transactions, results):
    table = Table(title="Precision / Recall vs. ground truth (any non-APPROVE = flagged)")
    table.add_column("Agent")
    table.add_column("TP"); table.add_column("FP"); table.add_column("TN"); table.add_column("FN")
    table.add_column("Precision"); table.add_column("Recall"); table.add_column("F1")

    for name in AGENT_ORDER:
        s = score(transactions, results[name])
        table.add_row(
            AGENT_LABELS[name],
            str(s["tp"]), str(s["fp"]), str(s["tn"]), str(s["fn"]),
            f"{s['precision']:.2f}", f"{s['recall']:.2f}", f"{s['f1']:.2f}",
        )
    console.print(table)


def deep_dive(transaction_id: str, use_llm: bool):
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        console.print(f"[red]No transaction with id {transaction_id}[/]")
        return
    agents = build_agents(use_llm=use_llm)
    agents["learning_agent"].fit(db.fetch_all_transactions())

    console.print(Panel(
        f"[bold]{txn.transaction_id}[/] -- ${txn.amount:,.2f} {txn.currency} at "
        f"{txn.merchant} ({txn.merchant_category})\n"
        f"user={txn.user_id} country={txn.country} (home={txn.user_home_country}) "
        f"channel={txn.channel}\nground truth: "
        f"{'[red]FRAUD[/]' if txn.is_fraud else '[green]legit[/]'}",
        title="Transaction",
    ))

    for name in AGENT_ORDER:
        d = agents[name].run(txn)
        color = ACTION_COLOR.get(d.action.value, "white")
        body = f"[{color}]{d.action.value}[/]  (risk={d.risk_score:.2f})\n"
        body += "\n".join(f"  - {r}" for r in d.reasons)
        if d.explanation:
            body += f"\n\n[italic]{d.explanation}[/]"
        console.print(Panel(body, title=AGENT_LABELS[name]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="skip Mistral explanations")
    parser.add_argument("--txn", type=str, default=None, help="deep-dive a single transaction id")
    args = parser.parse_args()

    use_llm = not args.no_llm
    if use_llm and not llm.is_available():
        console.print(
            "[yellow]Ollama/Mistral not reachable at "
            f"{llm.OLLAMA_HOST} -- falling back to template explanations. "
            "Run `ollama serve` and `ollama pull mistral` to enable live LLM narration.[/]"
        )

    if args.txn:
        deep_dive(args.txn, use_llm=use_llm)
        return

    transactions = db.fetch_all_transactions()
    if not transactions:
        console.print("[red]No transactions found. Run: python data/generate_transactions.py[/]")
        return

    console.print(f"[bold]Loaded {len(transactions)} transactions "
                   f"({sum(t.is_fraud for t in transactions)} labeled fraud)[/]\n")

    results = run_all(transactions, use_llm=use_llm)
    print_comparison_table(transactions, results)
    console.print()
    print_scoreboard(transactions, results)


if __name__ == "__main__":
    main()
