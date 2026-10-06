"""CLI demo: runs all five agent types over the 50 sample transactions and
prints a side-by-side comparison, plus precision/recall against the
ground-truth fraud labels.

Usage:
    python scripts/run_demo.py                       # Mistral explains flagged (non-APPROVE) goal-based decisions
    python scripts/run_demo.py --llm-mode off         # no LLM calls (fastest, fully offline)
    python scripts/run_demo.py --llm-mode all          # explain every goal-based decision
    python scripts/run_demo.py --no-cache               # ignore the SQLite LLM cache
    python scripts/run_demo.py -v                        # log every LLM call
    python scripts/run_demo.py --txn <id>                 # deep-dive a single transaction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src import db, llm
from src.orchestrator import run_all, score, build_agents

# To export the rich console output to HTML, initialize the console with record=True. This will capture all output and allow us to save it later.
console = Console(record=True)

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


def print_llm_summary(run):
    s = run.llm_stats
    goal = run.decisions["goal_based"]
    explained = sum(1 for d in goal if d.explanation)
    console.print(
        f"[bold]LLM usage[/] (run {run.run_id}, mode={run.llm_mode}): "
        f"{explained}/{len(goal)} goal-based decisions explained -> "
        f"{s.requests} Mistral requests ({s.ok} ok, {s.errors} failed), "
        f"{s.cache_hits} cache hits, {s.fallbacks} template fallbacks, "
        f"avg latency {s.avg_latency_s:.2f}s"
    )


def deep_dive(transaction_id: str, llm_mode: str, use_cache: bool):
    txn = db.fetch_transaction(transaction_id)
    if not txn:
        console.print(f"[red]No transaction with id {transaction_id}[/]")
        return
    agents = build_agents(llm_mode=llm_mode, use_cache=use_cache)
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
    parser.add_argument("--llm-mode", choices=["off", "flagged", "all"], default="flagged",
                        help="when Mistral explains goal-based decisions (default: flagged)")
    parser.add_argument("--no-cache", action="store_true", help="ignore the SQLite LLM response cache")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every LLM call")
    parser.add_argument("--txn", type=str, default=None, help="deep-dive a single transaction id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s")
    db.init_db()

    use_cache = not args.no_cache
    if args.llm_mode != "off" and not llm.is_available():
        console.print(
            "[yellow]Ollama/Mistral not reachable at "
            f"{llm.OLLAMA_HOST} -- falling back to template explanations. "
            "Run `ollama serve` and `ollama pull mistral` to enable live LLM narration.[/]"
        )

    if args.txn:
        deep_dive(args.txn, llm_mode=args.llm_mode, use_cache=use_cache)
        return

    transactions = db.fetch_all_transactions()
    if not transactions:
        console.print("[red]No transactions found. Run: python data/generate_transactions.py[/]")
        return

    console.print(f"[bold]Loaded {len(transactions)} transactions "
                   f"({sum(t.is_fraud for t in transactions)} labeled fraud)[/]\n")

    run = run_all(transactions, llm_mode=args.llm_mode, use_cache=use_cache)
    print_comparison_table(transactions, run.decisions)
    console.print()
    print_scoreboard(transactions, run.decisions)
    console.print()
    print_llm_summary(run)

    Path("output").mkdir(exist_ok=True)
    #console.save_html("output/agent_decisions.html")
    #console.save_svg("output/agent_decisions.svg")
    


if __name__ == "__main__":
    main()
