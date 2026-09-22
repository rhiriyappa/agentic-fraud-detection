"""Common interface every agent type implements.

perceive(transaction) -> percept dict     (what the agent senses)
decide(percept)        -> Decision         (what the agent does)

Splitting perceive/decide makes the difference between agent types explicit:
a simple reflex agent's percept is just the transaction; a model-based agent's
percept also carries the internal state it looked up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import Transaction, Decision


class Agent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def perceive(self, transaction: Transaction) -> dict:
        ...

    @abstractmethod
    def decide(self, percept: dict) -> Decision:
        ...

    def run(self, transaction: Transaction) -> Decision:
        percept = self.perceive(transaction)
        return self.decide(percept)
