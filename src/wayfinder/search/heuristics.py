"""Heuristics over search states.

Every heuristic exposes ``evaluate(states) -> list[float]`` and is called once per expansion with all
newly generated successor states, so a learned heuristic can batch its inference.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from wayfinder.cost import CostModel


class Heuristic:
    name = "heuristic"

    def evaluate(self, states: Sequence[int]) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError


class ZeroHeuristic(Heuristic):
    name = "zero"

    def evaluate(self, states):
        return [0.0] * len(states)


class NodeHeuristic(Heuristic):
    """A value per node; a state is scored by the node it has arrived at."""

    def __init__(self, h_node: np.ndarray, head_list: list[int], name: str = "node"):
        self.values = np.asarray(h_node, dtype=float)
        self._h = self.values.tolist()
        self._head = head_list
        self.name = name

    def evaluate(self, states):
        h, head = self._h, self._head
        return [h[head[s]] for s in states]


class StateHeuristic(Heuristic):
    """A value per state (used for the perfect heuristic h*, which depends on the incoming edge)."""

    def __init__(self, h_state: np.ndarray, name: str = "state"):
        self.values = np.asarray(h_state, dtype=float)
        self._h = self.values.tolist()
        self.name = name

    def evaluate(self, states):
        h = self._h
        return [h[s] for s in states]


def admissible_heuristic(cm: CostModel, target: int, w) -> NodeHeuristic:
    """h_adm(v, t, w) = rho_min(w) * haversine(v, t): admissible and consistent (see cost.py)."""
    return NodeHeuristic(cm.h_adm(target, w), cm.tg.head_list, name="h_adm")


def perfect_heuristic(state_cost_to_go: np.ndarray) -> StateHeuristic:
    """h*: exact optimal cost-to-go of every state from a backward Dijkstra run."""
    return StateHeuristic(state_cost_to_go, name="h_star")


class TunedClassicalHeuristic:
    """Week-4 fallback guide: h_tuned = kappa(w) * h_adm with log kappa linear in w.

    kappa is fitted by least squares on log(h* / h_adm) over the training labels. It uses no node
    information, so it is a tuned version of the classical heuristic, not a learned one.
    """

    def __init__(self, coef: np.ndarray):
        self.coef = np.asarray(coef, dtype=float)   # [intercept, w_1 .. w_5]

    @classmethod
    def fit(cls, W: np.ndarray, log_ratio: np.ndarray) -> "TunedClassicalHeuristic":
        X = np.column_stack([np.ones(len(W)), W])
        coef, *_ = np.linalg.lstsq(X, log_ratio, rcond=None)
        return cls(coef)

    def kappa(self, w) -> float:
        w = np.asarray(w, dtype=float)
        return float(max(1.0, np.exp(self.coef[0] + w @ self.coef[1:])))

    def heuristic(self, cm: CostModel, target: int, w) -> NodeHeuristic:
        return NodeHeuristic(self.kappa(w) * cm.h_adm(target, w), cm.tg.head_list, name="h_tuned")

    def to_dict(self) -> dict:
        return {"coef": self.coef.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "TunedClassicalHeuristic":
        return cls(np.array(d["coef"]))
