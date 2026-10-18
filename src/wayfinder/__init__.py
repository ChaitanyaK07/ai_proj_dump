"""Routes That Fit You: a personalised multi-objective walking route planner with learned search heuristics."""

__version__ = "0.1.0"

WEIGHT_KEYS = ("time", "junction", "traffic", "canopy", "climb")
"""The five sliders, always in this order. ``junction`` is ``w_turn`` in the cost formulas."""

PER_METRE_KEYS = ("time", "traffic", "canopy", "climb")
"""The four per-metre features ``f_k(e)``; junction load is priced per transition instead."""
