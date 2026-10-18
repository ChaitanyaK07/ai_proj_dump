"""The five feature extractors of the cost model (proposal, Section 3).

Four are per-metre scores ``f_k(e) in [0, 1]`` (walking time, traffic exposure, canopy, climb); junction
load is priced per transition between two edges from the turn angle and the node degree.

| Feature           | Source                               | Status            |
|-------------------|--------------------------------------|-------------------|
| Walking time      | OSM highway class, segment length    | measured          |
| Junction load     | node degree, way geometry (bearings) | derived, exact    |
| Traffic exposure  | ranking over OSM highway class       | proxy             |
| Canopy proximity  | ESA WorldCover 2021, 15 m buffer     | proxy (not shade) |
| Climb             | Copernicus GLO-30 DEM; highway=steps | derived, reliable |
"""
from __future__ import annotations

import math

import numpy as np

from wayfinder import geo

FEATURE_STATUS = {
    "time": ("OSM highway class, segment length", "measured"),
    "junction": ("node degree, way geometry", "derived, exact"),
    "traffic": ("ranking over OSM highway class", "proxy"),
    "canopy": ("ESA WorldCover 2021 tree cover, 15 m buffer", "proxy"),
    "climb": ("Copernicus GLO-30 DEM grade; highway=steps", "derived, reliable"),
}

# Assumed walking speed per highway class, m/s (1.3 on footways down to 0.5 on steps).
SPEED = {"footway": 1.3, "path": 1.3, "pedestrian": 1.3, "cycleway": 1.3, "corridor": 1.3,
         "track": 1.2, "living_street": 1.25, "residential": 1.2, "service": 1.2, "unclassified": 1.2,
         "road": 1.2, "tertiary": 1.15, "secondary": 1.1, "primary": 1.1, "trunk": 1.0, "steps": 0.5}
DEFAULT_SPEED = 1.2
MIN_SPEED = 0.5                  # steps; normalises the time feature to [0, 1]

# Ordinal traffic-exposure proxy. OSM carries no traffic-volume tag.
TRAFFIC = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 0.8, "tertiary": 0.6,
           "unclassified": 0.4, "road": 0.4, "residential": 0.3, "living_street": 0.2, "service": 0.2,
           "track": 0.1, "footway": 0.0, "path": 0.0, "pedestrian": 0.0, "cycleway": 0.0,
           "corridor": 0.0, "steps": 0.0}
DEFAULT_TRAFFIC = 0.3

GRADE_CAP = 0.15                 # uphill grade clipped at 15% (30 m DEM noise)
STAIRS_CLIMB = 1.0               # fixed climb penalty for highway=steps

# Junction load
TURN_CLASSES = ((30.0, 0.0), (75.0, 0.3), (150.0, 1.0))   # (upper angle bound, penalty)
UTURN_PENALTY = 3.0
JUNCTION_PENALTY = 0.2           # passing through a node with >= 3 neighbours


def edge_speed(highway: str) -> float:
    """Walking speed of an edge; merged edges take their slowest class."""
    return min(SPEED.get(c, DEFAULT_SPEED) for c in highway.split("|"))


def time_feature(highway: np.ndarray) -> np.ndarray:
    """Seconds per metre, normalised so steps (0.5 m/s) score 1."""
    return np.array([MIN_SPEED / edge_speed(h) for h in highway])


def traffic_feature(highway: np.ndarray) -> np.ndarray:
    """Ordinal exposure; merged edges take their busiest class."""
    return np.array([max(TRAFFIC.get(c, DEFAULT_TRAFFIC) for c in h.split("|")) for h in highway])


def canopy_feature(canopy: np.ndarray) -> np.ndarray:
    """Cost of *lacking* tree cover: 1 - tree-cover share in the 15 m buffer."""
    return 1.0 - np.clip(canopy, 0.0, 1.0)


def climb_feature(grade: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Uphill grade max(0, rise) / length clipped at 15%, scaled to [0, 1]; stairs get a fixed 1."""
    f = np.clip(grade, 0.0, GRADE_CAP) / GRADE_CAP
    return np.where(steps, STAIRS_CLIMB, f)


def per_metre_features(rg) -> np.ndarray:
    """(m, 4) matrix of ``f_k(e)`` in the order of ``wayfinder.PER_METRE_KEYS``."""
    F = np.column_stack([time_feature(rg.highway), traffic_feature(rg.highway),
                         canopy_feature(rg.canopy), climb_feature(rg.grade, rg.steps)])
    assert F.min() >= 0.0 and F.max() <= 1.0, "features must be per-metre scores in [0, 1]"
    return F


def angle_penalty(b_in: float, b_out: float) -> float:
    if math.isnan(b_in) or math.isnan(b_out):
        return 0.0
    d = geo.angle_between(b_in, b_out)
    for bound, pen in TURN_CLASSES:
        if d < bound:
            return pen
    return UTURN_PENALTY


def turn_penalty(b_in: float, b_out: float, node_degree: int) -> float:
    """Junction load of moving from an edge arriving with bearing ``b_in`` onto one leaving with ``b_out``."""
    return angle_penalty(b_in, b_out) + (JUNCTION_PENALTY if node_degree >= 3 else 0.0)
