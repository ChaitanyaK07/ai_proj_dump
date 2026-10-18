"""Geodesy helpers: UTM <-> WGS84, haversine distance, bearings and WKT parsing.

The graph is stored in UTM zone 44N (EPSG:32644, metres). Edge lengths were computed by OSMnx as sums
of great-circle segment lengths on a sphere of radius 6,371,009 m, so the haversine distance on that
same sphere is a lower bound on every path length. That is what makes ``h_adm`` admissible.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_009.0      # the radius OSMnx uses for edge lengths

# WGS84 ellipsoid and UTM constants
_A = 6_378_137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996
UTM_ZONE = 44                       # Hyderabad
UTM_EPSG = 32644


def central_meridian(zone: int) -> float:
    return (zone - 1) * 6 - 180 + 3


def latlon_to_utm(lat, lon, zone: int = UTM_ZONE):
    """WGS84 degrees -> UTM easting/northing (northern hemisphere), Snyder (1987) series."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.asarray(lon, dtype=float)
    lam = np.radians(lon - central_meridian(zone))
    e2, ep2 = _E2, _EP2
    e4, e6 = e2 * e2, e2 * e2 * e2
    sin, cos, tan = np.sin(lat), np.cos(lat), np.tan(lat)
    N = _A / np.sqrt(1 - e2 * sin ** 2)
    T = tan ** 2
    C = ep2 * cos ** 2
    A = lam * cos
    M = _A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
              - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * lat)
              + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * lat)
              - (35 * e6 / 3072) * np.sin(6 * lat))
    x = _K0 * N * (A + (1 - T + C) * A ** 3 / 6
                   + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500_000.0
    y = _K0 * (M + N * tan * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                              + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


def utm_to_latlon(x, y, zone: int = UTM_ZONE):
    """UTM easting/northing (northern hemisphere) -> WGS84 degrees, Snyder (1987) series."""
    x = np.asarray(x, dtype=float) - 500_000.0
    y = np.asarray(y, dtype=float)
    e2, ep2 = _E2, _EP2
    e4, e6 = e2 * e2, e2 * e2 * e2
    M = y / _K0
    mu = M / (_A * (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    sin1, cos1, tan1 = np.sin(phi1), np.cos(phi1), np.tan(phi1)
    C1 = ep2 * cos1 ** 2
    T1 = tan1 ** 2
    N1 = _A / np.sqrt(1 - e2 * sin1 ** 2)
    R1 = _A * (1 - e2) / (1 - e2 * sin1 ** 2) ** 1.5
    D = x / (N1 * _K0)
    lat = phi1 - (N1 * tan1 / R1) * (D ** 2 / 2
                                      - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
                                      + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2)
                                      * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120) / cos1
    return np.degrees(lat), central_meridian(zone) + np.degrees(lon)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres on the OSMnx sphere. Vectorised over numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing(p, q) -> float:
    """Planar bearing in degrees clockwise from grid north, or NaN for a zero-length segment."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    if dx == 0 and dy == 0:
        return math.nan
    return math.degrees(math.atan2(dx, dy)) % 360.0


def angle_between(b_in: float, b_out: float) -> float:
    """Absolute heading change in degrees, in [0, 180]."""
    return abs(((b_out - b_in + 180.0) % 360.0) - 180.0)


def parse_linestring(wkt: str) -> list[tuple[float, float]]:
    """Parse a WKT ``LINESTRING (x y, x y, ...)`` into a list of (x, y) tuples."""
    s = wkt.strip()
    if not s.upper().startswith("LINESTRING"):
        raise ValueError(f"not a LINESTRING: {s[:40]!r}")
    body = s[s.index("(") + 1: s.rindex(")")]
    pts = []
    for pair in body.split(","):
        parts = pair.split()
        pts.append((float(parts[0]), float(parts[1])))
    return pts
