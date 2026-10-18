import math

import numpy as np
import pytest

from wayfinder import geo
from wayfinder.graph import features as F


def test_utm_roundtrip_is_sub_millimetre():
    lat = np.array([17.39, 17.422, 17.45])
    lon = np.array([78.38, 78.42, 78.46])
    x, y = geo.latlon_to_utm(lat, lon)
    lat2, lon2 = geo.utm_to_latlon(x, y)
    assert np.allclose(lat, lat2, atol=1e-8) and np.allclose(lon, lon2, atol=1e-8)


def test_study_centre_projects_inside_graph_extent():
    x, y = geo.latlon_to_utm(17.4220, 78.4200)
    assert 220_000 < x < 232_000 and 1_924_000 < y < 1_932_000


def test_haversine_one_degree_of_latitude():
    d = geo.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(math.pi / 180 * geo.EARTH_RADIUS_M, rel=1e-12)


def test_haversine_triangle_inequality():
    rng = np.random.default_rng(1)
    p = rng.uniform([17.39, 78.38], [17.45, 78.46], size=(300, 2))
    a, b, c = p[:100], p[100:200], p[200:]
    ab = geo.haversine_m(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    bc = geo.haversine_m(b[:, 0], b[:, 1], c[:, 0], c[:, 1])
    ac = geo.haversine_m(a[:, 0], a[:, 1], c[:, 0], c[:, 1])
    assert np.all(ac <= ab + bc + 1e-6)


def test_bearing_and_angle():
    assert geo.bearing((0, 0), (0, 1)) == pytest.approx(0)
    assert geo.bearing((0, 0), (1, 0)) == pytest.approx(90)
    assert math.isnan(geo.bearing((1, 1), (1, 1)))
    assert geo.angle_between(350, 10) == pytest.approx(20)
    assert geo.angle_between(0, 180) == pytest.approx(180)


def test_parse_linestring():
    pts = geo.parse_linestring("LINESTRING (1.5 2, 3 4.25, 5 6)")
    assert pts == [(1.5, 2.0), (3.0, 4.25), (5.0, 6.0)]
    with pytest.raises(ValueError):
        geo.parse_linestring("POINT (1 2)")


@pytest.mark.parametrize("delta,expected", [(0, 0.0), (29, 0.0), (45, 0.3), (90, 1.0), (149, 1.0), (180, 3.0)])
def test_angle_penalty_classes(delta, expected):
    assert F.angle_penalty(10.0, 10.0 + delta) == expected


def test_junction_term_needs_three_neighbours():
    assert F.turn_penalty(0, 0, 2) == 0.0
    assert F.turn_penalty(0, 0, 3) == pytest.approx(F.JUNCTION_PENALTY)
    assert F.turn_penalty(0, 90, 4) == pytest.approx(1.0 + F.JUNCTION_PENALTY)
    assert F.turn_penalty(math.nan, 90, 2) == 0.0


def test_feature_definitions():
    hw = np.array(["footway", "steps", "trunk", "residential|primary", "unknown_class"])
    t = F.time_feature(hw)
    assert t[1] == pytest.approx(1.0) and t[0] == pytest.approx(0.5 / 1.3)
    assert t[3] == pytest.approx(0.5 / 1.1)                 # slowest class of a merged edge
    tr = F.traffic_feature(hw)
    assert tr[0] == 0.0 and tr[2] == 1.0 and tr[3] == 1.0 and tr[4] == F.DEFAULT_TRAFFIC
    assert np.allclose(F.canopy_feature(np.array([0.0, 0.25, 1.0])), [1.0, 0.75, 0.0])
    cl = F.climb_feature(np.array([-0.1, 0.075, 0.3, 0.0]), np.array([False, False, False, True]))
    assert np.allclose(cl, [0.0, 0.5, 1.0, 1.0])


def test_all_features_in_unit_interval(net):
    assert net.cm.F.shape == (net.rg.m, 4)
    assert net.cm.F.min() >= 0 and net.cm.F.max() <= 1
