"""Regression tests against published circumstances (NASA/Espenak canon).

Tolerances are deliberately tight -- these are the numbers that caught both
the dual lunar-radius-factor bug and the hybrid-detection sampling bug.
"""
import numpy as np
import pytest

from solar_eclipse.core import DAY, delta_t, jd_to_year, fmt
from solar_eclipse.search import (
    refine_conjunction, refine_greatest, classify,
    greatest_point, durations, character,
)
from solar_eclipse.catalog import hybrid_scan

# date seed, gamma, magnitude, lat, lon, central duration [s], type
PUBLISHED = [
    ((1991, 7, 11), -0.0041, 1.0800, 22.00, -105.21, 413.1, "Total"),
    ((1999, 8, 11), +0.5062, 1.0286, 45.08, +24.30, 143.0, "Total"),
    ((2017, 8, 21), +0.4367, 1.0306, 36.97, -87.66, 160.2, "Total"),
    ((2024, 4, 8), +0.3431, 1.0566, 25.29, -104.13, 268.1, "Total"),
    ((2023, 10, 14), +0.3753, 0.9520, 11.37, -83.09, 317.0, "Annular"),
    ((2027, 8, 2), +0.1420, 1.0790, 25.50, +33.20, 383.0, "Total"),
]

KNOWN_HYBRIDS = [(1986, 10, 3), (2005, 4, 8), (2013, 11, 3),
                 (2023, 4, 20), (1930, 4, 28)]


def _jd(y, m, d):
    """Julian Date at 12h TT, proleptic Gregorian."""
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1))
            + d + b - 1524.5 + 0.5)


def _solve(seeds):
    jd = refine_greatest(refine_conjunction(np.array(seeds, dtype=float)), iters=5)
    c = classify(jd)
    lat, lon = greatest_point(c, jd)
    d_rm, sep, Rs, Rm, _ = character(lat, lon, jd)
    cent, part = durations(lat, lon, jd, c["hit"])
    mag = np.where(c["hit"], Rm / Rs, (Rs + Rm - sep) / (2 * Rs))
    return jd, c, lat, lon, mag, cent, d_rm


@pytest.mark.parametrize("case", PUBLISHED, ids=lambda c: "%04d-%02d-%02d" % c[0])
def test_published_circumstances(case):
    (y, m, d), gamma, mag, lat, lon, dur, kind = case
    jd, c, la, lo, mg, cent, d_rm = _solve([_jd(y, m, d)])

    assert c["gamma"][0] == pytest.approx(gamma, abs=2e-4)
    assert mg[0] == pytest.approx(mag, abs=5e-4)
    assert la[0] == pytest.approx(lat, abs=0.02)
    assert lo[0] == pytest.approx(lon, abs=0.02)
    assert cent[0] == pytest.approx(dur, abs=1.5)
    assert c["hit"][0]
    assert ("Total" if d_rm[0] > 0 else "Annular") == kind


@pytest.mark.parametrize("ymd", KNOWN_HYBRIDS, ids=lambda c: "%04d-%02d-%02d" % c)
def test_known_hybrids(ymd):
    """Annular segments can be minutes long and sit at the path ends."""
    jd, c, la, lo, mg, cent, d_rm = _solve([_jd(*ymd)])
    assert c["hit"][0]
    hyb, _ = hybrid_scan(jd)
    assert bool(hyb[0]), "expected annular-total (hybrid)"


def test_delta_t_continuity():
    """Piecewise Delta-T must not jump at segment boundaries."""
    edges = [-500, 500, 1600, 1700, 1800, 1860, 1900, 1920, 1941,
             1961, 1986, 2005, 2050, 2150]
    for e in edges:
        lo = delta_t(np.array([e - 1e-6]))[0]
        hi = delta_t(np.array([e + 1e-6]))[0]
        assert abs(hi - lo) < 2.5, f"Delta-T discontinuity at {e}: {lo} -> {hi}"


def test_greatest_eclipse_is_a_minimum():
    """Newton must land on the minimum of the axis distance, not a stationary
    point of the wrong sign."""
    from solar_eclipse.core import axis_distance
    jd = refine_greatest(refine_conjunction(
        np.array([_jd(2017, 8, 21), _jd(2024, 4, 8)])), iters=5)
    h = 600.0 / DAY
    assert np.all(axis_distance(jd) < axis_distance(jd + h))
    assert np.all(axis_distance(jd) < axis_distance(jd - h))
