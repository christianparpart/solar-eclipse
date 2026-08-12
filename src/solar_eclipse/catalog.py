"""Bulk catalogue generation: sweep a year range and classify every eclipse."""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

from .core import (
    DAY, R_SUN, R_MOON_U, besselian, axis_hits_earth,
    delta_t, jd_to_year, jd_to_cal, fmt,
)
from .search import (
    mean_new_moons, refine_conjunction, refine_greatest, classify,
    greatest_point, durations, character, _bisect,
)

COLUMNS = [
    "greatest_eclipse_UT1", "deltaT_s", "type", "gamma", "magnitude",
    "lat", "lon", "central_dur_s", "partial_dur_s", "path_dur_min",
]


@dataclass
class Eclipse:
    greatest_eclipse_UT1: str
    deltaT_s: float
    type: str
    gamma: float
    magnitude: float
    lat: float
    lon: float
    central_dur_s: float
    partial_dur_s: float
    path_dur_min: float


def _m1(t):
    b = besselian(t)
    return np.hypot(b["x"], b["y"] / b["rho1"]) - 1.0


def hybrid_scan(jd, n=25):
    """Walk the central line and test total-vs-annular character at each step.

    Sampling is cosine-clustered towards the path ends: hybrid eclipses very
    often have their annular segments confined to the first/last few minutes,
    and uniform sampling misses them (e.g. 2013-11-03, 2023-04-20).

    The character test is exact rather than differential -- it asks whether the
    umbral cone vertex falls short of the surface (antumbra -> annular). This
    avoids the catastrophic cancellation of subtracting topocentric angular
    radii near the limb.
    """
    W = 5.0 / 24.0
    t0 = _bisect(lambda t: -_m1(t), jd - W, jd, 26)
    t1 = _bisect(_m1, jd, jd + W, 26)
    total = np.zeros_like(jd, dtype=bool)
    annular = np.zeros_like(jd, dtype=bool)
    u = np.linspace(0, 1, n)
    for fr in 0.001 + 0.998 * 0.5 * (1 - np.cos(np.pi * u)):
        t = t0 + (t1 - t0) * fr
        b = besselian(t)
        hit, P = axis_hits_earth(b["S"], b["M"])
        sin_f2 = (R_SUN - R_MOON_U) / b["Dsm"]
        vertex = R_MOON_U / sin_f2
        d_surf = np.linalg.norm(P - b["M"], axis=0)
        total |= hit & (d_surf < vertex)
        annular |= hit & (d_surf > vertex)
    return total & annular, (t1 - t0) * DAY


def sweep(year_from, year_to, chunk=3000, progress=None):
    """Yield every solar eclipse with greatest eclipse in [year_from, year_to]."""
    seeds_all = mean_new_moons(year_from - 1, year_to + 2)
    for i in range(0, len(seeds_all), chunk):
        jd = refine_greatest(refine_conjunction(seeds_all[i:i + chunk]), iters=5)
        c = classify(jd)
        keep = c["penumbral"]
        if not keep.any():
            continue
        jd = jd[keep]
        c = {k: (v[..., keep] if np.ndim(v) else v) for k, v in c.items()}

        lat, lon = greatest_point(c, jd)
        d_rm, sep, Rs, Rm, alt = character(lat, lon, jd)
        cent, part = durations(lat, lon, jd, c["hit"])
        mag = np.where(c["hit"], Rm / Rs, (Rs + Rm - sep) / (2 * Rs))

        hyb = np.zeros_like(jd, dtype=bool)
        path = np.zeros_like(jd)
        if c["hit"].any():
            h, pd = hybrid_scan(jd[c["hit"]])
            hyb[c["hit"]] = h
            path[c["hit"]] = pd

        typ = np.where(~c["hit"], "Partial",
                       np.where(hyb, "Hybrid", np.where(d_rm > 0, "Total", "Annular")))
        dt = delta_t(jd_to_year(jd))
        ut = jd - dt / DAY
        yy = jd_to_cal(ut)[0]
        stamps = fmt(ut)

        for j in range(len(jd)):
            if not (year_from <= yy[j] <= year_to):
                continue
            yield Eclipse(stamps[j], round(float(dt[j]), 1), str(typ[j]),
                          round(float(c["gamma"][j]), 4), round(float(mag[j]), 4),
                          round(float(lat[j]), 2), round(float(lon[j]), 2),
                          round(float(cent[j]), 1), round(float(part[j]), 1),
                          round(float(path[j]) / 60, 1))
        if progress:
            progress(int(jd_to_year(jd)[-1]))


def write_csv(path, events):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    y0 = int(argv[0]) if argv else 0
    y1 = int(argv[1]) if len(argv) > 1 else 2500
    out = argv[2] if len(argv) > 2 else f"solar_eclipses_{y0:04d}_{y1:04d}.csv"
    t0 = time.time()
    rows = list(sweep(y0, y1, progress=lambda y: print(f"  ...{y:5d}", flush=True)))
    write_csv(out, rows)
    print(f"{len(rows)} eclipses, {y0}..{y1}, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
