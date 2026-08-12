"""
Solar eclipse engine -- prototype for the C++ port.

Layer 1  time scales   : TT <-> UT1 via Delta-T  (Espenak & Meeus polynomials)
Layer 2  ephemeris     : apparent geocentric Sun/Moon, equator & equinox of date
Layer 3  shadow geometry: Besselian elements -> gamma, type, location, duration

Units: distances in km internally, Earth equatorial radii where Besselian.
"""
import os

import numpy as np
from skyfield.api import load
from skyfield import framelib

# ----------------------------------------------------------------- constants
AE      = 6378.137                  # Earth equatorial radius, km (WGS84)
FLAT    = 1.0 / 298.257223563
E2      = 2 * FLAT - FLAT ** 2
POLAR   = AE * (1 - FLAT)
R_SUN   = 696000.0                  # km
K_PEN   = 0.2725076                 # lunar radius factor, penumbral contacts
K_UMB   = 0.2722810                 # lunar radius factor, umbral contacts
K_MOON  = K_PEN
R_MOON  = K_PEN * AE
R_MOON_U= K_UMB * AE
DAY     = 86400.0

_FRAME = framelib.true_equator_and_equinox_of_date
_ts = load.timescale()
_eph = _earth = _sun = _moon = None

DEFAULT_KERNEL = os.environ.get("SOLAR_ECLIPSE_KERNEL", "de406.bsp")


def use_kernel(path=None):
    """Bind the ephemeris backend. Any JPL DE kernel Skyfield can read works;
    de406.bsp (-3000..+3000) is the default. This is the swap point for a
    different ephemeris implementation."""
    global _eph, _earth, _sun, _moon
    _eph = load(path or DEFAULT_KERNEL)
    _earth, _sun, _moon = _eph["earth"], _eph["sun"], _eph["moon"]
    return _eph


def _ensure():
    if _eph is None:
        use_kernel()


# ----------------------------------------------------- Layer 1: time scales
def delta_t(year):
    """Delta-T = TT - UT1 in seconds. Espenak & Meeus piecewise polynomials."""
    y = np.asarray(year, dtype=float)
    out = np.empty_like(y)

    def seg(mask, val):
        out[mask] = val[mask] if isinstance(val, np.ndarray) else val

    u = (y - 1820) / 100
    out[:] = -20 + 32 * u ** 2                                   # default/far

    m = y < -500
    seg(m, -20 + 32 * u ** 2)

    m = (y >= -500) & (y < 500)
    u2 = y / 100
    seg(m, (10583.6 - 1014.41 * u2 + 33.78311 * u2**2 - 5.952053 * u2**3
            - 0.1798452 * u2**4 + 0.022174192 * u2**5 + 0.0090316521 * u2**6))

    m = (y >= 500) & (y < 1600)
    u2 = (y - 1000) / 100
    seg(m, (1574.2 - 556.01 * u2 + 71.23472 * u2**2 + 0.319781 * u2**3
            - 0.8503463 * u2**4 - 0.005050998 * u2**5 + 0.0083572073 * u2**6))

    m = (y >= 1600) & (y < 1700)
    t = y - 1600
    seg(m, 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129)

    m = (y >= 1700) & (y < 1800)
    t = y - 1700
    seg(m, 8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3 - t**4 / 1174000)

    m = (y >= 1800) & (y < 1860)
    t = y - 1800
    seg(m, (13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3
            - 0.00037436 * t**4 + 0.0000121272 * t**5 - 0.0000001699 * t**6
            + 0.000000000875 * t**7))

    m = (y >= 1860) & (y < 1900)
    t = y - 1860
    seg(m, (7.62 + 0.5737 * t - 0.251754 * t**2 + 0.01680668 * t**3
            - 0.0004473624 * t**4 + t**5 / 233174))

    m = (y >= 1900) & (y < 1920)
    t = y - 1900
    seg(m, -2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3 - 0.000197 * t**4)

    m = (y >= 1920) & (y < 1941)
    t = y - 1920
    seg(m, 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3)

    m = (y >= 1941) & (y < 1961)
    t = y - 1950
    seg(m, 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547)

    m = (y >= 1961) & (y < 1986)
    t = y - 1975
    seg(m, 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718)

    m = (y >= 1986) & (y < 2005)
    t = y - 2000
    seg(m, (63.86 + 0.3345 * t - 0.060374 * t**2 + 0.0017275 * t**3
            + 0.000651814 * t**4 + 0.00002373599 * t**5))

    m = (y >= 2005) & (y < 2050)
    t = y - 2000
    seg(m, 62.92 + 0.32217 * t + 0.005589 * t**2)

    m = (y >= 2050) & (y < 2150)
    seg(m, -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y))

    return out


def jd_to_year(jd):
    """Decimal year (Julian/Gregorian mixed is irrelevant at this precision)."""
    return 2000.0 + (jd - 2451545.0) / 365.25


def gast_rad(jd_tt):
    """Greenwich apparent sidereal time [rad], using our own Delta-T."""
    jd_tt = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    dt = delta_t(jd_to_year(jd_tt))
    jd_ut1 = jd_tt - dt / DAY
    T = (jd_tt - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0)
            + 0.000387933 * T**2 - T**3 / 38710000.0)
    t = _ts.tt_jd(jd_tt)
    eqeq = (t.gast - t.gmst + 12.0) % 24.0 - 12.0        # hours, nutation only
    return np.radians((gmst + eqeq * 15.0) % 360.0)


# ------------------------------------------------------- Layer 2: ephemeris
def sun_moon(jd_tt):
    """Apparent geocentric Sun & Moon, equator/equinox of date, km. (3,N) each."""
    _ensure()
    t = _ts.tt_jd(np.atleast_1d(jd_tt))
    e = _earth.at(t)
    S = e.observe(_sun).apparent().frame_xyz(_FRAME).km
    M = e.observe(_moon).apparent().frame_xyz(_FRAME).km
    return np.atleast_2d(S), np.atleast_2d(M)


# -------------------------------------------------- Layer 3: shadow geometry
def axis_distance(jd_tt):
    """Perpendicular distance Earth-centre <-> shadow axis, in Earth radii."""
    S, M = sun_moon(jd_tt)
    u = S - M
    u = u / np.linalg.norm(u, axis=0)
    perp = M - np.sum(M * u, axis=0) * u
    return np.linalg.norm(perp, axis=0) / AE


def besselian(jd_tt):
    """Full Besselian element set + auxiliaries."""
    S, M = sun_moon(jd_tt)
    g = S - M
    Dsm = np.linalg.norm(g, axis=0)
    a = np.arctan2(g[1], g[0])
    d = np.arctan2(g[2], np.hypot(g[0], g[1]))

    sa, ca, sd, cd = np.sin(a), np.cos(a), np.sin(d), np.cos(d)

    def to_bes(V):
        x = -V[0] * sa + V[1] * ca
        y = -V[0] * sd * ca - V[1] * sd * sa + V[2] * cd
        z = V[0] * cd * ca + V[1] * cd * sa + V[2] * sd
        return x / AE, y / AE, z / AE

    xm, ym, zm = to_bes(M)

    sf1 = (R_SUN + R_MOON) / Dsm
    sf2 = (R_SUN - R_MOON_U) / Dsm
    f1, f2 = np.arcsin(sf1), np.arcsin(sf2)
    c1 = zm + K_PEN / sf1
    c2 = zm - K_UMB / sf2
    l1 = c1 * np.tan(f1)
    l2 = c2 * np.tan(f2)

    rho1 = np.sqrt(1 - E2 * cd**2)
    return dict(x=xm, y=ym, z=zm, d=d, a=a, l1=l1, l2=l2,
                tanf1=np.tan(f1), tanf2=np.tan(f2), rho1=rho1,
                S=S, M=M, Dsm=Dsm)


def axis_hits_earth(S, M):
    """Exact line/ellipsoid intersection. Returns (hit, point_km 3xN)."""
    u = S - M
    u = u / np.linalg.norm(u, axis=0)
    sc = np.array([1.0, 1.0, 1.0 / (1 - FLAT)])[:, None]
    Ms, us = M * sc, u * sc
    A = np.sum(us * us, axis=0)
    B = 2 * np.sum(Ms * us, axis=0)
    C = np.sum(Ms * Ms, axis=0) - AE**2
    disc = B**2 - 4 * A * C
    hit = disc > 0
    sq = np.sqrt(np.maximum(disc, 0))
    s1 = (-B - sq) / (2 * A)          # near side (towards the Sun)
    s2 = (-B + sq) / (2 * A)
    s = np.where(np.abs(s1) < np.abs(s2), s1, s2)
    P = M + s * u
    return hit, P


def ecef_to_geodetic(P, gast):
    """Equator-of-date XYZ (km, on the ellipsoid) -> geodetic lat/lon degrees."""
    cg, sg = np.cos(gast), np.sin(gast)
    xe = P[0] * cg + P[1] * sg
    ye = -P[0] * sg + P[1] * cg
    ze = P[2]
    lon = np.degrees(np.arctan2(ye, xe))
    lat = np.degrees(np.arctan2(ze, (1 - E2) * np.hypot(xe, ye)))
    return lat, (lon + 180) % 360 - 180


def geodetic_to_ecef(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    N = AE / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.array([N * np.cos(la) * np.cos(lo),
                     N * np.cos(la) * np.sin(lo),
                     N * (1 - E2) * np.sin(la)])


def observer_xyz(ecef, jd_tt):
    """Rotate fixed ECEF observer into equator-of-date frame at time(s)."""
    g = gast_rad(jd_tt)
    cg, sg = np.cos(g), np.sin(g)
    return np.array([ecef[0] * cg - ecef[1] * sg,
                     ecef[0] * sg + ecef[1] * cg,
                     ecef[2] * np.ones_like(cg)])


def topocentric(ecef, jd_tt):
    """Angular separation and apparent radii of Sun/Moon from a fixed site."""
    S, M = sun_moon(jd_tt)
    O = observer_xyz(ecef, jd_tt)
    ds, dm = S - O, M - O
    rs = np.linalg.norm(ds, axis=0)
    rm = np.linalg.norm(dm, axis=0)
    cospsi = np.clip(np.sum(ds * dm, axis=0) / (rs * rm), -1, 1)
    sep = np.arccos(cospsi)
    Rs = np.arcsin(np.clip(R_SUN / rs, -1, 1))
    Rm = np.arcsin(np.clip(R_MOON_U / rm, -1, 1))   # umbral k
    Rmp = np.arcsin(np.clip(R_MOON / rm, -1, 1))    # penumbral k
    alt = np.sum(ds * O, axis=0) / (rs * np.linalg.norm(O, axis=0))   # cos(zenith)
    return sep, Rs, Rm, alt, Rmp


def jd_to_cal(jd):
    """Proleptic Gregorian calendar from Julian Date (no time-scale semantics)."""
    jd = np.atleast_1d(np.asarray(jd, float))
    z = np.floor(jd + 0.5).astype(np.int64)
    f = jd + 0.5 - z
    a = z + 1 + (z - 1867216.25) // 36524.25 - ((z - 1867216.25) // 36524.25) // 4
    a = np.where(z < 2299161, z, a).astype(np.int64)
    alpha = ((z - 1867216.25) // 36524.25).astype(np.int64)
    a = np.where(z < 2299161, z, z + 1 + alpha - alpha // 4)
    b = a + 1524
    c = ((b - 122.1) / 365.25).astype(np.int64)
    d = (365.25 * c).astype(np.int64)
    e = ((b - d) / 30.6001).astype(np.int64)
    day = b - d - (30.6001 * e).astype(np.int64)
    month = np.where(e < 14, e - 1, e - 13)
    year = np.where(month > 2, c - 4716, c - 4715)
    secs = f * 86400.0
    hh = (secs // 3600).astype(int); mm = ((secs % 3600) // 60).astype(int)
    ss = secs % 60
    return year, month, day, hh, mm, ss


def fmt(jd):
    y, mo, d, hh, mm, ss = jd_to_cal(jd)
    return np.array([f"{a:+05d}-{b:02d}-{c:02d} {h:02d}:{m:02d}:{s:04.1f}"
                     for a, b, c, h, m, s in zip(y, mo, d, hh, mm, ss)])
