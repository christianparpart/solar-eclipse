import numpy as np
from .core import *

H = 120.0 / DAY          # finite-difference step (2 min) in days


def mean_new_moons(y0, y1):
    k = np.arange(np.floor((y0 - 2000) * 12.3685) - 2,
                  np.ceil((y1 - 2000) * 12.3685) + 2)
    T = k / 1236.85
    jde = (2451550.09766 + 29.530588861 * k + 0.00015437 * T**2
           - 0.000000150 * T**3 + 0.00000000073 * T**4)
    return jde


def refine_conjunction(jd):
    """Pull the seed onto conjunction in apparent ecliptic longitude."""
    for _ in range(3):
        S, M = sun_moon(jd)
        # obliquity of date (mean is plenty here)
        T = (jd - 2451545.0) / 36525.0
        eps = np.radians(23.439291 - 0.0130042 * T)
        ce, se = np.cos(eps), np.sin(eps)
        lam_s = np.arctan2(S[1] * ce + S[2] * se, S[0])
        lam_m = np.arctan2(M[1] * ce + M[2] * se, M[0])
        dl = np.degrees((lam_m - lam_s + np.pi) % (2 * np.pi) - np.pi)
        jd = jd - dl / 12.19
    return jd


def refine_greatest(jd, iters=4):
    """Newton on d/dt of the squared axis distance -> instant of greatest eclipse."""
    for _ in range(iters):
        f0 = axis_distance(jd) ** 2
        fp = axis_distance(jd + H) ** 2
        fm = axis_distance(jd - H) ** 2
        d1 = (fp - fm) / (2 * H)
        d2 = (fp - 2 * f0 + fm) / (H * H)
        step = np.where(np.abs(d2) > 1e-12, -d1 / d2, 0.0)
        jd = jd + np.clip(step, -0.25, 0.25)
    return jd


def classify(jd):
    """Returns dict of per-event geometry at the instant of greatest eclipse."""
    b = besselian(jd)
    gamma = np.hypot(b['x'], b['y']) * np.sign(b['y'])
    m1 = np.hypot(b['x'], b['y'] / b['rho1'])          # flattened axis distance
    hit, P = axis_hits_earth(b['S'], b['M'])
    penumbral = m1 < (1.0 + b['l1'])
    return dict(gamma=gamma, m1=m1, hit=hit, P=P, penumbral=penumbral, **b)


def greatest_point(c, jd):
    """Lat/lon of greatest eclipse: axis footpoint if central, else limb point."""
    g = gast_rad(jd)
    lat, lon = ecef_to_geodetic(c['P'], g)
    # non-central: nearest surface point in the flattened (unit-sphere) frame
    r = np.maximum(c['m1'], 1e-12)
    xb, yb, zb = c['x'] / r, (c['y'] / r), 0.0
    yb = yb / c['rho1'] * c['rho1']          # eta1 -> eta  (zeta = 0)
    sa, ca = np.sin(c['a']), np.cos(c['a'])
    sd, cd = np.sin(c['d']), np.cos(c['d'])
    X = -xb * sa - yb * sd * ca + zb * cd * ca
    Y = xb * ca - yb * sd * sa + zb * cd * sa
    Z = yb * cd + zb * sd
    # project onto the ellipsoid
    V = np.array([X, Y, Z])
    sc = np.sqrt(1.0 / (V[0]**2 + V[1]**2 + (V[2] / (1 - FLAT))**2))
    latp, lonp = ecef_to_geodetic(V * sc * AE, g)
    return (np.where(c['hit'], lat, latp), np.where(c['hit'], lon, lonp))


def _bisect(fun, lo, hi, n=34):
    """Vectorised bisection; assumes fun(lo)<0<fun(hi) elementwise."""
    flo = fun(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
        flo = np.where(same, fm, flo)
    return 0.5 * (lo + hi)


def durations(lat, lon, jd, central):
    """Central (totality/annularity) and partial-phase duration, seconds,
    for an observer standing at the point of greatest eclipse."""
    ecef = geodetic_to_ecef(lat, lon)

    def gc(t):                       # central contacts (2nd/3rd)
        sep, Rs, Rm, _, _ = topocentric(ecef, t)
        return sep - np.abs(Rm - Rs)

    def gp(t):                       # partial contacts (1st/4th)
        sep, Rs, Rm, _, Rmp = topocentric(ecef, t)
        return sep - (Rs + Rmp)

    W = 3.0 / 24.0
    t1 = _bisect(lambda t: -gp(t), jd - W, jd)
    t4 = _bisect(gp, jd, jd + W)
    part = (t4 - t1) * DAY

    ok = central & (gc(jd) < 0)
    W2 = 20.0 / 1440.0
    t2 = _bisect(lambda t: -gc(t), jd - W2, jd)
    t3 = _bisect(gc, jd, jd + W2)
    cent = np.where(ok, (t3 - t2) * DAY, 0.0)
    return cent, part


def character(lat, lon, jd):
    """Total vs annular at a point: sign of Rm - Rs (topocentric)."""
    ecef = geodetic_to_ecef(lat, lon)
    sep, Rs, Rm, alt, Rmp = topocentric(ecef, jd)
    return Rm - Rs, sep, Rs, Rm, alt


def analyse(jd_seed):
    jd = refine_greatest(refine_conjunction(jd_seed))
    c = classify(jd)
    keep = c['penumbral']
    return jd, c, keep
