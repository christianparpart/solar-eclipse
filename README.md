# solar-eclipse

Solar eclipse prediction — **when**, **what type**, **where**, and **for how long** — for any epoch a JPL DE kernel covers. Reference implementation in Python; intended as the algorithmic spec for a C++ port.

Reproduces the NASA/Espenak *Five Millennium Canon* to within γ ≈ 2×10⁻⁴, magnitude ≈ 5×10⁻⁴, duration ≈ 0.3 s and timing ≈ 2 s.

## Quickstart

```sh
pip install -r requirements.txt
./tools/fetch_ephemeris.sh          # JPL DE406, ~287 MiB, spans -3000..+3000
PYTHONPATH=src python -m solar_eclipse.catalog 0 2500
```

```
5927 eclipses, 0..2500, 164s -> solar_eclipses_0000_2500.csv
```

Columns: `greatest_eclipse_UT1, deltaT_s, type, gamma, magnitude, lat, lon,
central_dur_s, partial_dur_s, path_dur_min`.

## Results, AD 0–2500

| Type | Count | Share | Canon |
|---|---:|---:|---:|
| Partial | 2118 | 35.7 % | 35.3 % |
| Annular | 1959 | 33.1 % | 33.2 % |
| Total | 1605 | 27.1 % | 26.7 % |
| Hybrid | 245 | 4.1 % | 4.8 % |
| **All** | **5927** | 2.370 /yr | 2.380 /yr |

Longest totality in range: **2186-07-16, 7m29.0s** (7.4°N 46.5°W).
Longest annularity: **0150-12-06, 12m23.4s** (11.6°N 153.4°W).

## Architecture

Three layers, deliberately separable — the seams are where a C++ port will want
to substitute implementations.

### 1. Time scales (`core.delta_t`, `core.gast_rad`)

The ephemeris runs on **TT**; Earth's rotation — which decides *where* the
shadow lands — runs on **UT1**. ΔT = TT − UT1 couples them, via the
Espenak & Meeus piecewise polynomials.

In C++ this should be **type-level**: `Instant<TT>` and `Instant<UT1>` as
distinct strong types, converted only through an explicit `DeltaTModel`.
Silently mixing them does not crash; it displaces the track in longitude.

### 2. Ephemeris (`core.sun_moon`, `core.use_kernel`)

Apparent geocentric Sun and Moon, light-time and aberration corrected, in the
true equator and equinox of date. Backed by a JPL DE kernel via Skyfield.

`use_kernel()` is the single swap point. A C++ port would define this as a
concept with two backends: an analytic one (truncated VSOP87 + ELP-2000/82,
zero dependencies, seconds-level over ±1000 years) and a JPL DE reader
(CALCEPH is the least painful route).

### 3. Shadow geometry (`core.besselian`, `search`, `catalog`)

Everything reduces to Besselian elements `(x, y, d, μ, l₁, l₂, tan f₁, tan f₂)`
in the fundamental plane; global circumstances, γ, magnitude, the central line
and local circumstances all derive from that one computation.

Search: seed from the mean-new-moon series → refine to conjunction in apparent
ecliptic longitude → Newton on d/dt of the squared axis distance to find
greatest eclipse → classify.

## Three things that will bite a reimplementation

1. **Two lunar radius factors, not one.** `k = 0.2725076` for *penumbral*
   contacts, `k = 0.2722810` for *umbral*. Using a single value put durations
   2.7 % long and shifted magnitudes in the 4th decimal. See `core.K_PEN` /
   `core.K_UMB`.

2. **Hybrid detection needs end-clustered sampling.** Annular segments of a
   hybrid are often confined to the first and last minutes of the path;
   uniform sampling along the central line silently misclassifies them as
   plain totals (2013-11-03, 2023-04-20 both failed this way). `catalog.hybrid_scan`
   samples cosine-clustered towards the ends, and tests character *exactly* —
   is the umbral cone vertex short of the surface? — rather than by
   differencing topocentric angular radii, which cancels badly near the limb.

3. **ΔT, not the ephemeris, is the accuracy floor going backward.** At AD 0 the
   polynomial gives ≈ +2h50m with several minutes of uncertainty — tens of
   degrees of longitude in the greatest-eclipse point. γ, type, magnitude and
   duration are all ΔT-independent and remain good; **only longitude degrades**.
   A `DeltaTModel` should therefore return a value *and* a σ, so the API can
   emit a longitude uncertainty band rather than a false-precision coordinate.

## Testing

```sh
PYTHONPATH=src python -m pytest tests -q
```

Regression fixtures are published circumstances for six eclipses spanning
total/annular/hybrid, plus five known hybrids, ΔT segment-boundary continuity,
and a check that the Newton refinement lands on a minimum. These are the tests
that caught bugs (1) and (2) above.

## Accuracy notes

- Lunar limb irregularities (Watts profiles) are **not** modelled; real contact
  times at a given site vary by ~1–2 s from the mean-limb values here.
- Duration is reported at the point of greatest eclipse, matching canon
  convention — not the maximum duration anywhere along the path.
- Type shares drift by era; the hybrid fraction alone ranges from 6 to 50 per
  250 years across this window, so a single global percentage is a weak check.

## License

Apache-2.0.
