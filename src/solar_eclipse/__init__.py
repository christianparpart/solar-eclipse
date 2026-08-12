"""Solar eclipse computation: local circumstances and bulk catalogues.

Three layers, deliberately separable (see README):
  core    -- time scales (Delta-T), ephemeris backend, Besselian geometry
  search  -- conjunction search, greatest-eclipse refinement, classification
  catalog -- bulk sweep over a year range
"""
from .core import use_kernel, delta_t, besselian, DEFAULT_KERNEL  # noqa: F401
from .search import (  # noqa: F401
    mean_new_moons, refine_conjunction, refine_greatest,
    classify, greatest_point, durations, character,
)
from .catalog import Eclipse, sweep, write_csv, hybrid_scan  # noqa: F401

__version__ = "0.1.0"
__all__ = ["use_kernel", "delta_t", "besselian", "Eclipse", "sweep",
           "write_csv", "hybrid_scan", "classify", "durations"]
