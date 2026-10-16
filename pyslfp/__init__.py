"""
pyslfp: A physical engine for sea-level fingerprinting and glacial isostatic adjustment.

One sea-level engine, two drivers: `core` holds the Earth model, the
background state and the sea level equation solvers; `elastic` wraps
them as instantaneous fingerprints and `viscoelastic` as forward GIA in
time.  `love_numbers` solves the radial problem that serves both, and
`linear_operators` is the inference layer over the elastic side.
"""

# ---------------------------------------------------------------------------
# 1. The shared core: configuration, state and the sea level engine
# ---------------------------------------------------------------------------
from .core import EarthModel, EarthModelParameters, EarthState, SeaLevelEquation
from .love_numbers import LoveNumbers

# ---------------------------------------------------------------------------
# 2. The elastic driver
# ---------------------------------------------------------------------------
from .elastic import LinearSeaLevelEquation

# ---------------------------------------------------------------------------
# 3. Visualization Utilities
# ---------------------------------------------------------------------------
from .plot import plot, plot_points, create_map_figure, plot_coastline, subplots

# ---------------------------------------------------------------------------
# 4. Thread control for the numerical backends
# ---------------------------------------------------------------------------
from .parallel import set_num_threads, physical_core_count

# ---------------------------------------------------------------------------
# 5. Expose Sub-packages for Namespaced Access
# ---------------------------------------------------------------------------
# Allows users to do: `pyslfp.linear_operators.ocean_average_operator(...)`
from . import core
from . import elastic
from . import ice
from . import linear_operators
from . import love_numbers
from . import viscoelastic

# ---------------------------------------------------------------------------
# 6. Define the Top-Level API (__all__)
# ---------------------------------------------------------------------------
__all__ = [
    # Core & State
    "EarthModelParameters",
    "LoveNumbers",
    "EarthModel",
    "EarthState",
    # Physics Solvers
    "SeaLevelEquation",
    "LinearSeaLevelEquation",
    # Visualization
    "plot",
    "plot_points",
    "create_map_figure",
    "plot_coastline",
    "subplots",
    # Thread control
    "set_num_threads",
    "physical_core_count",
    # Sub-packages
    "core",
    "elastic",
    "ice",
    "linear_operators",
    "love_numbers",
    "viscoelastic",
]
