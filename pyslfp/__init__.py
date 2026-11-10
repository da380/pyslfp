"""
pyslfp: A physical engine for sea-level fingerprinting and glacial isostatic adjustment.

One sea-level engine, two drivers: `core` holds the Earth model, the
background state, the sea level equation solvers, the spaces of loads,
responses and histories, and the operators both drivers share; `elastic`
wraps the solvers as instantaneous fingerprints and `viscoelastic` as
forward GIA in time, each with its pygeoinf operators.  `love_numbers`
solves the radial problem that serves both, `observations` samples the
response as tide gauges, altimetry and GRACE do, and `models` builds model
spaces for inversions from named load components.
"""

# ---------------------------------------------------------------------------
# 1. The shared core: configuration, state and the sea level engine
# ---------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core import EarthModel, EarthModelParameters, EarthState, SeaLevelEquation
from .love_numbers import LoveNumbers

# ---------------------------------------------------------------------------
# 2. The elastic driver
# ---------------------------------------------------------------------------
from .elastic import LinearSeaLevelEquation

# ---------------------------------------------------------------------------
# 3. Visualization Utilities
# ---------------------------------------------------------------------------
from .plotting import (
    plot,
    plot_displacement,
    plot_points,
    plot_profile,
    plot_viscosity,
    create_map_figure,
    plot_coastline,
    subplots,
)

# ---------------------------------------------------------------------------
# 4. Thread control for the numerical backends
# ---------------------------------------------------------------------------
from .core.parallel import set_num_threads, physical_core_count

# ---------------------------------------------------------------------------
# 5. Expose Sub-packages for Namespaced Access
# ---------------------------------------------------------------------------
# Allows users to do: `pyslfp.core.ocean_average_operator(...)`
from . import core
from . import elastic
from . import ice
from . import love_numbers
from . import models
from . import observations
from . import plotting
from . import viscoelastic

# ---------------------------------------------------------------------------
# 6. Define the Top-Level API (__all__)
# ---------------------------------------------------------------------------
try:
    __version__ = _version("pyslfp")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "4.1.0"

__all__ = [
    "__version__",
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
    "plot_displacement",
    "plot_profile",
    "plot_viscosity",
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
    "love_numbers",
    "models",
    "observations",
    "plotting",
    "viscoelastic",
]
