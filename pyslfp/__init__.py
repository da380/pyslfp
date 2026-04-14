"""
pyslfp: A physical engine for sea-level fingerprinting and glacial isostatic adjustment.
"""

# ---------------------------------------------------------------------------
# 1. Core Configuration & State Containers
# ---------------------------------------------------------------------------
from .core import EarthModelParameters, LoveNumbers, EarthModel
from .state import EarthState

# ---------------------------------------------------------------------------
# 2. Physics Engine & Solvers
# ---------------------------------------------------------------------------
from .physics import SeaLevelEquation, LinearSeaLevelEquation


# ---------------------------------------------------------------------------
# 4. Visualization Utilities
# ---------------------------------------------------------------------------
from .plotting import plot, plot_points, create_map_figure, plot_coastline

# ---------------------------------------------------------------------------
# 5. Expose Sub-packages for Namespaced Access
# ---------------------------------------------------------------------------
# Allows users to do: `pyslfp.linear_operators.ocean_average_operator(...)`
from . import linear_operators
from . import ice


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
    # Sub-packages
    "linear_operators",
    "ice",
]
