"""Plotting: fields of the sphere on maps (`maps`), with pyslfp's defaults.

The Love numbers have plots of their own, in `pyslfp.love_numbers`. Radial
profiles on a material's mesh, viscosities and kernels among them, are drawn
with planetmodel's profile plots (`plot_profile`, `plot_viscosity`,
`plot_log_viscosity`).
"""

from .profiles import plot_log_viscosity, plot_profile, plot_viscosity, radial_fields
from .maps import (
    create_map_figure,
    plot,
    plot_coastline,
    plot_displacement,
    plot_points,
    subplots,
)

__all__ = [
    "plot",
    "plot_displacement",
    "plot_points",
    "create_map_figure",
    "plot_coastline",
    "subplots",
    "plot_profile",
    "plot_viscosity",
    "plot_log_viscosity",
    "radial_fields",
]
