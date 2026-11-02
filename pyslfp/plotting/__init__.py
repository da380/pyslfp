"""Plotting: fields of the sphere on maps (`maps`), with pyslfp's defaults.

The Love numbers have plots of their own, in `pyslfp.love_numbers`.
"""

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
]
