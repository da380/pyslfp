"""Figures of Love numbers against degree, and of the radial solution of
one degree; matplotlib, radius drawn upward."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .solver import DegreeSolution
from .table import LoveNumbers

__all__ = ["plot_love_numbers", "plot_degree_solution"]


def _real(values: np.ndarray) -> np.ndarray:
    return values.real if values.dtype.kind == "c" else values


def plot_love_numbers(
    love: LoveNumbers, /, *, axes: Optional[np.ndarray] = None
) -> Tuple[Figure, np.ndarray]:
    """The load Love numbers and the tidal Love numbers against degree.

    The left axes show the dimensionless load numbers h', l l' and l k'
    from degree 1, the factor of l making the three comparable in size;
    the right axes the geodetic tidal numbers k^T, h^T, l^T from degree
    2. A column the table lacks (the tangential numbers of the shipped
    table) is left out, and a complex table is drawn by its real parts.

    Args:
        love: The Love numbers.
        axes: Two matplotlib axes to draw on; a new figure otherwise.

    Returns:
        The figure and the two axes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    else:
        fig = axes[0].figure
    load, tidal = axes
    degree = love.degree.astype(float)
    conv = love.conventional()
    keep = degree >= 1
    for values, label in (
        (conv["h"], "$h'$"),
        (degree * conv["l"], "$l\\,l'$"),
        (degree * conv["k"], "$l\\,k'$"),
    ):
        if np.all(np.isnan(values[keep])):
            continue
        load.semilogx(degree[keep], _real(values[keep]), label=label)
    load.set_xlabel("degree $l$")
    load.set_title("load Love numbers")
    load.grid(True, linestyle=":", alpha=0.6)
    load.legend()

    tide = love.tidal()
    keep = degree >= 2
    for values, label in (
        (tide["k"], "$k^T$"),
        (tide["h"], "$h^T$"),
        (tide["l"], "$l^T$"),
    ):
        if np.all(np.isnan(values[keep])):
            continue
        tidal.semilogx(degree[keep], _real(values[keep]), label=label)
    tidal.set_xlabel("degree $l$")
    tidal.set_title("tidal Love numbers")
    tidal.grid(True, linestyle=":", alpha=0.6)
    tidal.legend()
    return fig, axes


def plot_degree_solution(
    solution: DegreeSolution,
    /,
    *,
    ax: Optional[Axes] = None,
    n_points: int = 600,
    normalise: bool = True,
) -> Tuple[Figure, Axes]:
    """U, V and phi of one degree against radius, radius upward, with the
    model's boundaries drawn as faint lines.

    With `normalise` each component is divided by its surface value where
    that is not zero, so that the three share one axis. A complex
    solution is drawn by its real parts. The radius is in the mesh's
    units.

    Args:
        solution: A `DegreeSolution` from `solve_degree`.
        ax: A matplotlib axes to draw on; a new figure otherwise.
        n_points: The number of radii the solution is evaluated at.
        normalise: Whether to scale each component by its surface value.

    Returns:
        The figure and the axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7), layout="constrained")
    else:
        fig = ax.figure
    mesh = solution.mesh
    radii = np.linspace(mesh.left[0], mesh.right[-1], n_points)
    components = solution.evaluate(radii)
    surface = solution.surface
    for values, top, label in zip(components, surface, ("$U$", "$V$", "$\\phi$")):
        scale = abs(top) if normalise and top != 0.0 else 1.0
        ax.plot(_real(values) / scale, radii, label=label)
    for boundary in mesh.skeleton.boundaries[1:-1]:
        ax.axhline(boundary, color="0.85", lw=0.8, zorder=0.5)
    ax.set_ylabel("radius")
    ax.set_xlabel("value" + (" / surface value" if normalise else ""))
    ax.set_title(f"degree {solution.l}, forcing {solution.forcing!r}")
    ax.set_ylim(mesh.left[0], mesh.right[-1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    return fig, ax
