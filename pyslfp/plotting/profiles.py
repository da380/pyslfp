"""
Radial profiles on a material's mesh, drawn with planetmodel's profile
plots: a nodal array over the mesh, a rheology's viscosity, and a vector of
a log-viscosity space.

A nodal array of shape (nspec, ngll), as the viscoelastic kernels and the
branch data of a `Rheology` are laid out, is a piecewise polynomial on the
mesh. Each layer of the model is drawn as one segment, with planetmodel's
joiners across the discontinuities and guides at the layer boundaries, so a
profile looks like the model's own fields drawn with
`planetmodel.plotting.radial_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from planetmodel.plotting import profile

from ..love_numbers import Material

__all__ = ["radial_fields", "plot_profile", "plot_viscosity", "plot_log_viscosity"]


@dataclass(frozen=True)
class _Scalar:
    rank: int = 0


class _LayerField:
    """A nodal array's piecewise polynomial over one layer of the mesh, in
    the shape planetmodel's profile plots take: a radial field of rank 0
    on an interval, callable at radii."""

    is_radial = True
    character = _Scalar()

    def __init__(self, material: Material, nodal: np.ndarray, layer: int) -> None:
        mesh = material.mesh
        elements = np.flatnonzero(mesh.layer == layer)
        self.interval = (float(mesh.r[elements].min()), float(mesh.r[elements].max()))
        self.name = f"layer {layer}"
        self._ppoly = mesh.to_ppoly(
            nodal, elements=(int(elements[0]), int(elements[-1]) + 1)
        )

    def __call__(self, r: Any) -> np.ndarray:
        return self._ppoly(np.asarray(r, dtype=float))

    def evaluate(self, r: Any, theta: Any, phi: Any, *, frame: str = "spherical"):
        return self(r)


def radial_fields(
    material: Material,
    nodal: np.ndarray,
    /,
    *,
    layers: Optional[Sequence[int]] = None,
) -> list:
    """
    A nodal array over the material's mesh as one radial field per layer of
    the model, in the form `planetmodel.plotting.profile` draws: None for a
    layer left out.

    Args:
        material: The material whose mesh the array is laid out on.
        nodal: An (nspec, ngll) array of values at the mesh's nodes.
        layers: The model's layer indices to draw; all by default.

    Raises:
        ValueError: If the array is not of the mesh's shape.
    """
    mesh = material.mesh
    nodal = np.asarray(nodal, dtype=float)
    if nodal.shape != mesh.r.shape:
        raise ValueError(
            f"expected an array of shape {mesh.r.shape}, got {nodal.shape}"
        )
    wanted = None if layers is None else {int(i) for i in layers}
    return [
        _LayerField(material, nodal, int(i))
        if wanted is None or int(i) in wanted
        else None
        for i in np.unique(mesh.layer)
    ]


def plot_profile(
    ax: Any,
    material: Material,
    nodal: np.ndarray,
    /,
    *,
    layers: Optional[Sequence[int]] = None,
    scale: float = 1.0,
    value_scale: float = 1.0,
    label: Optional[str] = None,
    **style: Any,
) -> list:
    """
    Draws a nodal array over the material's mesh against radius, radius
    upward, one segment per layer, with planetmodel's joiners across the
    discontinuities and guides at the layer boundaries.

    Args:
        ax: The matplotlib axes.
        material: The material whose mesh the array is laid out on.
        nodal: An (nspec, ngll) array, such as a viscosity kernel.
        layers: The model's layer indices to draw; all by default.
        scale: Multiplies the radii before drawing, in the material's
            units; 1e-3 of the length unit's metres gives kilometres.
        value_scale: Multiplies the values before drawing.
        label: The legend label of the profile.
        **style: Line style, as for `planetmodel.plotting.profile`.

    Returns:
        The lines drawn.
    """
    return profile(
        ax,
        radial_fields(material, nodal, layers=layers),
        scale=scale,
        value_scale=value_scale,
        label=label,
        **style,
    )


def plot_viscosity(
    ax: Any,
    rheology,
    /,
    *,
    branch: int = 0,
    scale: float = 1.0,
    label: Optional[str] = None,
    **style: Any,
) -> list:
    """
    Draws the viscosity of one branch of a rheology against radius, in
    pascal seconds on a logarithmic axis, over the layers in which the
    branch relaxes.

    The viscosity at a node is the branch's shear modulus over its inverse
    relaxation time, in the material's units, times the unit of viscosity
    of the material's frame, mass over length and time.

    Args:
        ax: The matplotlib axes.
        rheology: The `Rheology`.
        branch: Which branch.
        scale: Multiplies the radii before drawing.
        label: The legend label.
        **style: Line style.

    Returns:
        The lines drawn.
    """
    material = rheology.material
    mesh = material.mesh
    mu, itau = rheology.mu[branch], rheology.itau[branch]
    live = (mu > 0.0) & (itau > 0.0)
    scales = material.scales
    unit = scales.mass / (scales.length * scales.time)
    eta = np.where(live, mu / np.where(live, itau, 1.0), np.nan) * unit
    layers = [int(i) for i in np.unique(mesh.layer) if live[mesh.layer == i].all()]
    lines = plot_profile(
        ax,
        material,
        np.where(live, eta, 0.0),
        layers=layers,
        scale=scale,
        label=label,
        **style,
    )
    ax.set_xscale("log")
    return lines


def plot_log_viscosity(
    ax: Any,
    space,
    chi,
    /,
    *,
    branch: int = 0,
    scale: float = 1.0,
    label: Optional[str] = None,
    **style: Any,
) -> list:
    """
    Draws a vector of a `LogViscositySpace`, the log viscosity
    log(eta / eta_0) of one branch, against radius over the branch's layers.

    Args:
        ax: The matplotlib axes.
        space: The `LogViscositySpace`.
        chi: A vector of it.
        branch: Which branch.
        scale: Multiplies the radii before drawing.
        label: The legend label.
        **style: Line style.

    Returns:
        The lines drawn.
    """
    nodal = space.nodal(chi)[branch]
    return plot_profile(
        ax,
        space.reference_rheology.material,
        nodal,
        layers=space.branch_layers(branch),
        scale=scale,
        label=label,
        **style,
    )
