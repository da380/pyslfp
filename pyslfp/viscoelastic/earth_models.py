"""Viscoelastic earth models: a radial model with a Maxwell mantle, and
the solver objects built from it.

A viscoelastic earth model here is a planetmodel model whose mantle
layers hold a `viscosity` field, planetmodel's convention for a Maxwell
body (tau = viscosity / mu). One model then serves every calculation:
`Rheology.from_model` reads the viscosity for the time-domain solver,
`EarthModel.from_planet_model` takes the elastic Love numbers and
ignores it, and planetmodel's `frozen` gives the frequency-domain Love
numbers of the same body. The viscosity is a field of the model, so it
is given in the model's units, Pa s for a model straight from
planetmodel, and converts with the model into the library's frame.

Radii, not depths, place the structure: they are what planetmodel works
in, and they fall exactly on a model's own interfaces, ocean or not. A
boundary that is not already an interface is inserted, so the mesh
honours it.

- `maxwell_mantle(model, viscosity, lithosphere=..., boundaries=...)` —
  any model with a piecewise-constant Maxwell mantle under an elastic
  lithosphere.
- `prem_maxwell()` — PREM without its ocean, isotropic and without its
  quality factors, with an elastic lid and a two-layer mantle split at
  the 670 km discontinuity.
- `viscoelastic_earth(model, lmax)` — the `EarthModel` and `Rheology` of
  a model, on one mesh, in the library's frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from planetmodel import PREM, constant_field, is_fluid

from ..core import EarthModel, EarthModelParameters
from ..love_numbers import Material, graded_mesh
from .rheology import Rheology

__all__ = ["maxwell_mantle", "prem_maxwell", "viscoelastic_earth"]

# PREM's upper and lower mantle viscosities in the commonly used two-layer
# reference structure, Pa s.
_UPPER_MANTLE_VISCOSITY = 5.0e20
_LOWER_MANTLE_VISCOSITY = 2.0e21


def maxwell_mantle(
    model,
    viscosity: Union[float, Sequence[float]],
    /,
    *,
    lithosphere: float,
    boundaries: Sequence[float] = (),
):
    """The model with a Maxwell mantle under an elastic lithosphere.

    The mantle is the solid region between the outermost fluid layer, the
    core (the centre for a model without one), and the radius
    `lithosphere`. Its viscosity is piecewise constant: one value, or one
    value per interval between the radii `boundaries`, ordered upward from
    the core. Above `lithosphere` the model stays elastic; `lithosphere`
    at the outer radius leaves no elastic lithosphere. Any viscosity the
    model holds is replaced. Radii and viscosities are in the model's
    units, SI for a model straight from planetmodel.

    Args:
        model: A planetmodel model of a solid body, its layers holding
            elastic moduli.
        viscosity: The viscosity, or the viscosities upward from the core.
        lithosphere: The radius of the lithosphere's base.
        boundaries: The radii between the mantle's viscosity intervals,
            increasing.

    Returns:
        The model, of the same type, with `viscosity` on every mantle layer
        and interfaces at `lithosphere` and `boundaries`.

    Raises:
        ValueError: If the radii are out of order or out of range, the
            number of viscosities does not match the intervals, or a
            viscosity is not positive.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    viscosities = np.atleast_1d(np.asarray(viscosity, dtype=float))
    if viscosities.size != boundaries.size + 1:
        raise ValueError(
            f"{boundaries.size} boundaries make {boundaries.size + 1} intervals, "
            f"but {viscosities.size} viscosities were given"
        )
    if np.any(viscosities <= 0.0):
        raise ValueError("viscosities must be positive")

    outer = model.layers[-1].interval[1]
    base = max(
        (layer.interval[1] for layer in model.layers if is_fluid(layer)),
        default=0.0,
    )
    radii = np.concatenate([[base], boundaries, [lithosphere]])
    if np.any(np.diff(radii) <= 0.0) or lithosphere > outer:
        raise ValueError(
            f"the radii must increase from the mantle's base, {base}, through "
            f"the boundaries to the lithosphere's base, at most the outer "
            f"radius {outer}; got boundaries {boundaries.tolist()} and "
            f"lithosphere {lithosphere}"
        )

    interfaces = model.skeleton.boundaries
    rtol = model.geometry.rtol
    new = [
        r
        for r in radii[1:]
        if not np.any(np.isclose(r, interfaces, rtol=rtol, atol=0.0))
    ]
    if new:
        model = model.refined(new)
    model = model.without_field("viscosity")

    for layer in model.layers:
        lo, hi = layer.interval
        middle = 0.5 * (lo + hi)
        if middle < base or middle > lithosphere or is_fluid(layer):
            continue
        eta = viscosities[np.searchsorted(boundaries, middle)]
        model = model.with_field(
            layer.index,
            "viscosity",
            constant_field(layer.interval, eta, name="viscosity"),
        )
    return model


def prem_maxwell(
    *,
    lithosphere: Optional[float] = None,
    upper_mantle: float = _UPPER_MANTLE_VISCOSITY,
    lower_mantle: float = _LOWER_MANTLE_VISCOSITY,
):
    """PREM with an elastic lid and a two-layer Maxwell mantle, in SI.

    PREM without its ocean, collapsed to its Voigt isotropic average, since
    the viscoelastic solver relaxes an isotropic shear modulus, and without
    its quality factors: their constant-Q dispersion about PREM's 1 s
    reference would otherwise reach the long periods of the frozen Love
    numbers. The mantle is split at PREM's 670 km discontinuity, the radius
    5701 km, into an upper mantle and transition zone above and a lower
    mantle below.

    Args:
        lithosphere: The radius of the elastic lithosphere's base, m. The
            base of PREM's lid, 6291 km, if None.
        upper_mantle: The viscosity above the 670 km discontinuity, Pa s.
        lower_mantle: The viscosity below it, Pa s.

    Returns:
        The model; convert it into the library's frame with
        `converted(scales)`, or pass it to `viscoelastic_earth`.
    """
    prem = PREM(ocean=False).isotropic()
    prem = prem.without_field("qmu").without_field("qkappa")
    transition = prem.layer("transition_zone_lower").interval[0]
    if lithosphere is None:
        lithosphere = prem.layer("lid").interval[0]
    return maxwell_mantle(
        prem,
        (lower_mantle, upper_mantle),
        boundaries=(transition,),
        lithosphere=lithosphere,
    )


def viscoelastic_earth(
    model,
    lmax: int,
    /,
    *,
    parameters: Optional[EarthModelParameters] = None,
    ngll: int = 5,
) -> tuple[EarthModel, Rheology]:
    """The Earth model and the rheology of a viscoelastic model, on one mesh.

    The model is converted into the frame of `parameters`, meshed for
    `lmax`, and read twice on that mesh: for its elastic Love numbers, in
    the `EarthModel`, and for its viscosity, in the `Rheology`. The mesh
    and the converted model are `rheology.material.mesh` and
    `rheology.material.model`.

    Args:
        model: A planetmodel model with a viscosity field, such as
            `prem_maxwell()`, in any units.
        lmax: The truncation degree.
        parameters: The parameters whose scales set the frame; the
            library's defaults if None.
        ngll: The nodes per element of the mesh.

    Returns:
        The Earth model and the rheology.
    """
    scales = (
        EarthModelParameters.from_defaults() if parameters is None else parameters
    ).scales
    converted = model.converted(scales)
    mesh = graded_mesh(converted, lmax, ngll=ngll)
    rheology = Rheology.from_model(Material(mesh, converted))
    earth_model = EarthModel.from_planet_model(
        converted, lmax, parameters=parameters, mesh=mesh
    )
    return earth_model, rheology
