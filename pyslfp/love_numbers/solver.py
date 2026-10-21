"""Degree solutions and the computation of Love numbers.

A degree is solved for one forcing: a unit surface density through its
traction-like piece, its attraction piece, or both, a unit tangential
traction, a unit external potential psi = (r/a)^l, or the centrifugal
potential (r/a)^2 whatever the degree.  The surface values of U, V and
phi per unit forcing are the generalised Love numbers; `table` states
the conventions, the units and the file.  At degree 0 the solutions
also give the axial numbers: the surface response to the centrifugal
potential and the inertia moments sqrt(4 pi) int rho U r^3 dr of each
response, which the axial rotational feedback of the sea level equation
needs.  Everything here is in the material's units.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from planetmodel import RadialMesh
from planetmodel.mesh1d.gll import lagrange_basis
from planetmodel.units import Dimensions, Scales
from .assembly import DegreeSystem
from .material import Material, NodalModuli
from .table import NAMES, LoveNumbers

if TYPE_CHECKING:
    from planetmodel import Model

__all__ = [
    "FORCINGS",
    "DegreeSolution",
    "TractionNumbers",
    "fluid_tidal_number",
    "graded_mesh",
    "inertia_moment",
    "solve_degree",
    "love_numbers",
    "traction_numbers",
]

FORCINGS = (
    "load",
    "load_force",
    "load_potential",
    "load_tangential",
    "tide",
    "centrifugal",
    "traction",
    "traction_tangential",
)

#: The forcings defined without gravity: the unit tractions.
NONGRAVITATING_FORCINGS = ("traction", "traction_tangential")

#: Displacement per unit traction.
_PER_TRACTION_LENGTH = Dimensions(mass=-1, length=2, time=2)


@dataclass(frozen=True)
class DegreeSolution:
    """The radial solution of one degree for one forcing.

    `U`, `V` and `phi` are nodal arrays of shape (nspec, ngll) over the
    whole mesh, zero below the solved sub-mesh and where a component has
    no dof; `evaluate(radii)` interpolates them within their elements.
    Above degree zero the displacement within a fluid region is not
    determined by the quasi-static problem, and `U` and `V` are NaN
    through its elements, so that a plot breaks there.  `surface` is
    (U, V, phi) at the outer boundary.
    """

    l: int
    forcing: str
    mesh: RadialMesh = field(repr=False)
    U: np.ndarray = field(repr=False)
    V: np.ndarray = field(repr=False)
    phi: np.ndarray = field(repr=False)

    @property
    def surface(self) -> tuple[complex, complex, complex]:
        """(U, V, phi) at the outer boundary."""
        return (self.U[-1, -1].item(), self.V[-1, -1].item(), self.phi[-1, -1].item())

    def evaluate(self, radii: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(U, V, phi) at `radii`, each element's polynomial evaluated on
        its own interval; a radius on an element boundary takes the
        element above it.  U and V are NaN within a fluid region above
        degree zero, and at the top of a fluid region they are the values
        of the solid above it."""
        r = np.asarray(radii, dtype=float)
        flat = r.reshape(-1)
        mesh = self.mesh
        if flat.size and (flat.min() < mesh.left[0] or flat.max() > mesh.right[-1]):
            raise ValueError("radii must lie within the mesh")
        out = [np.zeros(flat.shape, dtype=a.dtype) for a in (self.U, self.V, self.phi)]
        es = np.array([mesh.element_of(x) for x in flat], dtype=int)
        for e in np.unique(es):
            m = es == e
            basis = lagrange_basis(mesh.r[e], flat[m])
            for o, a in zip(out, (self.U, self.V, self.phi)):
                o[m] = basis @ a[e]
        return tuple(o.reshape(r.shape) for o in out)


def graded_mesh(
    model: Model, lmax: int, *, ngll: int = 5, eps: float = 1e-8
) -> RadialMesh:
    """A radial mesh for every degree up to `lmax`, fine at the surface and
    widening with depth.

    The degree-l solution is negligible (below `eps` of its surface value,
    the level `DegreeSystem` truncates at) deeper than
    D_l = a (1 - eps^(1/(l+1))), and the uniform rule
    `RadialMesh(model, lmax=l)` resolves it with elements no wider than
    0.1 a / (l + 1).  At depth d below the surface the highest degree
    whose solution reaches it has (l + 1) = ln(eps) / ln(1 - d/a), so an
    element there need be no narrower than 0.1 a ln(1 - d/a) / ln(eps),
    about d / 184 for eps = 1e-8.  This mesh has that width, floored at
    the uniform rule's width for `lmax`, so that every degree sees at
    least the resolution the uniform rule would give it throughout its
    own sub-mesh, while the whole mesh has a few hundred elements rather
    than the ten (lmax + 1) of the uniform rule.  Every skeleton boundary
    is an element boundary; the last of the elements above a boundary is
    shortened, or split in two, so that none is wider than the rule at
    its top.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    boundaries = np.asarray(model.skeleton.boundaries, dtype=float)
    a = float(boundaries[-1])
    floor = 0.1 * a / (lmax + 1)
    log_eps = np.log(eps)

    def width(r: float) -> float:
        depth = a - r
        if depth <= 0.0:
            return floor
        return max(floor, 0.1 * a * np.log1p(-depth / a) / log_eps)

    edges = [a]
    for hi, lo in zip(boundaries[::-1][:-1], boundaries[::-1][1:]):
        r = float(hi)
        while True:
            w = width(r)
            if r - lo <= 2.0 * w:
                # the rest of the span in one or two equal elements, each
                # no wider than the rule at its top
                n = 1 if r - lo <= w else 2
                edges.extend(np.linspace(r, lo, n + 1)[1:].tolist())
                break
            r -= w
            edges.append(r)
    return RadialMesh(model, ngll=ngll, edges=np.array(edges[::-1]))


def _material(
    model_or_material: Model | Material,
    *,
    mesh: RadialMesh | None,
    ngll: int,
    lmax: int,
    eps: float,
) -> Material:
    if isinstance(model_or_material, Material):
        if mesh is not None:
            raise ValueError("a Material already fixes the mesh")
        return model_or_material
    if mesh is None:
        mesh = graded_mesh(model_or_material, lmax, ngll=ngll, eps=eps)
    return Material(mesh, model_or_material)


def solve_degree(
    model_or_material: Model | Material,
    l: int,
    *,
    forcing: str = "load",
    mesh: RadialMesh | None = None,
    ngll: int = 5,
    eps: float = 1e-8,
    gravitating: bool = True,
) -> DegreeSolution:
    """The degree-l solution of a model, or of a ready `Material`, for
    one forcing of `FORCINGS`: a unit surface density through both
    channels, its traction-like or attraction piece alone, a unit
    tangential traction, a unit external potential (r/a)^l, the
    centrifugal potential (r/a)^2, or a unit normal or tangential
    traction.  Given a model, the mesh is built with the `lmax=l` rule
    unless supplied, with `ngll` nodes per element; `eps` is the
    relative level below which the degree's solution is truncated with
    depth.  `gravitating=False` solves the non-gravitating problem,
    which takes a traction forcing, a solid model and l != 1."""
    if forcing not in FORCINGS:
        raise ValueError(f"forcing must be one of {FORCINGS}, got {forcing!r}")
    if not gravitating and forcing not in NONGRAVITATING_FORCINGS:
        raise ValueError(
            f"the non-gravitating problem takes a forcing of "
            f"{NONGRAVITATING_FORCINGS}, got {forcing!r}"
        )
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=l, eps=eps)
    system = DegreeSystem(material, l, eps=eps, gravitating=gravitating)
    if forcing == "tide":
        b = system.tide()
    elif forcing == "centrifugal":
        b = system.tide(power=2)
    elif forcing == "load_tangential":
        b = system.tangential()
    elif forcing == "traction":
        b = system.traction()
    elif forcing == "traction_tangential":
        b = system.traction_tangential()
    else:
        part = {"load": "both", "load_force": "force", "load_potential": "potential"}[
            forcing
        ]
        b = system.load(part=part)
    U, V, phi = system.expand(system.solve(b))
    return DegreeSolution(int(l), forcing, material.mesh, U, V, phi)


def fluid_tidal_number(
    model_or_material: Model | Material,
    l: int = 2,
    *,
    mesh: RadialMesh | None = None,
    ngll: int = 5,
    eps: float = 1e-8,
) -> float:
    """The degree-l tidal Love number of the model's hydrostatic fluid
    limit: every region fluid, the response to the unit tide found from
    the potential alone with its stratification term (the fluid
    boundary-value problem, not the limit of a viscoelastic k_t(s),
    which degrades numerically as s -> 0).

    This is the k_hyd of the rotational-stability literature: the
    equilibrium oblateness a body can sustain is k_hyd Omega^2 a^5 /
    (3 G), and the moments of inertia used in the rotational feedback
    must exceed it for the feedback to be stable (Mitrovica et al.
    2005; Mitrovica & Wahr 2011).  See
    `EarthModelParameters.with_hydrostatic_moments`.

    The mesh arguments are `solve_degree`'s.  The fluid limit is
    solved as the zero-rigidity solid: the shear
    moduli are set exactly to zero and the degree system falls back to
    its LU path, which handles the semi-definite elastic block.  Only
    the density profile then matters -- a homogeneous body gives
    Kelvin's 3 / (2 (l - 1)) to solver precision, compressible or not
    -- and the potential-only treatment of flagged fluid regions is
    unchanged.
    """
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=l, eps=eps)
    if material.is_complex:
        raise ValueError("the fluid limit takes a real material")
    if l < 2:
        raise ValueError("the fluid tidal number is defined for degree >= 2")
    kappa, _ = material.moduli.kappa_mu()
    kappa = np.asarray(kappa).real
    fluid = material.with_moduli(NodalModuli.isotropic(kappa, np.zeros_like(kappa)))
    return float(solve_degree(fluid, l, forcing="tide").surface[2].real)


def inertia_moment(material: Material, U: np.ndarray) -> complex:
    """sqrt(4 pi) int rho U r^3 dr over the mesh for a nodal radial
    displacement U of degree 0, which is int rho u . x dV, a quarter of
    the change in the trace of the inertia tensor that the displacement
    u = U Y_00 r_hat produces."""
    mesh = material.mesh
    weights = mesh.w[None, :] * mesh.jac[:, None]
    return np.sqrt(4.0 * np.pi) * np.sum(weights * material.rho * U * mesh.r**3)


def love_numbers(
    model_or_material: Model | Material,
    lmax: int,
    *,
    mesh: RadialMesh | None = None,
    ngll: int = 5,
    eps: float = 1e-8,
) -> LoveNumbers:
    """The Love numbers of a model, or of a ready `Material`, for every
    degree from 0 to `lmax`, with the axial numbers from degree 0.

    Each degree is assembled once and solved for the four forcings, and
    degree 0 for the centrifugal potential besides.  Given a model, the
    mesh is `graded_mesh(model, lmax, ngll=ngll, eps=eps)` unless
    supplied.  A model frozen at a frequency gives complex numbers.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=lmax, eps=eps)
    dtype = complex if material.is_complex else float
    out = {name: np.zeros(lmax + 1, dtype=dtype) for name in NAMES}
    axial = {}
    for l in range(lmax + 1):
        system = DegreeSystem(material, l, eps=eps)
        columns = [
            system.load(part="force"),
            system.load(part="potential"),
            system.tangential(),
            system.tide(),
        ]
        if l == 0:
            columns.append(system.tide(power=2))
        X = system.solve(np.column_stack(columns))
        for j, channel in enumerate(("u", "phi", "v", "t")):
            for letter, component in (("h", "U"), ("l", "V"), ("k", "phi")):
                i = system.surface_dof(component)
                if i >= 0:
                    out[f"{letter}_{channel}"][l] = X[i, j]
        if l == 0:
            axial["h_c"] = X[system.surface_dof("U"), 4]
            axial["k_c"] = X[system.surface_dof("phi"), 4]
            for name, j in (("m_u", 0), ("m_phi", 1), ("m_c", 4)):
                U, _, _ = system.expand(X[:, j])
                axial[name] = inertia_moment(material, U)
    return LoveNumbers(
        np.arange(lmax + 1),
        radius=material.radius,
        surface_gravity=material.surface_gravity,
        G=material.G,
        scales=material.scales,
        omega=material.omega,
        **out,
        **axial,
    )


@dataclass(frozen=True)
class TractionNumbers:
    """Surface response per unit surface traction, by degree, in the
    units of `scales`.

    `h_u` and `l_u` are U(a) and V(a) per unit normal traction pressing
    on the surface, `h_v` and `l_v` the same per unit tangential
    traction: the traction channels of the generalised Love numbers per
    unit of traction rather than of surface density, so on the
    gravitating problem they are those channels divided by g.
    `gravitating` records which problem was solved; on the
    non-gravitating one degree 1 is excluded and its entries are NaN.
    The problem's self-adjointness gives h_v = l(l + 1) l_u, which
    `reciprocity_residual` measures.
    """

    degree: np.ndarray
    h_u: np.ndarray
    l_u: np.ndarray
    h_v: np.ndarray
    l_v: np.ndarray
    radius: float
    gravitating: bool
    scales: Scales = Scales.SI

    def reciprocity_residual(self) -> np.ndarray:
        """``abs(h_v - l(l+1) l_u)`` by degree, relative to the larger of
        the two (zero where both vanish, and on an excluded degree)."""
        k2 = self.degree * (self.degree + 1.0)
        diff = np.abs(self.h_v - k2 * self.l_u)
        scale = np.maximum(np.abs(self.h_v), np.abs(k2 * self.l_u))
        with np.errstate(invalid="ignore"):
            live = scale > 0.0
        return np.where(live, diff / np.where(live, scale, 1.0), 0.0)

    def converted(self, scales: Scales, /) -> TractionNumbers:
        """The same numbers under other scales."""
        if scales == self.scales:
            return self
        ratio = self.scales.factor(_PER_TRACTION_LENGTH) / scales.factor(
            _PER_TRACTION_LENGTH
        )
        length = self.scales.factor(Dimensions(length=1)) / scales.factor(
            Dimensions(length=1)
        )
        return replace(
            self,
            h_u=self.h_u * ratio,
            l_u=self.l_u * ratio,
            h_v=self.h_v * ratio,
            l_v=self.l_v * ratio,
            radius=self.radius * length,
            scales=scales,
        )


def traction_numbers(
    model_or_material: Model | Material,
    lmax: int,
    *,
    gravitating: bool = False,
    mesh: RadialMesh | None = None,
    ngll: int = 5,
    eps: float = 1e-8,
) -> TractionNumbers:
    """The surface response per unit normal and tangential traction for
    every degree from 0 to `lmax`, on the non-gravitating problem by
    default, or on the full problem with `gravitating=True` -- the same
    forcing on both, so the pair isolates the effect of gravity.  The
    non-gravitating problem takes a solid model, and its degree 1 is
    NaN (a surface traction there has a net force).  The mesh arguments
    are `love_numbers`'s."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    material = _material(model_or_material, mesh=mesh, ngll=ngll, lmax=lmax, eps=eps)
    dtype = complex if material.is_complex else float
    out = {name: np.full(lmax + 1, np.nan, dtype=dtype)
           for name in ("h_u", "l_u", "h_v", "l_v")}  # fmt: skip
    for l in range(lmax + 1):
        if l == 1 and not gravitating:
            continue
        system = DegreeSystem(material, l, eps=eps, gravitating=gravitating)
        X = system.solve(
            np.column_stack([system.traction(), system.traction_tangential()])
        )
        iU, iV = system.surface_dof("U"), system.surface_dof("V")
        for j, channel in enumerate(("u", "v")):
            out[f"h_{channel}"][l] = X[iU, j]
            out[f"l_{channel}"][l] = X[iV, j] if iV >= 0 else 0.0
    return TractionNumbers(
        np.arange(lmax + 1),
        radius=material.radius,
        gravitating=bool(gravitating),
        scales=material.scales,
        **out,
    )
