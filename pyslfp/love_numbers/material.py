"""What the loading solver needs of a model on a radial mesh, as nodal arrays.

`NodalModuli` holds the transversely isotropic moduli A, C, F, L, N at
the nodes of a `RadialMesh`, per element, real or complex; a fluid node
has L = N = 0 and A = C = F = kappa.  `nodal_moduli` reads them from the
model through `planetmodel.moduli`, so the solver's model is anything
holding density and, on each layer, whatever `moduli` reads the five
from.  A viscoelastic body enters as a model frozen at a frequency
(`planetmodel.frozen`), whose moduli are complex; the dtype follows the
fields.

`Material` bundles, once, everything the degree systems assemble from:
density and its radial derivative, gravity, the fluid flag of every
element and the moduli, all in the model's units with the model's G.
It refuses a mesh that does not span the whole model and a model whose
surface is fluid, since the loaded surface must be solid and the mass
below it must be the model's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from planetmodel.fields import Field
from planetmodel.materials import is_fluid, kappa_mu_from_moduli, moduli
from planetmodel import RadialMesh
from planetmodel.units import Scales

if TYPE_CHECKING:
    from planetmodel import Model

__all__ = ["NodalModuli", "nodal_moduli", "Material"]

MODULI = ("A", "C", "F", "L", "N")


@dataclass(frozen=True)
class NodalModuli:
    """The moduli A, C, F, L, N at the nodes of a radial mesh.

    Each is an array of shape (nspec, ngll), every element carrying its
    own layer's values so that a shared node holds both one-sided
    values.  Real or complex; the five share one dtype.
    """

    A: np.ndarray
    C: np.ndarray
    F: np.ndarray
    L: np.ndarray
    N: np.ndarray

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, n)) for n in MODULI]
        if any(a.dtype.kind not in "fc" for a in arrays):
            raise TypeError("moduli must be real or complex floating arrays")
        dtype = np.result_type(*arrays)
        shape = arrays[0].shape
        for n, a in zip(MODULI, arrays):
            if a.shape != shape:
                raise ValueError(f"{n} has shape {a.shape}, A has {shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{n} is not finite everywhere")
            a = a.astype(dtype, copy=True)
            a.setflags(write=False)
            object.__setattr__(self, n, a)

    @classmethod
    def isotropic(cls, kappa: ArrayLike, mu: ArrayLike) -> NodalModuli:
        """A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3, L = N = mu."""
        kappa = np.asarray(kappa)
        mu = np.asarray(mu)
        A = kappa + 4.0 * mu / 3.0
        return cls(A, A, kappa - 2.0 * mu / 3.0, mu, mu)

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    @property
    def is_complex(self) -> bool:
        return self.A.dtype.kind == "c"

    def kappa_mu(self) -> tuple[np.ndarray, np.ndarray]:
        """The Voigt-averaged isotropic (kappa, mu) of the five."""
        return kappa_mu_from_moduli(self.A, self.C, self.F, self.L, self.N)

    def voigt(self) -> NodalModuli:
        """The isotropic collapse: the Voigt average recomposed."""
        return NodalModuli.isotropic(*self.kappa_mu())


def _check_mesh(mesh: RadialMesh, model: Model) -> None:
    if not isinstance(mesh, RadialMesh):
        raise TypeError(f"expected a RadialMesh, got {type(mesh).__name__}")
    b = model.skeleton.boundaries
    if mesh.skeleton != model.skeleton:
        raise ValueError(
            "the mesh was built over a different skeleton from " "the model's"
        )
    if mesh.left[0] != b[0] or mesh.right[-1] != b[-1]:
        raise ValueError(
            f"the mesh spans [{mesh.left[0]:g}, {mesh.right[-1]:g}] but the "
            f"model [{b[0]:g}, {b[-1]:g}]; truncate the model, not the mesh"
        )


def _nodal(mesh: RadialMesh, fields_by_layer: Mapping[int, Field]) -> np.ndarray:
    """A field per layer index, evaluated on each element's own nodes;
    complex where any field is."""
    pieces = {
        i: field.evaluate(mesh.r[mesh.layer == i], 0.0, 0.0)
        for i, field in fields_by_layer.items()
    }
    dtype = np.result_type(*(v.dtype for v in pieces.values()))
    out = np.empty((mesh.nspec, mesh.ngll), dtype=dtype)
    for i, values in pieces.items():
        out[mesh.layer == i] = values
    return out


def nodal_moduli(mesh: RadialMesh, model: Model) -> NodalModuli:
    """The model's moduli A, C, F, L, N at the nodes of `mesh`.

    Each layer's moduli are what `planetmodel.moduli` reads from the
    fields it holds, so an isotropic layer gives A = C, F = A - 2L,
    L = N, and a fluid layer L = N = 0.
    """
    _check_mesh(mesh, model)
    per_layer = {int(i): moduli(model.layer(int(i))) for i in np.unique(mesh.layer)}
    return NodalModuli(
        *(_nodal(mesh, {i: m[n] for i, m in per_layer.items()}) for n in MODULI)
    )


class Material:
    """A model on a radial mesh, as the arrays the degree systems use.

    Attributes, each of shape (nspec, ngll) unless said otherwise: `rho`,
    `drho` (its radial derivative), `g` (gravity from the whole model),
    `moduli` (a `NodalModuli`, complex for a frozen viscoelastic model),
    `fluid` (per element, shape (nspec,)), the numbers `G`, `radius`
    (the outer boundary) and `surface_gravity`, and `omega`, the
    frequency a frozen model records, else None.  Every quantity is in
    the model's units.
    """

    def __init__(self, mesh: RadialMesh, model: Model) -> None:
        _check_mesh(mesh, model)
        self.mesh = mesh
        self.model = model
        self.rho = mesh.nodal(model, "rho")
        self.drho = mesh.nodal(model, "rho", nu=1)
        self.g = mesh.nodal_gravity(model)
        fluid_layer = {
            int(i): is_fluid(model.layer(int(i))) for i in np.unique(mesh.layer)
        }
        self.fluid = np.array([fluid_layer[int(i)] for i in mesh.layer], dtype=bool)
        if self.fluid[-1]:
            raise ValueError(
                "the surface of the model is fluid, so it cannot carry a load; "
                "truncate the model at its solid surface (PREM(ocean=False), "
                "or model.truncated(radius))"
            )
        self.moduli = nodal_moduli(mesh, model)
        for a in (self.rho, self.drho, self.g, self.fluid):
            a.setflags(write=False)
        self.G = float(model.G)
        self.radius = float(mesh.right[-1])
        self.surface_gravity = float(self.g[-1, -1])
        self.omega = (
            float(model.constant("omega")) if "omega" in model.constants else None
        )

    @property
    def is_complex(self) -> bool:
        """Whether the moduli, hence the solutions, are complex."""
        return self.moduli.is_complex

    @property
    def scales(self) -> Scales:
        """The model's scales, carried to the Love numbers."""
        return self.model.scales

    def __repr__(self) -> str:
        kind = "complex" if self.is_complex else "real"
        at = "" if self.omega is None else f", omega={self.omega:g}"
        return (
            f"Material({self.mesh.nspec} elements, "
            f"{int(self.fluid.sum())} fluid, {kind} moduli{at})"
        )
