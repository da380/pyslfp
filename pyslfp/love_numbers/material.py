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
below it must be the model's.  It also carries the volume measure of
its mesh, the weights through which a sensitivity kernel, a density per
unit volume sampled at the nodes, pairs with a perturbation of a radial
parameter (`volume_weights`, `volume_pairing`, `kernel_samples`).
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
        """The nodal array shape (elements, nodes per element)."""
        return self.A.shape

    @property
    def is_complex(self) -> bool:
        """True for moduli frozen at a frequency."""
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
    for i, field in fields_by_layer.items():
        if not getattr(field, "is_radial", False):
            label = repr(field.name) if field.name else "a modulus"
            raise ValueError(
                f"{label} on layer {i} depends on direction; the loading "
                "problem is spherically symmetric and reads radial fields "
                "only (a model's mapping is ignored) — average the model "
                "first with planetmodel.spherically_averaged"
            )
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

    def with_moduli(self, moduli: NodalModuli, /) -> Material:
        """The same material with the moduli replaced: density, gravity,
        the fluid flags and the mesh are shared, only the five moduli
        change.  The replacement must have the mesh's nodal shape."""
        if not isinstance(moduli, NodalModuli):
            raise TypeError(f"expected a NodalModuli, got {type(moduli).__name__}")
        if moduli.shape != self.moduli.shape:
            raise ValueError(
                f"moduli have nodal shape {moduli.shape}, "
                f"the mesh has {self.moduli.shape}"
            )
        other = object.__new__(Material)
        other.__dict__.update(self.__dict__)
        other.moduli = moduli
        return other

    @property
    def is_complex(self) -> bool:
        """Whether the moduli, hence the solutions, are complex."""
        return self.moduli.is_complex

    @property
    def scales(self) -> Scales:
        """The model's scales, carried to the Love numbers."""
        return self.model.scales

    # -- the volume measure --------------------------------------------------

    @property
    def volume_weights(self) -> np.ndarray:
        """The volume weight W of every node, shape (nspec, ngll):

            W = 4 pi r^2 w jac,

        with w the node's Gauss-Lobatto-Legendre weight on the reference
        element and jac the element's Jacobian, so that sum W f(r) is
        the quadrature of the volume integral 4 pi int f(r) r^2 dr of a
        function of radius.  A node shared by two elements carries a
        weight in each; a node at the centre has weight zero.  In the
        material's units of volume."""
        mesh = self.mesh
        return 4.0 * np.pi * mesh.r**2 * mesh.w[None, :] * mesh.jac[:, None]

    def volume_pairing(self, kernel: ArrayLike, direction: ArrayLike) -> float:
        """The change of an objective under a perturbation of a radial
        parameter, from the parameter's sensitivity kernel.

        A kernel with respect to a parameter of the Earth, such as
        `IceKernels.itau` of the viscoelastic adjoints, is a density per
        unit volume held as its samples K(r) at the nodes: with
        delta(r) the perturbation of the parameter,

            dJ = int K delta dV = 4 pi int K(r) delta(r) r^2 dr
               ~ sum W K delta,

        W the `volume_weights`.  Neither the r^2 nor the 4 pi is in the
        kernel: both belong to the volume measure.  For a kernel of a
        discrete adjoint and a perturbation given by its nodal values,
        the sum is the exact derivative of the discrete model; for a
        smooth perturbation it is the quadrature of the integral.

        Args:
            kernel: The kernel's samples, shape (nspec, ngll).
            direction: The perturbation's nodal values, of that shape or
                broadcastable to it (a constant gives the kernel's
                volume integral).

        Returns:
            sum W K delta, in the objective's units.

        Raises:
            ValueError: If the kernel is not of the mesh's nodal shape,
                or the direction does not broadcast to it.
        """
        weights = self.volume_weights
        kernel = np.asarray(kernel)
        if kernel.shape != weights.shape:
            raise ValueError(
                f"the kernel has shape {kernel.shape}, the mesh's nodes "
                f"{weights.shape}"
            )
        direction = np.broadcast_to(np.asarray(direction), weights.shape)
        return float(np.sum(weights * kernel * direction))

    def kernel_samples(self, gradient: ArrayLike) -> np.ndarray:
        """A kernel's samples from the gradient with respect to the nodal
        values of a parameter.

        The gradient entry of a node, the derivative of the objective
        with respect to the parameter's value there, is the kernel times
        the node's volume weight, G = W K (`volume_pairing`), and the
        kernel is G / W.  A node without volume, at the centre, has a
        zero gradient entry and no sample of its own: its entry is not
        read, and its sample is the value there of the polynomial
        through the element's other nodes.

        Args:
            gradient: The gradient entries, shape (nspec, ngll).

        Returns:
            The kernel's samples, a new array of that shape.

        Raises:
            ValueError: If the gradient is not of the mesh's nodal shape.
        """
        weights = self.volume_weights
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != weights.shape:
            raise ValueError(
                f"the gradient has shape {gradient.shape}, the mesh's nodes "
                f"{weights.shape}"
            )
        live = weights > 0.0
        out = np.where(live, gradient / np.where(live, weights, 1.0), 0.0)
        # the Lagrange basis of the element's live nodes, at the dead one
        xi = self.mesh.xi
        for e, i in zip(*np.nonzero(~live)):
            nodes = np.flatnonzero(live[e])
            basis = [
                np.prod([(xi[i] - xi[q]) / (xi[p] - xi[q]) for q in nodes if q != p])
                for p in nodes
            ]
            out[e, i] = np.dot(basis, out[e, nodes])
        return out

    def __repr__(self) -> str:
        kind = "complex" if self.is_complex else "real"
        at = "" if self.omega is None else f", omega={self.omega:g}"
        return (
            f"Material({self.mesh.nspec} elements, "
            f"{int(self.fluid.sum())} fluid, {kind} moduli{at})"
        )
