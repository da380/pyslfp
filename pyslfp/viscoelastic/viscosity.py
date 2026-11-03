"""
Log viscosity as a function of radius: its Hilbert space, its values at the
nodes of the radial mesh, and the rheology it names.

The viscosity of branch k of a rheology is a function of radius on the
layers of the mesh in which the branch relaxes, continuous within a layer and
free to jump at the material interfaces between layers. Its parameter is

    chi_k = log(eta_k / eta_0),

with eta_0 the unit of viscosity of the material's frame, mass over length
and time, so that in the frame's own numbers chi_k = log eta_k; the shear
modulus of the branch is fixed, so the inverse relaxation time is
tau_k^{-1} = mu_k exp(-chi_k) / eta_0.

`LogViscositySpace` is the Hilbert space of these functions: for each branch
a pygeoinf layered radial space (`pygeoinf.interval.radial.Layered`), one
Sobolev space of radial profiles per relaxing layer under the volume measure
r^2 dr, and the direct sum over the branches. A vector is a tuple over the
branches of tuples over the layers of layer fields. The space evaluates a
vector at the mesh's nodes, which is how a rheology is built from it, and
transposes that evaluation, which is how a nodal derivative, such as the
kernel for the inverse relaxation times, becomes a functional on the space
and, through the metric, its Riesz representer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

from pygeoinf import LinearFunctional
from pygeoinf.algebra.direct_sum import CoordinateDirectSum
from pygeoinf.interval.radial import Layered, Lebesgue, Sobolev

from .rheology import Rheology

__all__ = ["LogViscositySpace"]


class LogViscositySpace(CoordinateDirectSum):
    """
    The space of log viscosities of a rheology's branches, as functions of
    radius: the direct sum over the branches of a layered radial space, one
    Sobolev space per layer in which the branch relaxes.

    A layer is a layer of the material's model, and a branch relaxes in it
    when every node of the layer carries the branch with a finite viscosity.
    The layers of a branch are reached by `branch_layers`, their space by
    `branch_space`; the direct sum's labels are `branch_k`, and each branch's
    layers are labelled `layer_i` by the model's layer index.
    """

    def __new__(cls, *args, **kwargs) -> LogViscositySpace:
        # The direct sum's __new__ takes the summands; here they are built
        # from the rheology, so the arguments are left to __init__.
        return object.__new__(cls)

    def __init__(
        self,
        rheology: Rheology,
        modes: int | Mapping[int, int],
        /,
        *,
        order: float = 1.0,
        length_scale: Optional[float] = None,
        padding: Optional[float | tuple[float, float]] = None,
        boundary="robin",
        ngll: int = 5,
        element_length: Optional[float] = None,
    ) -> None:
        """
        Args:
            rheology: The reference rheology, which fixes the material, the
                shear modulus of each branch and the layers in which it
                relaxes.
            modes: The eigenfunctions kept in each layer's space, one number
                for every layer or a mapping from the model's layer index to
                its number.
            order: The Sobolev order of every layer's space; above one half
                the functions are continuous within a layer and evaluation at
                the nodes is bounded. Zero gives the Lebesgue spaces.
            length_scale: The length scale of every layer's space, in the
                material's units; by default each layer's own thickness.
            padding: The reach of each layer's mesh past its ends, as for
                `pygeoinf.interval.radial.Radial`; by default two length
                scales.
            boundary: The condition at the ends of each layer's mesh, as for
                `pygeoinf.interval.radial.Radial`.
            ngll: Gauss-Lobatto-Legendre nodes per element of the layers'
                meshes.
            element_length: The longest an element of those meshes may be.

        Raises:
            ValueError: If a branch relaxes in part of a layer only, or
                nowhere.
        """
        mesh = rheology.material.mesh
        self._rheology = rheology
        self._mesh = mesh
        self._records: list[list[tuple[int, np.ndarray, np.ndarray]]] = []
        branches = []
        for k in range(rheology.num_branches):
            active = (rheology.mu[k] > 0.0) & (rheology.itau[k] > 0.0)
            layers = []
            for i in np.unique(mesh.layer):
                flags = active[mesh.layer == i]
                if flags.all():
                    layers.append(int(i))
                elif flags.any():
                    raise ValueError(
                        f"branch {k} relaxes in part of layer {int(i)} only"
                    )
            if not layers:
                raise ValueError(f"branch {k} relaxes nowhere")
            spaces, records = [], []
            for i in layers:
                elements = np.flatnonzero(mesh.layer == i)
                radii = mesh.r[elements]
                lower, upper = float(radii.min()), float(radii.max())
                n = int(modes[i]) if isinstance(modes, Mapping) else int(modes)
                scale = (
                    float(length_scale) if length_scale is not None else upper - lower
                )
                options = dict(
                    radius=upper,
                    inner_radius=lower,
                    padding=padding,
                    boundary=boundary,
                    ngll=ngll,
                    element_length=element_length,
                )
                if order == 0.0:
                    space = Lebesgue(n, length_scale=scale, **options)
                else:
                    space = Sobolev(n, order, scale, **options)
                spaces.append(space)
                records.append((i, elements, space.basis_matrix(radii.ravel())))
            branches.append(
                Layered.from_layers(spaces, labels=[f"layer_{i}" for i in layers])
            )
            self._records.append(records)
        super().__init__(branches, labels=[f"branch_{k}" for k in range(len(branches))])

    # -- structure ---------------------------------------------------------

    @property
    def reference_rheology(self) -> Rheology:
        """The rheology the space was built from."""
        return self._rheology

    @property
    def mesh(self):
        """The radial mesh of the material."""
        return self._mesh

    @property
    def num_branches(self) -> int:
        """The number of branches, the summands of the sum."""
        return len(self.subspaces)

    @property
    def viscosity_unit(self) -> float:
        """eta_0, the unit of viscosity of the material's frame in SI: a
        viscosity is eta_0 exp(chi) pascal seconds."""
        scales = self._rheology.material.scales
        return scales.mass / (scales.length * scales.time)

    def branch_space(self, k: int, /) -> Layered:
        """The layered radial space of branch k."""
        return self.subspaces[k]

    def branch_layers(self, k: int, /) -> tuple[int, ...]:
        """The model's layer indices in which branch k relaxes, in order."""
        return tuple(i for i, _, _ in self._records[k])

    # -- the mesh ----------------------------------------------------------

    def nodal(self, chi: tuple, /) -> list[np.ndarray]:
        """
        A vector's values at the mesh's nodes: one (nspec, ngll) array per
        branch, zero outside the branch's layers. A node shared by two
        elements of one layer takes one value; a node on an interface takes
        each layer's own in that layer's element.
        """
        out = []
        for k, records in enumerate(self._records):
            values = np.zeros(self._mesh.r.shape)
            branch = self.subspaces[k]
            for (_, elements, basis), layer, field in zip(
                records, branch.layers, chi[k]
            ):
                values[elements] = (basis @ layer.to_components(field)).reshape(
                    elements.size, -1
                )
            out.append(values)
        return out

    def derivative_components(self, nodal: Sequence[np.ndarray], /) -> np.ndarray:
        """
        The transpose of `nodal`: the components of the functional
        chi -> sum_k sum_nodes nodal[k] * chi_k(r_node), the sum over the
        (nspec, ngll) entries, so that a shared node counts in each element
        it belongs to.

        Args:
            nodal: One (nspec, ngll) array per branch; entries outside the
                branch's layers are ignored.
        """
        parts = []
        for k, records in enumerate(self._records):
            values = np.asarray(nodal[k], dtype=float)
            if values.shape != self._mesh.r.shape:
                raise ValueError(
                    f"branch {k}: expected an array of shape {self._mesh.r.shape}, "
                    f"got {values.shape}"
                )
            for _, elements, basis in records:
                parts.append(basis.T @ values[elements].ravel())
        return np.concatenate(parts)

    def derivative(self, nodal: Sequence[np.ndarray], /) -> LinearFunctional:
        """The functional of `derivative_components`, on this space."""
        return LinearFunctional.from_derivative_components(
            self, self.derivative_components(nodal)
        )

    def gradient(self, nodal: Sequence[np.ndarray], /) -> tuple:
        """The Riesz representer of `derivative`, a vector of this space."""
        return self.representer(self.derivative_components(nodal))

    def from_itau_kernels(
        self, kernels: Sequence[np.ndarray], rheology: Rheology, /
    ) -> LinearFunctional:
        """
        The derivative in log viscosity of an objective whose kernels in the
        inverse relaxation times are given: d tau^{-1} = -tau^{-1} d chi, so
        the nodal derivative in chi_k is -tau_k^{-1} dJ / d tau_k^{-1}.

        Args:
            kernels: The nodal kernels dJ / d tau_k^{-1}, one per branch, as
                `IceKernels.itau` holds them.
            rheology: The rheology the kernels were computed at.
        """
        if len(kernels) != self.num_branches:
            raise ValueError(
                f"expected {self.num_branches} kernels, got {len(kernels)}"
            )
        nodal = [
            -itau * np.asarray(k, dtype=float)
            for itau, k in zip(rheology.itau, kernels)
        ]
        return self.derivative(nodal)

    # -- rheologies --------------------------------------------------------

    def rheology(self, chi: tuple, /) -> Rheology:
        """
        The rheology a vector names: the reference rheology's branches, each
        with its shear modulus and, in the layers in which it relaxes, the
        inverse relaxation time mu_k exp(-chi_k) in the material's units.
        """
        reference = self._rheology
        nodal = self.nodal(chi)
        branches = []
        for k, records in enumerate(self._records):
            itau = np.array(reference.itau[k])
            for _, elements, _ in records:
                itau[elements] = reference.mu[k][elements] * np.exp(-nodal[k][elements])
            branches.append((reference.mu[k], itau))
        return Rheology(reference.material, branches)

    def log_viscosity(self, rheology: Optional[Rheology] = None, /) -> tuple:
        """
        The vector of this space naming a rheology's log viscosity, by
        default the reference's: in each layer, the field in the span of the
        layer's modes that best fits log(mu_k / tau_k^{-1}) over the layer,
        read off the mesh's nodal values through their spectral-element
        interpolation.

        Raises:
            ValueError: If the rheology is not on the reference's material
                with its branches, or does not relax where the reference
                does.
        """
        if rheology is None:
            rheology = self._rheology
        if rheology.num_branches != self.num_branches:
            raise ValueError("the rheology must have the reference's branches")
        if rheology.material.mesh is not self._mesh:
            raise ValueError("the rheology must be on the reference's mesh")
        out = []
        for k, records in enumerate(self._records):
            mu, itau = rheology.mu[k], rheology.itau[k]
            fields = []
            for (_, elements, _), layer in zip(records, self.subspaces[k].layers):
                if not np.all((mu[elements] > 0.0) & (itau[elements] > 0.0)):
                    raise ValueError(
                        f"branch {k} does not relax throughout the reference's layers"
                    )
                log_eta = np.zeros(self._mesh.r.shape)
                log_eta[elements] = np.log(mu[elements] / itau[elements])
                profile = self._mesh.to_ppoly(
                    log_eta, elements=(int(elements[0]), int(elements[-1]) + 1)
                )
                fields.append(layer.project_function(lambda r, p=profile: float(p(r))))
            out.append(tuple(fields))
        return tuple(out)

    def __repr__(self) -> str:
        layers = ", ".join(str(self.branch_layers(k)) for k in range(self.num_branches))
        return f"LogViscositySpace({self.num_branches} branch(es), layers {layers}; dim={self.dim})"
