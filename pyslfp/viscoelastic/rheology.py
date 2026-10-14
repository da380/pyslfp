"""Generalised Maxwell (Prony) branch data on the nodes of a radial mesh.

A `Rheology` pairs an elastic `Material` with relaxing shear branches:
per branch k, nodal arrays mu_k and itau_k = 1 / tau_k of the mesh's
shape (nspec, ngll).  The material's own moduli are the UNRELAXED ones,
so the long-term shear modulus is mu_inf = mu_U - sum_k mu_k, with
mu_U the Voigt-averaged shear modulus of the material; the stress is

    T = kappa tr(eps) I + 2 mu_inf d + sum_k 2 mu_k (d - m_k),
    dm_k/dt = (d - m_k) / tau_k.

The classical Maxwell body is one branch with mu_1 = mu_U (so
mu_inf = 0) and tau = eta / mu_U, the convention of planetmodel's
`viscosity` field, so a model frozen at a frequency and the time
stepping describe the same body.  A node with mu_k = 0 is inactive in
branch k; itau_k = 0 means an infinite relaxation time, under which
the branch's internal variable never moves.  Bulk relaxation is not
modelled.

Where any branch is active the material must be isotropic: the
degree-basis internal variables of Al-Attar & Tromp (2014) Appendix D
assume isotropic shear relaxation.  A transversely isotropic model is
refused with a pointer to `NodalModuli.voigt`; the general anisotropic
case (relaxation towards zero deviatoric stress, AdGIA's
`DeviatoricMaxwell`) is deliberately deferred.  Fluid regions carry no
branches.

Constructors, all in the material's units:

- `maxwell(material, viscosity)` — a Maxwell body with the viscosity
  given as a constant, a callable of radius, or a mapping from layer
  index to either (a two-layer mantle is a two-entry mapping); layers
  not named are elastic.
- `from_relaxation_time(material, tau)` — the relaxation-time
  parameterisation: tau(r) prescribed in the same three forms, so the
  implied viscosity eta = mu tau inherits its depth dependence from
  the shear modulus.
- `from_model(material)` — reads planetmodel's own convention: each
  layer holding a `viscosity` field is a Maxwell body with
  tau = viscosity / mu.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ..love_numbers.material import Material, NodalModuli

__all__ = ["Rheology"]

RadialValue = Union[float, Callable[[np.ndarray], ArrayLike]]


def _nodal(material: Material, value: RadialValue, elements: np.ndarray) -> np.ndarray:
    """A nodal array over the whole mesh from a constant or a callable of
    radius, non-zero on `elements` only."""
    mesh = material.mesh
    out = np.zeros((mesh.nspec, mesh.ngll))
    if callable(value):
        out[elements] = np.asarray(value(mesh.r[elements]), dtype=float)
    else:
        out[elements] = float(value)
    return out


def _per_layer(
    material: Material, values: Mapping[int, RadialValue] | RadialValue
) -> list[tuple[np.ndarray, RadialValue]]:
    """(elements, value) pairs: per layer index for a mapping, else every
    solid element under the one value."""
    mesh = material.mesh
    if isinstance(values, Mapping):
        pairs = []
        for i, value in values.items():
            elements = np.flatnonzero(mesh.layer == int(i))
            if elements.size == 0:
                raise ValueError(f"the mesh has no elements in layer {i}")
            if material.fluid[elements].any():
                raise ValueError(f"layer {i} is fluid and cannot relax")
            pairs.append((elements, value))
        return pairs
    return [(np.flatnonzero(~material.fluid), values)]


class Rheology:
    """A material and its relaxing shear branches, as nodal arrays.

    `material` is the elastic (unrelaxed) `Material`; `mu` and `itau`
    are lists of per-branch nodal arrays of shape (nspec, ngll), read
    only; `kappa` and `mu_unrelaxed` the Voigt pair of the material;
    `mu_long_term` = mu_unrelaxed - sum_k mu_k; `active` the per-node
    mask of any relaxing branch.
    """

    def __init__(
        self,
        material: Material,
        branches: list[tuple[ArrayLike, ArrayLike]],
        *,
        rtol: float = 1e-10,
    ) -> None:
        """`branches` is a list of (mu_k, itau_k) nodal pairs; itau is the
        INVERSE relaxation time, zero meaning infinite."""
        if not isinstance(material, Material):
            raise TypeError(f"expected a Material, got {type(material).__name__}")
        if material.is_complex:
            raise ValueError("the time-domain rheology takes a real material")
        self.material = material
        shape = material.moduli.shape
        self.mu: list[np.ndarray] = []
        self.itau: list[np.ndarray] = []
        for k, (mu_k, itau_k) in enumerate(branches):
            mu_k = np.asarray(mu_k, dtype=float)
            itau_k = np.asarray(itau_k, dtype=float)
            for name, a in (("mu", mu_k), ("itau", itau_k)):
                if a.shape != shape:
                    raise ValueError(
                        f"branch {k}: {name} has shape {a.shape}, "
                        f"the mesh has {shape}"
                    )
                if not np.all(np.isfinite(a)) or np.any(a < 0.0):
                    raise ValueError(f"branch {k}: {name} must be finite and >= 0")
            mu_k = mu_k.copy()
            itau_k = itau_k.copy()
            mu_k.setflags(write=False)
            itau_k.setflags(write=False)
            self.mu.append(mu_k)
            self.itau.append(itau_k)

        self.kappa, self.mu_unrelaxed = (
            np.asarray(a).real for a in material.moduli.kappa_mu()
        )
        total = sum(self.mu, start=np.zeros(shape))
        self.mu_long_term = self.mu_unrelaxed - total
        self.active = total > 0.0

        scale = np.max(self.mu_unrelaxed)
        if np.any(self.mu_long_term < -rtol * scale):
            raise ValueError(
                "the branch moduli exceed the material's shear modulus: "
                "mu_inf = mu - sum_k mu_k would be negative"
            )
        np.clip(self.mu_long_term, 0.0, None, out=self.mu_long_term)
        self.mu_long_term.setflags(write=False)

        if np.any(self.active[material.fluid]):
            raise ValueError("a fluid element cannot carry a relaxing branch")

        m = material.moduli
        iso = NodalModuli.isotropic(self.kappa, self.mu_unrelaxed)
        for name in ("A", "C", "F", "L", "N"):
            a, b = getattr(m, name), getattr(iso, name)
            bad = self.active & (np.abs(a - b) > rtol * scale)
            if np.any(bad):
                raise ValueError(
                    "the material is anisotropic where a branch relaxes; the "
                    "degree-basis internal variables assume isotropic shear "
                    "relaxation.  Collapse the model first (NodalModuli.voigt, "
                    "or planetmodel's Model.isotropic)."
                )

    # -- constructors -----------------------------------------------------

    @classmethod
    def maxwell(
        cls,
        material: Material,
        viscosity: Mapping[int, RadialValue] | RadialValue,
    ) -> Rheology:
        """A classical Maxwell body, tau = eta / mu with mu the Voigt
        shear modulus: `viscosity` is a constant, a callable of radius,
        or a mapping from layer index to either; with a mapping, layers
        not named stay elastic."""
        _, mu = (np.asarray(a).real for a in material.moduli.kappa_mu())
        mu_1 = np.zeros_like(mu)
        itau_1 = np.zeros_like(mu)
        for elements, value in _per_layer(material, viscosity):
            eta = _nodal(material, value, elements)[elements]
            if np.any(eta <= 0.0):
                raise ValueError("viscosity must be positive")
            mu_1[elements] = mu[elements]
            itau_1[elements] = mu[elements] / eta
        return cls(material, [(mu_1, itau_1)])

    @classmethod
    def from_relaxation_time(
        cls,
        material: Material,
        tau: Mapping[int, RadialValue] | RadialValue,
    ) -> Rheology:
        """A Maxwell body with the relaxation time prescribed directly,
        the implied viscosity eta = mu tau inheriting its depth
        dependence from the shear modulus; `tau` takes the same three
        forms as `maxwell`'s viscosity."""
        _, mu = (np.asarray(a).real for a in material.moduli.kappa_mu())
        mu_1 = np.zeros_like(mu)
        itau_1 = np.zeros_like(mu)
        for elements, value in _per_layer(material, tau):
            times = _nodal(material, value, elements)[elements]
            if np.any(times <= 0.0):
                raise ValueError("relaxation times must be positive")
            mu_1[elements] = mu[elements]
            itau_1[elements] = 1.0 / times
        return cls(material, [(mu_1, itau_1)])

    @classmethod
    def from_model(cls, material: Material) -> Rheology:
        """The rheology planetmodel's own fields declare: each layer of
        the material's model holding a `viscosity` field is a Maxwell
        body with tau = viscosity / mu; the rest stay elastic."""
        mesh, model = material.mesh, material.model
        viscosity: dict[int, RadialValue] = {}
        for i in np.unique(mesh.layer):
            layer = model.layer(int(i))
            if "viscosity" in layer.fields:
                field = layer.fields["viscosity"]
                viscosity[int(i)] = lambda r, f=field: f.evaluate(r, 0.0, 0.0)
        if not viscosity:
            raise ValueError(
                "no layer of the model holds a viscosity field; give the "
                "rheology explicitly (Rheology.maxwell, say)"
            )
        return cls.maxwell(material, viscosity)

    # -- derived data ------------------------------------------------------

    @property
    def num_branches(self) -> int:
        return len(self.mu)

    def min_relaxation_time(self) -> float:
        """The smallest relaxation time of any active node, the explicit
        stability scale; infinity when nothing relaxes."""
        worst = 0.0
        for mu_k, itau_k in zip(self.mu, self.itau):
            live = mu_k > 0.0
            if np.any(live):
                worst = max(worst, float(np.max(itau_k[live])))
        return 1.0 / worst if worst > 0.0 else np.inf

    def effective_moduli(self, weights: list[np.ndarray]) -> NodalModuli:
        """The moduli with each branch's shear scaled by its nodal weight:
        delta mu = sum_k (w_k - 1) mu_k applied through the isotropic
        perturbation rule (A, C += 4/3 delta mu, F -= 2/3 delta mu,
        L, N += delta mu), which leaves inactive regions untouched."""
        if len(weights) != self.num_branches:
            raise ValueError(
                f"expected {self.num_branches} weight arrays, got {len(weights)}"
            )
        delta = sum((w - 1.0) * mu_k for w, mu_k in zip(weights, self.mu))
        m = self.material.moduli
        return NodalModuli(
            m.A + (4.0 / 3.0) * delta,
            m.C + (4.0 / 3.0) * delta,
            m.F - (2.0 / 3.0) * delta,
            m.L + delta,
            m.N + delta,
        )

    def long_term_material(self) -> Material:
        """The fully relaxed elastic solid, every branch's weight zero:
        the t -> infinity limit a time-domain run must approach."""
        return self.material.with_moduli(
            self.effective_moduli([np.zeros_like(mu) for mu in self.mu])
        )

    def __repr__(self) -> str:
        n = int(np.count_nonzero(self.active))
        return (
            f"Rheology({self.num_branches} branch(es), "
            f"{n} of {self.active.size} nodes relaxing)"
        )
