"""The radial viscoelastic core: state, slaved degree solves, steppers.

The state is the internal variables alone.  Per branch and degree the
spheroidal reduction of Al-Attar & Tromp (2014), Appendix D, leaves
three radial scalars per coefficient column, stored in the order
(M, R, S) with driving strain scalars

    e_M = V,
    e_R = r d_r U - U + (1/2) k^2 V,
    e_S = r d_r V - V + U,

and force weights k^2 (k^2 - 2), 4/3 and k^2; the algebra is recorded
in docs/planning/gia_notes.md.  Evaluating anything else -- the
displacement, the potential, a rate, a step -- contains the
quasi-static degree solves: the elastic system assembled with the
unrelaxed or a per-step effective shear modulus, forced by the load
channels and by the memory force

    b_mem(U', V') = sum_k int mu_k [ (4/3) R_k e_R(U', V')
                                     + k^2 S_k e_S(U', V')
                                     + k^2 (k^2 - 2) M_k V' ] dr.

The steppers are AdGIA's: backward Euler, exponential Euler and the
exponential trapezoid (the default), with the elimination weights of
`weights.py` and one factorised banded system per degree and flavour.
Each degree's columns are independent right-hand sides, so a state may
carry one column (a Love-number history) or the full set of spherical
harmonic orders.

The strain scalars at degree 0 need care: V has no dof, so e_M
vanishes; e_S would evaluate to U, but the S variable carries zero
force weight (k^2 = 0) at that degree, so its drive is zeroed rather
than integrating junk.  Fluid elements have no displacement dofs and
no active branches, so their strain gathers to zero harmlessly.

Everything is in the material's units; times and dt likewise.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..love_numbers.assembly import DegreeSystem
from ..love_numbers.material import Material
from ..love_numbers.solver import inertia_moment, love_numbers
from ..love_numbers.table import LoveNumbers
from .rheology import Rheology
from .weights import (
    backward_euler_weight,
    backward_euler_weight_derivative,
    exponential_trapezoid_weight_derivatives,
    exponential_trapezoid_weights,
)

__all__ = [
    "SCHEMES",
    "ViscoelasticState",
    "ViscoelasticStep",
    "ViscoelasticOperator",
]

SCHEMES = ("exponential_trapezoid", "backward_euler", "exponential_euler")

#: Index order of the internal-variable scalars per branch.
SCALARS = ("M", "R", "S")

#: Maximum number of (scheme, dt) flavours whose factorisations are kept.
_FLAVOUR_CACHE = 4

LoadCallable = Callable[[int, float], Optional[np.ndarray]]


class ViscoelasticState:
    """The internal variables at one time.

    `m[l]` has shape (num_branches, 3, ne_l, ngll, ncols) over the
    degree-l sub-mesh, in the scalar order `SCALARS`; `t` is the time.
    The strain cache holds the degree solves' strain scalars when the
    last step left the displacement consistent with (m, t); a change of
    load at exactly `t` (a load jump) must be followed by
    `invalidate()`, or the step before the jump sees the wrong limit.
    """

    def __init__(self, m: dict[int, np.ndarray], t: float) -> None:
        self.m = m
        self.t = float(t)
        self._strain: dict[int, np.ndarray] | None = None

    @property
    def degrees(self) -> list[int]:
        """The spherical harmonic degrees the state carries."""
        return sorted(self.m)

    def ncols(self, l: int) -> int:
        """The number of load columns carried at degree l."""
        return self.m[l].shape[-1]

    @property
    def cached_strain(self) -> dict[int, np.ndarray] | None:
        """The strain scalars of the last step's final solve, per
        degree, when that solve left the displacement consistent with
        (m, t); None otherwise.  See `ViscoelasticOperator.strain` for
        the layout."""
        return self._strain

    def invalidate(self) -> None:
        """Forgets the cached strain, forcing the next step to re-solve
        at the start time; required after a load jump."""
        self._strain = None

    def copy(self) -> ViscoelasticState:
        """An independent copy of the internal variables, without the
        strain cache."""
        out = ViscoelasticState({l: a.copy() for l, a in self.m.items()}, self.t)
        return out


class _Degree:
    """Per-degree workspace: the unrelaxed system, the strain operators
    and the branch data restricted to the degree's sub-mesh."""

    def __init__(self, rheology: Rheology, l: int, eps: float) -> None:
        self.l = l
        self.k2 = l * (l + 1.0)
        self.system = DegreeSystem(rheology.material, l, eps=eps)
        mesh = rheology.material.mesh
        e0 = self.system.first_element
        self.e0 = e0
        self.ne = mesh.nspec - e0
        jac = mesh.jac[e0:]
        self.B = mesh.r[e0:][:, :, None] * mesh.deriv[None, :, :] / jac[:, None, None]
        self.Wq = mesh.w[None, :] * jac[:, None]
        dof = self.system.dof[mesh.gmap[e0:]]
        self.Udof = dof[:, :, 0]
        self.Vdof = dof[:, :, 1]
        self.Umask = self.Udof >= 0
        self.Vmask = self.Vdof >= 0
        self.mu = [a[e0:] for a in rheology.mu]
        self.itau = [a[e0:] for a in rheology.itau]
        self._load_vec: np.ndarray | None = None
        self._tide_vec: np.ndarray | None = None
        self._load_force_vec: np.ndarray | None = None
        self._load_potential_vec: np.ndarray | None = None
        self._moment_row: np.ndarray | None = None
        # the dof-assembly of nodal force values, [U-block; V-block]
        # flattened, as one sparse product: shared nodes accumulate
        nn = self.ne * mesh.ngll
        rows = np.concatenate((self.Udof[self.Umask], self.Vdof[self.Vmask]))
        cols = np.concatenate(
            (
                np.flatnonzero(self.Umask.ravel()),
                nn + np.flatnonzero(self.Vmask.ravel()),
            )
        )
        self._scatter = csr_matrix(
            (np.ones(cols.size), (rows, cols)), shape=(self.system.ndof, 2 * nn)
        )

    @property
    def load_vec(self) -> np.ndarray:
        if self._load_vec is None:
            self._load_vec = self.system.load()
        return self._load_vec

    @property
    def tide_vec(self) -> np.ndarray:
        """The external-potential force; at degree 0 the centrifugal
        (r/a)^2 potential, the one non-harmonic case in use, so that
        the channel carries the full centrifugal perturbation."""
        if self._tide_vec is None:
            self._tide_vec = self.system.tide(power=2 if self.l == 0 else None)
        return self._tide_vec

    @property
    def load_force_vec(self) -> np.ndarray:
        """The traction-only piece of the unit surface load: the
        zeta^u channel of the generalised Love numbers."""
        if self._load_force_vec is None:
            self._load_force_vec = self.system.load(part="force")
        return self._load_force_vec

    @property
    def load_potential_vec(self) -> np.ndarray:
        """The attraction-only piece of the unit surface load: the
        zeta^phi channel."""
        if self._load_potential_vec is None:
            self._load_potential_vec = self.system.load(part="potential")
        return self._load_potential_vec

    def _gather(self, x: np.ndarray, dofs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Nodal values (ne, ngll, ncols) of one component, zero where absent."""
        return np.where(mask[:, :, None], x[np.clip(dofs, 0, None)], 0.0)

    def strain(self, x: np.ndarray) -> np.ndarray:
        """The driving scalars (3, ne, ngll, ncols) of a solution array,
        in the order (e_M, e_R, e_S)."""
        U = self._gather(x, self.Udof, self.Umask)
        V = self._gather(x, self.Vdof, self.Vmask)
        d = np.empty((3,) + U.shape)
        if self.l == 0:
            d[0] = 0.0
            d[2] = 0.0
        else:
            d[0] = V
            d[2] = np.einsum("eqi,eic->eqc", self.B, V) - V + U
        d[1] = np.einsum("eqi,eic->eqc", self.B, U) - U + 0.5 * self.k2 * V
        return d

    def dual_from_scalars(
        self, cM: np.ndarray, cR: np.ndarray, cS: np.ndarray
    ) -> np.ndarray:
        """The dual vector pairing nodal scalar coefficients with the
        strain operators of the test functions: the transpose of
        `strain` under the plain nodal pairing when the coefficients
        carry no weights, and the memory force when they carry the
        branch moduli and quadrature."""
        ncols = cR.shape[-1]
        k2 = self.k2
        bU = np.einsum("eqi,eqc->eic", self.B, cR) - cR + cS
        bV = 0.5 * k2 * cR + np.einsum("eqi,eqc->eic", self.B, cS) - cS + cM
        nn = cR.shape[0] * cR.shape[1]
        stacked = np.concatenate((bU.reshape(nn, ncols), bV.reshape(nn, ncols)))
        return self._scatter @ stacked

    def apply_weights(self, field: np.ndarray) -> np.ndarray:
        """The nodal weight W of the memory pairing, per branch: the
        branch modulus, the quadrature and the scalar weights
        (k^2 (k^2 - 2), 4/3, k^2) in the order `SCALARS`.  `field` has
        shape (3, ne, ngll, ncols), one set shared by every branch, or
        (K, 3, ne, ngll, ncols); the result always carries the branch
        axis."""
        k2 = self.k2
        w_c = np.array([k2 * (k2 - 2.0), 4.0 / 3.0, k2])
        out = np.empty((len(self.mu),) + field.shape[-4:])
        for k, mu_k in enumerate(self.mu):
            base = field if field.ndim == 4 else field[k]
            out[k] = w_c[:, None, None, None] * (mu_k * self.Wq)[:, :, None] * base
        return out

    def memory_force(self, mem: np.ndarray) -> np.ndarray:
        """The force of the branch states `mem`, shape (K, 3, ne, ngll,
        ncols), on the degree system: b of shape (ndof, ncols)."""
        weighted = self.apply_weights(mem).sum(axis=0)
        return self.dual_from_scalars(weighted[0], weighted[1], weighted[2])

    def strain_transpose(self, y: np.ndarray) -> np.ndarray:
        """The plain transpose of `strain`: y of shape (3, ne, ngll,
        ncols) to a dual vector of shape (ndof, ncols)."""
        return self.dual_from_scalars(y[0], y[1], y[2])

    def surface_dual(self, component: str, values: np.ndarray) -> np.ndarray:
        """The dual vector of the surface-value read: `values` of shape
        (ncols,) placed at the component's surface dof; zero when the
        component has none."""
        b = np.zeros((self.system.ndof, values.shape[-1]))
        i = self.system.surface_dof(component)
        if i >= 0:
            b[i] = values
        return b

    @property
    def moment_row(self) -> np.ndarray:
        """The inertia-moment functional as a vector over the degree's
        dofs (degree 0): q with q . x = sqrt(4 pi) int rho U r^3 dr."""
        if self._moment_row is None:
            material = self.system.material
            mesh = material.mesh
            weights = (
                np.sqrt(4.0 * np.pi)
                * mesh.w[None, :]
                * mesh.jac[:, None]
                * material.rho
                * mesh.r**3
            )[self.e0 :]
            q = np.zeros(self.system.ndof)
            np.add.at(q, self.Udof[self.Umask], weights[self.Umask])
            self._moment_row = q
        return self._moment_row


class ViscoelasticOperator:
    """The viscoelastic loading operator of a rheology, degree by degree.

    Mirrors AdGIA's `ViscoelasticOperator`: the internal variables are
    the state, the displacement is slaved, and the per-(scheme, dt)
    effective systems are factorised once and reused.  `degrees` fixes
    which spherical-harmonic degrees are carried (all of 0..lmax by
    default).
    """

    def __init__(
        self,
        rheology: Rheology,
        lmax: int,
        *,
        degrees: Iterable[int] | None = None,
        eps: float = 1e-8,
    ) -> None:
        if not isinstance(rheology, Rheology):
            raise TypeError(f"expected a Rheology, got {type(rheology).__name__}")
        if lmax < 0:
            raise ValueError("lmax must be non-negative")
        self.rheology = rheology
        self.material = rheology.material
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.degrees = (
            list(range(lmax + 1)) if degrees is None else sorted(set(degrees))
        )
        if self.degrees and (self.degrees[0] < 0 or self.degrees[-1] > lmax):
            raise ValueError("degrees must lie in [0, lmax]")
        self._degree: dict[int, _Degree] = {}
        self._effective: OrderedDict[tuple[str, float], dict[int, DegreeSystem]] = (
            OrderedDict()
        )
        self._weights: dict[tuple[str, float], list[tuple[np.ndarray, ...]]] = {}
        self._weight_derivatives: dict[
            tuple[str, float], list[tuple[np.ndarray, ...]]
        ] = {}
        self._effective_love: dict[tuple[str, float], LoveNumbers] = {}
        # per flavour and degree, the solutions of the unit load and unit
        # tide forcings: the channel right-hand sides are rank one, so a
        # step adds their responses by outer products and never solves
        # for a load
        self._unit: dict[tuple[str, float], dict[int, np.ndarray]] = {}

    # -- machinery ---------------------------------------------------------

    def _deg(self, l: int) -> _Degree:
        if l not in self._degree:
            self._degree[l] = _Degree(self.rheology, l, self.eps)
        return self._degree[l]

    def _scheme_weights(self, scheme: str, dt: float) -> list[tuple[np.ndarray, ...]]:
        """Per-branch full-mesh weight arrays: (w,) for backward Euler
        and (e, alpha, beta) for the exponential schemes."""
        key = (scheme, float(dt))
        if key not in self._weights:
            out = []
            for itau in self.rheology.itau:
                h = dt * itau
                if scheme == "backward_euler":
                    out.append((backward_euler_weight(h),))
                else:
                    out.append(exponential_trapezoid_weights(h))
            self._weights[key] = out
        return self._weights[key]

    def effective_material(self, scheme: str, dt: float) -> Material:
        """The material of the step's effective elastic solve: branch k
        weighted by 1/(1 + h_k) (backward Euler) or phi1(h_k) (the
        exponential trapezoid)."""
        if scheme not in ("backward_euler", "exponential_trapezoid"):
            raise ValueError(f"no effective operator for scheme {scheme!r}")
        weights = self._scheme_weights(scheme, dt)
        if scheme == "backward_euler":
            w = [wk[0] for wk in weights]
        else:
            w = [1.0 - wk[2] for wk in weights]
        return self.material.with_moduli(self.rheology.effective_moduli(w))

    def _effective_system(self, scheme: str, dt: float, l: int) -> DegreeSystem:
        key = (scheme, float(dt))
        if key not in self._effective:
            while len(self._effective) >= _FLAVOUR_CACHE:
                evicted, _ = self._effective.popitem(last=False)
                self._unit.pop(evicted, None)
            self._effective[key] = {}
        self._effective.move_to_end(key)
        systems = self._effective[key]
        if l not in systems:
            systems[l] = DegreeSystem(
                self.effective_material(scheme, dt), l, eps=self.eps
            )
        return systems[l]

    def zero_state(self, ncols: int | Callable[[int], int] = 1) -> ViscoelasticState:
        """The undisturbed state at t = 0, with `ncols` coefficient
        columns per degree (an integer, or a callable of the degree)."""
        K = self.rheology.num_branches
        ngll = self.material.mesh.ngll
        m = {}
        for l in self.degrees:
            deg = self._deg(l)
            n = ncols(l) if callable(ncols) else int(ncols)
            m[l] = np.zeros((K, 3, deg.ne, ngll, n))
        return ViscoelasticState(m, 0.0)

    def _unit_solutions(
        self, key: tuple[str, float], system: DegreeSystem, deg: _Degree
    ) -> np.ndarray:
        """The flavour's responses to the unit load and unit tide of one
        degree, as columns, solved once and cached under the flavour."""
        per_degree = self._unit.setdefault(key, {})
        if deg.l not in per_degree:
            per_degree[deg.l] = system.solve(
                np.column_stack((deg.load_vec, deg.tide_vec))
            )
        return per_degree[deg.l]

    def _add_channels(
        self,
        x: np.ndarray,
        deg: _Degree,
        key: tuple[str, float],
        system: DegreeSystem,
        t: float,
        load: LoadCallable | None,
        tide: LoadCallable | None,
    ) -> np.ndarray:
        """Adds the load channels at time t to a solution in place, by
        the cached unit responses: solving is linear in the forces."""
        if load is None and tide is None:
            return x
        units = self._unit_solutions(key, system, deg)
        if load is not None:
            sigma = load(deg.l, t)
            if sigma is not None:
                x += units[:, 0, None] * np.asarray(sigma, dtype=float)[None, :]
        if tide is not None:
            psi = tide(deg.l, t)
            if psi is not None:
                x += units[:, 1, None] * np.asarray(psi, dtype=float)[None, :]
        return x

    # -- the slaved solve ----------------------------------------------------

    def solve_elastic(
        self,
        state: ViscoelasticState,
        *,
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> dict[int, np.ndarray]:
        """The quasi-static solutions consistent with (m, t): the
        unrelaxed systems forced by the loads at `state.t` and the
        memory force of m.  Returns the solution array per degree;
        `surface` reads off values, `DegreeSystem.expand` the fields."""
        out = {}
        for l in state.degrees:
            deg = self._deg(l)
            x = deg.system.solve(deg.memory_force(state.m[l]))
            out[l] = self._add_channels(
                x, deg, ("unrelaxed", 0.0), deg.system, state.t, load, tide
            )
        return out

    def strain(self, solutions: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """The driving strain scalars of degree solutions: per degree an
        array of shape (3, ne, ngll, ncols) over the degree's sub-mesh,
        in the scalar order `SCALARS` = (M, R, S), i.e. (e_M, e_R, e_S)."""
        return {l: self._deg(l).strain(x) for l, x in solutions.items()}

    def surface(
        self, solutions: dict[int, np.ndarray], component: str
    ) -> dict[int, np.ndarray]:
        """The surface value of U, V or phi per degree and column, zero
        where the component has no dof (V at l = 0, phi at l = 1)."""
        out = {}
        for l, x in solutions.items():
            i = self._deg(l).system.surface_dof(component)
            out[l] = x[i].copy() if i >= 0 else np.zeros(x.shape[-1])
        return out

    # -- steppers -----------------------------------------------------------

    def step(
        self,
        state: ViscoelasticState,
        dt: float,
        *,
        scheme: str = "exponential_trapezoid",
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> tuple[ViscoelasticState, dict[int, np.ndarray]]:
        """One step of `scheme` from `state.t`, under the given load
        channels (callables (l, t) -> coefficient columns or None).

        Returns the new state and the degree solutions the step
        computed last: for the implicit and trapezoid schemes these are
        consistent with the new state at t + dt; for the exponential
        Euler they are the start-of-step solutions at t.  The new state
        carries the strain cache the next step can reuse; call
        `invalidate()` on it across a load jump.
        """
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if scheme == "backward_euler":
            return self._step_backward_euler(state, dt, load, tide)
        if scheme == "exponential_euler":
            return self._step_exponential_euler(state, dt, load, tide)
        return self._step_exponential_trapezoid(state, dt, load, tide)

    def step_back(
        self,
        state: ViscoelasticState,
        dt: float,
        *,
        scheme: str = "exponential_trapezoid",
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> tuple[ViscoelasticState, dict[int, np.ndarray]]:
        """The exact algebraic inverse of `step`: the state one step of
        `dt` EARLIER, with the degree solutions at that earlier time
        and a consistent strain cache, so inversions chain.  The load
        channels given are the EARLIER time's; the state itself must
        carry its strain cache (a step, or a recording run, leaves
        one).  For the exponential trapezoid the inverse is the same
        update evaluated at -dt with the start and end strains
        swapped; backward Euler inverts explicitly.  Dissipation runs
        the other way in time, so roundoff is amplified by exp(h) per
        branch and node: a checkpointing driver budgets the
        accumulated h between stored states.
        """
        if scheme not in ("exponential_trapezoid", "backward_euler"):
            raise ValueError(
                "a step inverts for scheme 'exponential_trapezoid' or "
                f"'backward_euler', got {scheme!r}"
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if state._strain is None:
            raise ValueError(
                "the state must carry its strain cache; reconstruct from "
                "a recorded state or a completed step"
            )
        if scheme == "exponential_trapezoid":
            return ViscoelasticStep(self, state, -dt, scheme, None, None).finish(
                load=load, tide=tide
            )
        d1 = state._strain
        m_prev: dict[int, np.ndarray] = {}
        for l in state.degrees:
            deg = self._deg(l)
            m = np.empty_like(state.m[l])
            for k, itau in enumerate(deg.itau):
                hk = (dt * itau)[None, :, :, None]
                m[k] = (1.0 + hk) * state.m[l][k] - hk * d1[l]
            m_prev[l] = m
        prev = ViscoelasticState(m_prev, state.t - dt)
        solutions = self.solve_elastic(prev, load=load, tide=tide)
        prev._strain = {l: self._deg(l).strain(x) for l, x in solutions.items()}
        return prev, solutions

    def _start_strain(
        self,
        state: ViscoelasticState,
        load: LoadCallable | None,
        tide: LoadCallable | None,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """The strain scalars at the start of a step, from the cache when
        the previous step left the displacement consistent."""
        if state._strain is not None:
            return state._strain, {}
        solutions = self.solve_elastic(state, load=load, tide=tide)
        return {l: self._deg(l).strain(x) for l, x in solutions.items()}, solutions

    def _step_exponential_trapezoid(self, state, dt, load, tide):
        return self.start_step(
            state, dt, scheme="exponential_trapezoid", load=load, tide=tide
        ).finish(load=load, tide=tide)

    def _step_backward_euler(self, state, dt, load, tide):
        return self.start_step(
            state, dt, scheme="backward_euler", load=load, tide=tide
        ).finish(load=load, tide=tide)

    def start_step(
        self,
        state: ViscoelasticState,
        dt: float,
        *,
        scheme: str = "exponential_trapezoid",
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> ViscoelasticStep:
        """Opens an implicit or trapezoid step before the loads at the
        end time are known: the sea-level driver reads the step's
        memory responses, closes its equilibrium with the matching
        effective Love numbers, and completes the step with `finish`
        under the converged loads.  The load channels given here are
        those at the START time, needed by the trapezoid's d^n (and
        ignored when the state carries a consistent strain cache)."""
        if scheme not in ("exponential_trapezoid", "backward_euler"):
            raise ValueError(
                "a two-phase step is implicit: scheme must be "
                f"'exponential_trapezoid' or 'backward_euler', got {scheme!r}"
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        return ViscoelasticStep(self, state, dt, scheme, load, tide)

    def effective_love_numbers(self, scheme: str, dt: float) -> LoveNumbers:
        """The Love numbers of the step's effective elastic solid, the
        instantaneous response of an implicit or trapezoid step; cached
        per (scheme, dt)."""
        key = (scheme, float(dt))
        if key not in self._effective_love:
            self._effective_love[key] = love_numbers(
                self.effective_material(scheme, dt), self.lmax, eps=self.eps
            )
        return self._effective_love[key]

    def elastic_love_numbers(self) -> LoveNumbers:
        """The Love numbers of the unrelaxed solid: the instantaneous
        response at a time, before any step is taken; cached."""
        key = ("unrelaxed", 0.0)
        if key not in self._effective_love:
            self._effective_love[key] = love_numbers(
                self.material, self.lmax, eps=self.eps
            )
        return self._effective_love[key]

    def _step_exponential_euler(self, state, dt, load, tide):
        d_start, solutions = self._start_strain(state, load, tide)
        weights = self._scheme_weights("exponential_trapezoid", dt)
        m_new: dict[int, np.ndarray] = {}
        for l in state.degrees:
            deg = self._deg(l)
            mem = np.empty_like(state.m[l])
            for k, (e, _, _) in enumerate(weights):
                ek = e[deg.e0 :][None, :, :, None]
                mem[k] = ek * state.m[l][k] + (1.0 - ek) * d_start[l]
            m_new[l] = mem
        # the displacement is not consistent with the new state
        return ViscoelasticState(m_new, state.t + dt), solutions

    # -- views ----------------------------------------------------------------

    def min_relaxation_time(self) -> float:
        """The explicit stability scale of the rheology."""
        return self.rheology.min_relaxation_time()

    def prime_strain(
        self,
        state: ViscoelasticState,
        *,
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> dict[int, np.ndarray]:
        """Ensures the state carries a strain cache consistent with
        (m, t) under the given loads, solving once if it does not, and
        returns the cached strains; an adaptive driver needs them for
        the embedded error estimate."""
        strains, _ = self._start_strain(state, load, tide)
        if state._strain is None:
            state._strain = strains
        return strains

    # -- adaptive stepping --------------------------------------------------

    def adaptive_step(
        self,
        state: ViscoelasticState,
        dt: float,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
        min_factor: float = 0.2,
        max_factor: float = 4.0,
        safety: float = 0.9,
    ):
        """One trial exponential-trapezoid step with its embedded ETD1
        error estimate (nodal work only, no extra solve)::

            err = RMS of (m - m_hat) / (atol + rtol max(abs m, abs m_new)),
            dt_next = dt clamp(safety err^(-1/2), min_factor, max_factor),

        the step accepted when err <= 1.  The estimate is the
        first-order companion's, so it is conservative for the
        second-order solution that is propagated; choose rtol
        accordingly.  Returns (new_state, solutions, err, dt_next),
        the first two None on rejection.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        d_start, _ = self._start_strain(state, load, tide)
        if state._strain is None:
            state._strain = d_start  # the trial shares the start solve
        weights = self._scheme_weights("exponential_trapezoid", dt)
        new_state, solutions = self.step(
            state, dt, scheme="exponential_trapezoid", load=load, tide=tide
        )
        if atol is None:
            # the internal variables carry the degree-basis scalars'
            # units, so the absolute floor must follow the state's own
            # magnitude: components a millionth of the largest do not
            # steer the step
            top = max(
                (float(np.max(np.abs(new_state.m[l]))) for l in state.degrees),
                default=0.0,
            )
            atol = 1e-6 * rtol * top if top > 0.0 else 1e-300
        total = 0.0
        count = 0
        for l in state.degrees:
            deg = self._deg(l)
            for k, (e, _, _) in enumerate(weights):
                ek = e[deg.e0 :][None, :, :, None]
                companion = ek * state.m[l][k] + (1.0 - ek) * d_start[l]
                scale = atol + rtol * np.maximum(
                    np.abs(state.m[l][k]), np.abs(new_state.m[l][k])
                )
                ratio = (new_state.m[l][k] - companion) / scale
                total += float(np.sum(ratio * ratio))
                count += ratio.size
        err = np.sqrt(total / count) if count else 0.0
        factor = max_factor if err == 0.0 else safety / np.sqrt(err)
        dt_next = dt * float(np.clip(factor, min_factor, max_factor))
        if err <= 1.0:
            return new_state, solutions, err, dt_next
        return None, None, err, dt_next

    def integrate(
        self,
        state: ViscoelasticState,
        t_final: float,
        dt: float,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
        quantise: bool = False,
        max_steps: int = 10000,
    ):
        """Integrates to `t_final` with the adaptive trapezoid, hitting
        the end exactly.  With `quantise` the proposed steps snap down
        to dt 2^n, so the flavour cache's factorisations are reused
        instead of re-assembled at every acceptance -- the economy of
        a per-step effective operator is the reason to prefer it here.
        Returns (state, solutions, info) with the last accepted step's
        solutions and the counters and step history in `info`.
        """
        if t_final <= state.t:
            raise ValueError("t_final must lie beyond the state's time")
        base = dt
        accepted = 0
        rejected = 0
        history: list[float] = []
        solutions = None
        same_rung = 0
        for _ in range(max_steps):
            remaining = t_final - state.t
            if remaining <= 0.0:
                break
            trial = min(dt, remaining)
            new_state, sols, err, dt_next = self.adaptive_step(
                state,
                trial,
                rtol=rtol,
                atol=atol,
                load=load,
                tide=tide,
            )
            raw = dt_next
            if quantise and dt_next < remaining:
                # the floor keeps the ladder stable, and a stagnation
                # promoter climbs a rung after eight healthy accepts,
                # since the controller equilibrates below the factor of
                # two the floor alone would need
                dt_next = base * 2.0 ** np.floor(np.log2(dt_next / base))
            if new_state is None:
                rejected += 1
                same_rung = 0
                dt = dt_next
                continue
            state = new_state
            solutions = sols
            accepted += 1
            history.append(trial)
            if quantise:
                if dt_next == dt and raw > 1.5 * dt:
                    same_rung += 1
                    if same_rung >= 8:
                        dt_next = 2.0 * dt
                        same_rung = 0
                else:
                    same_rung = 0
            dt = dt_next
        else:
            raise RuntimeError("adaptive integration exceeded max_steps")
        info = {"accepted": accepted, "rejected": rejected, "steps": history}
        return state, solutions, info

    # -- the transposes' surface: what an adjoint driver reads ------------

    def degree(self, l: int) -> _Degree:
        """The degree workspace: the strain map and its transpose, the
        memory force, the weight application, the surface and moment
        dual rows.  Advanced use; the adjoint driver is the consumer."""
        return self._deg(l)

    def effective_system(self, scheme: str, dt: float, l: int) -> DegreeSystem:
        """The factorised effective system of a flavour and degree."""
        return self._effective_system(scheme, dt, l)

    def unit_responses(self, scheme: str | None, dt: float, l: int) -> np.ndarray:
        """The flavour's solutions of the unit load and unit tide of one
        degree, as the two columns; `scheme=None` gives the unrelaxed
        flavour's."""
        deg = self._deg(l)
        if scheme is None:
            return self._unit_solutions(("unrelaxed", 0.0), deg.system, deg)
        return self._unit_solutions(
            (scheme, float(dt)), self._effective_system(scheme, dt, l), deg
        )

    def split_unit_responses(self, scheme: str | None, dt: float, l: int) -> np.ndarray:
        """The flavour's solutions of the SPLIT unit load channels of
        one degree, as two columns: the traction-only (zeta^u) and
        attraction-only (zeta^phi) pieces, which the continuous
        adjoint's jump needs for displacement and potential
        measurements; `scheme=None` gives the unrelaxed flavour's."""
        deg = self._deg(l)
        key = (
            ("unrelaxed-split", 0.0)
            if scheme is None
            else (f"{scheme}-split", float(dt))
        )
        per_degree = self._unit.setdefault(key, {})
        if l not in per_degree:
            system = (
                deg.system if scheme is None else self._effective_system(scheme, dt, l)
            )
            per_degree[l] = system.solve(
                np.column_stack((deg.load_force_vec, deg.load_potential_vec))
            )
        return per_degree[l]

    def scheme_weight_derivatives(
        self, scheme: str, dt: float
    ) -> list[tuple[np.ndarray, ...]]:
        """The h-derivatives of the per-branch scheme weights at this
        step size, full-mesh nodal arrays: (dw,) for backward Euler and
        (de, dalpha, dbeta) for the exponential trapezoid.  A viscosity
        sensitivity pairs them with dh = dt d(1/tau)."""
        key = (scheme, float(dt))
        if key not in self._weight_derivatives:
            out = []
            for itau in self.rheology.itau:
                h = dt * itau
                if scheme == "backward_euler":
                    out.append((backward_euler_weight_derivative(h),))
                else:
                    out.append(exponential_trapezoid_weight_derivatives(h))
            self._weight_derivatives[key] = out
        return self._weight_derivatives[key]

    def scheme_weights(self, scheme: str, dt: float) -> list[tuple[np.ndarray, ...]]:
        """The per-branch nodal weight arrays of a flavour: (w,) for
        backward Euler, (e, alpha, beta) for the exponential schemes."""
        return self._scheme_weights(scheme, dt)

    def __repr__(self) -> str:
        return (
            f"ViscoelasticOperator({self.rheology!r}, "
            f"degrees {self.degrees[0]}..{self.degrees[-1]})"
        )


class ViscoelasticStep:
    """One opened implicit or trapezoid step, from `start_step`.

    On construction the start state is decayed to the step's memory
    state m~ (e m^n + alpha d^n for the trapezoid, m^n / (1 + h) for
    backward Euler), which is FIXED for the step: the end-time loads
    enter linearly on top of it.  `memory_solutions` gives the
    response to the memory alone through the effective systems --
    the field a sea-level driver folds into its direct effect -- and
    `finish` completes the step once the end-time loads are known.
    """

    def __init__(
        self,
        operator: ViscoelasticOperator,
        state: ViscoelasticState,
        dt: float,
        scheme: str,
        load: LoadCallable | None,
        tide: LoadCallable | None,
    ) -> None:
        self.operator = operator
        self.scheme = scheme
        self.dt = float(dt)
        self.t = state.t + dt
        self.degrees = state.degrees
        self._weights = operator._scheme_weights(scheme, dt)
        self._mem: dict[int, np.ndarray] = {}
        self._bmem: dict[int, np.ndarray] = {}
        self._memory_solutions: dict[int, np.ndarray] | None = None
        if scheme == "exponential_trapezoid":
            d_start, _ = operator._start_strain(state, load, tide)
        for l in self.degrees:
            deg = operator._deg(l)
            mem = np.empty_like(state.m[l])
            for k, wk in enumerate(self._weights):
                if scheme == "backward_euler":
                    w = wk[0][deg.e0 :][None, :, :, None]
                    mem[k] = w * state.m[l][k]
                else:
                    e = wk[0][deg.e0 :][None, :, :, None]
                    alpha = wk[1][deg.e0 :][None, :, :, None]
                    mem[k] = e * state.m[l][k] + alpha * d_start[l]
            self._mem[l] = mem

    def _system(self, l: int) -> DegreeSystem:
        return self.operator._effective_system(self.scheme, self.dt, l)

    def _memory_force(self, l: int) -> np.ndarray:
        if l not in self._bmem:
            self._bmem[l] = self.operator._deg(l).memory_force(self._mem[l])
        return self._bmem[l]

    def effective_love_numbers(self) -> LoveNumbers:
        """The Love numbers characterising this step's instantaneous
        response to its loads."""
        return self.operator.effective_love_numbers(self.scheme, self.dt)

    def memory_solutions(self) -> dict[int, np.ndarray]:
        """The degree solutions of the memory state alone (zero load)
        through the effective systems, cached."""
        if self._memory_solutions is None:
            self._memory_solutions = {
                l: self._system(l).solve(self._memory_force(l)) for l in self.degrees
            }
        return self._memory_solutions

    def memory_surface(self, component: str) -> dict[int, np.ndarray]:
        """The surface U, V or phi of the memory response, per degree
        and column."""
        return self.operator.surface(self.memory_solutions(), component)

    def inertia_moment(self) -> np.ndarray:
        """The inertia moment sqrt(4 pi) int rho U r^3 dr of the
        degree-0 memory response, per column: the memory field's
        contribution to the trace of the inertia tensor, which the
        axial rotational feedback needs beside the load-driven terms
        the effective axial Love numbers carry."""
        if 0 not in self.degrees:
            raise ValueError("the state does not carry degree 0")
        x = self.memory_solutions()[0]
        system = self.operator._deg(0).system
        material = self.operator.material
        out = np.empty(x.shape[-1])
        for c in range(x.shape[-1]):
            U, _, _ = system.expand(x[:, c])
            out[c] = inertia_moment(material, U).real
        return out

    def finish(
        self,
        *,
        load: LoadCallable | None = None,
        tide: LoadCallable | None = None,
    ) -> tuple[ViscoelasticState, dict[int, np.ndarray]]:
        """Completes the step under the end-time loads: one effective
        solve per degree, the state update, and the solutions, which
        are consistent with the new state."""
        op = self.operator
        memory = self.memory_solutions()
        m_new: dict[int, np.ndarray] = {}
        solutions: dict[int, np.ndarray] = {}
        strains: dict[int, np.ndarray] = {}
        for l in self.degrees:
            deg = op._deg(l)
            system = self._system(l)
            x = op._add_channels(
                memory[l].copy(),
                deg,
                (self.scheme, self.dt),
                system,
                self.t,
                load,
                tide,
            )
            d1 = deg.strain(x)
            mem = self._mem[l]
            for k, wk in enumerate(self._weights):
                if self.scheme == "backward_euler":
                    w = wk[0][deg.e0 :][None, :, :, None]
                    mem[k] = mem[k] + (1.0 - w) * d1
                else:
                    beta = wk[2][deg.e0 :][None, :, :, None]
                    mem[k] = mem[k] + beta * d1
            m_new[l] = mem
            solutions[l] = x
            strains[l] = d1
        out = ViscoelasticState(m_new, self.t)
        out._strain = strains
        return out, solutions
