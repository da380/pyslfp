"""The coupled forward GIA time loop, its adjoints, and the calibration
of the initial sea level.

Each implicit or trapezoid step opens on the radial side
(`ViscoelasticOperator.start_step`), which fixes the step's memory
state; its instantaneous response to loads is then an elastic one with
the step's effective Love numbers, so the sea level equation closes
with the solver of `pyslfp.core`: the memory field enters through the
response channels, its degree-0 inertia moment joins the axial
feedback, and the ocean and rotational couplings converge inside the
one sea level solve.  The converged load and centrifugal potential
then finish the radial step, updating the internal variables.  Under
frozen shorelines the ocean function is the background state's
throughout and each step solves the linear equation; under migrating
shorelines each step solves the nonlinear equation, the ocean function
following the flotation condition.

The adjoints give the sensitivity of an objective functional to the
ice history, the initial sea level and the relaxation times, either as
the exact transpose of the time-stepped map (`adjoint_ice_history`) or
by discretising the continuous adjoint equations
(`continuous_adjoint_ice_history`).

The driver's public quantities -- times, loads, the returned fields --
are in the Earth model's non-dimensional units, like the rest of the
library; the radial operator works in its material's units, and the
driver converts at the boundary between the two.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Sequence
from typing import Optional
from dataclasses import dataclass, field, replace

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from planetmodel import units

from ..core import EarthModel
from ..ice import BaseIceModel
from ..love_numbers import fluid_tidal_number
from ..core import AndersonMixer, SeaLevelEquation
from ..core import EarthState
from .operator import ViscoelasticOperator, ViscoelasticState
from .rheology import Rheology

__all__ = [
    "FinalSeaLevelMisfit",
    "ForwardTrajectory",
    "IceKernels",
    "InitialSeaLevelCalibration",
    "Measurement",
    "SeaLevelSnapshot",
    "SeaLevelEvolution",
    "calibrate_initial_sea_level",
]

#: Seconds per kiloyear (Julian years), for the date conversions.
KYR = 1.0e3 * 365.25 * 86400.0

POTENTIAL = units.LENGTH**2 * units.TIME**-2
SURFACE_DENSITY = units.DENSITY * units.LENGTH
MOMENT = units.MASS * units.LENGTH**2


@dataclass(frozen=True)
class Measurement:
    """The duals of one observation time: the objective gains

        int g_xi xi dS + int g_u u dS + int g_phi phi dS
        + g_omega . omega

    at that date, with the grids paired by the surface integral.  A
    bare SHGrid given as a measurement is shorthand for a sea-level
    dual alone."""

    sea_level: Optional[SHGrid] = None
    displacement: Optional[SHGrid] = None
    gravity_potential: Optional[SHGrid] = None
    angular_velocity: Optional[np.ndarray] = None


@dataclass(frozen=True)
class IceKernels:
    """Sensitivity kernels of one objective functional with respect to
    the ice history, from an adjoint calculation, in the Earth model's
    units.

    Two views of the same object are carried.  The DISCRETE view:
    `load_duals[n]` is the dual of the direct ice-load grid at date
    node n, and `thickness[n]` the dual of the ice thickness there,
    the grounding factor applied and the initial-thickness dependence
    folded into node 0; these satisfy the gradient identity exactly
    for perturbations on the date grid, but their values lump the
    kernel's singular parts in with the continuous part, so they do
    not converge pointwise under grid refinement.

    The DISTRIBUTIONAL view separates the parts (Yu et al. 2025,
    Section 3.7 and eq. 81): the kernel is

        K_I(x, t) = density(x, t)
                    + sum_i instantaneous_i(x) delta(t - t_i)
                    + baseline(x) delta(t - t0),

    with a delta at each observation time t_i, whose amplitude is the
    generalised (instantaneous, unrelaxed) fingerprint adjoint of that
    measurement and is independent of the date grid; the delta at the
    initial time from the thickness baseline; and a continuous
    `density` (per kiloyear) that converges under grid refinement.
    The QUELLED view (Backus; see the discussion in Yu et al. 2025)
    removes the singularities altogether by the conjugate change of
    variable: with the history written as I(t) = I(t0) + int di/dt,
    the kernel with respect to the RATE of thickness change is the
    cumulative time integral of K_I, which turns each delta into a
    finite jump: `rate[k]` is that kernel on the interval between
    dates[k-1] and dates[k] -- bounded, piecewise continuous in time
    with finite jumps at the observation dates only, and pairing with
    rate perturbations that generate thickness changes CONTINUOUS in
    time:

        delta J = sum_k h_k int rate[k] (dI/dt)_k dS,

    h_k the interval length in kiloyears and the rate constant on the
    interval.  In this parameterisation the dual of the initial
    thickness vanishes identically under frozen shorelines (a uniform
    thickness shift loads nothing), which the calculation satisfies by
    construction.  Any remaining inversion-side smoothing -- Sobolev
    duals in the manner of pygeoinf -- acts on these views."""

    dates: np.ndarray
    load_duals: list[SHGrid] = field(repr=False)
    thickness: list[SHGrid] = field(repr=False)
    baseline: SHGrid = field(repr=False)

    def at_dates(self, model_dates: "Sequence[float]") -> list[SHGrid]:
        """The thickness duals re-expressed on an ice model's own dates.

        An ice model interpolates linearly between its slices (the
        `BaseIceModel` contract), so a run-grid node value is a convex
        combination of the two bracketing slices; this applies the
        transpose of that interpolation, returning one dual grid per
        model date with

            delta J = sum_j int at_dates[j] delta I(T_j) dS

        exact whenever the forward run read its thicknesses through
        the same linear interpolation.  `model_dates` are kiloyears
        before present, decreasing, and must bracket the run's dates.
        """
        model_dates = np.asarray(model_dates, dtype=float)
        if (
            model_dates.ndim != 1
            or model_dates.size < 2
            or np.any(np.diff(model_dates) >= 0.0)
        ):
            raise ValueError("model dates must be decreasing")
        if (
            self.dates[0] > model_dates[0] + 1e-9
            or self.dates[-1] < model_dates[-1] - 1e-9
        ):
            raise ValueError("model dates must bracket the run's dates")
        out = [np.zeros_like(self.thickness[0].data) for _ in model_dates]
        for n, t in enumerate(self.dates):
            j = int(np.searchsorted(-model_dates, -t, side="right")) - 1
            j = min(max(j, 0), model_dates.size - 2)
            span = model_dates[j] - model_dates[j + 1]
            lam = (model_dates[j] - t) / span
            out[j] += (1.0 - lam) * self.thickness[n].data
            out[j + 1] += lam * self.thickness[n].data
        grid = self.thickness[0].grid
        return [SHGrid.from_array(o, grid=grid) for o in out]

    #: The kernel with respect to the INITIAL sea level: under a
    #: migrating trajectory, rho_w sum_n (C_n - C0) load_duals[n] (the
    #: discrete analogue of Yu et al. 2025, eq. 65), for objectives on
    #: the sea level CHANGE; identically zero under frozen shorelines.
    #: An objective on absolute sea level adds the sum of its sea-level
    #: duals.
    initial_sea_level: SHGrid = field(repr=False)
    instantaneous: dict[float, SHGrid] = field(repr=False)
    density: list[SHGrid] = field(repr=False)
    rate: list[SHGrid] = field(repr=False)
    #: The inverse-relaxation-time kernels dJ / d itau(r) per branch,
    #: nodal (nspec, ngll) arrays over the material's mesh in its
    #: units, when the adjoint was given the forward trajectory
    #: (`internal_states`); None otherwise.
    itau: Optional[list[np.ndarray]] = field(default=None, repr=False)

    def viscosity(self, rheology: Rheology) -> list[np.ndarray]:
        """The viscosity kernels dJ / d eta(r), nodal per branch, from
        the inverse-relaxation-time kernels: itau = mu / eta gives
        d itau / d eta = -itau^2 / mu, so an elastic branch (itau = 0)
        has no viscosity sensitivity in this parameterisation."""
        if self.itau is None:
            raise ValueError("no itau kernels: run the adjoint with internal_states")
        out = []
        for kernel, mu_k, itau_k in zip(self.itau, rheology.mu, rheology.itau):
            safe = np.where(mu_k > 0.0, mu_k, 1.0)
            out.append(np.where(mu_k > 0.0, -(itau_k**2) / safe, 0.0) * kernel)
        return out


@dataclass(frozen=True)
class SeaLevelSnapshot:
    """The surface fields at one time of an evolution, in the Earth
    model's non-dimensional units: the sea level change, vertical
    displacement and gravity potential change since the start, and the
    angular velocity change [omega_x, omega_y, omega_z].  A run driven
    by an ice history also records `date`, in kiloyears before present."""

    t: float
    sea_level_change: SHGrid = field(repr=False)
    displacement: SHGrid = field(repr=False)
    gravity_potential_change: SHGrid = field(repr=False)
    angular_velocity_change: np.ndarray = field(repr=False)
    date: float | None = None
    #: The step's ocean function under migrating shorelines (None when
    #: the shorelines are frozen at the background state's): the C(t)
    #: record an adjoint calculation reads back.
    ocean_function: SHGrid | None = field(default=None, repr=False)
    #: The radial internal state at this time, recorded on request
    #: (`record`): the forward trajectory a viscosity-kernel adjoint
    #: pairs its duals with.  Under a checkpointing stride only the
    #: kept nodes carry one.
    internal_state: "ViscoelasticState | None" = field(default=None, repr=False)
    #: The converged surface load and centrifugal coefficients of this
    #: node, kept by every recording run: with these a reverse sweep
    #: reconstructs the states between checkpoints by exact backward
    #: steps.
    load_coefficients: np.ndarray | None = field(default=None, repr=False)
    tide_coefficients: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class FinalSeaLevelMisfit:
    """The misfit of a run's final sea level to a target, and its
    gradient with respect to the initial sea level.

    With S0 the background state's sea level, xi(t_f) the sea level
    change at the last date and S_f the target,

        J = (1/2) int w (S0 + xi(t_f) - S_f)^2 dS,

    `value` is J, `residual` the field S0 + xi(t_f) - S_f, and
    `gradient` the derivative of J with respect to S0 in the surface
    pairing (None when not asked for).  `snapshots` and
    `internal_state` are those of the forward run."""

    value: float
    residual: SHGrid = field(repr=False)
    gradient: Optional[SHGrid] = field(repr=False)
    snapshots: list[SeaLevelSnapshot] = field(repr=False)
    internal_state: ViscoelasticState = field(repr=False)


@dataclass(frozen=True)
class InitialSeaLevelCalibration:
    """The outcome of calibrating the initial sea level against a final
    one.

    `state` is the calibrated background state at the first date, and
    `snapshots` and `internal_state` are the run from it.  `misfits`
    holds the weighted root-mean-square residual of every run, in the
    Earth model's units, and `updates` the largest update of each
    iteration relative to the largest value of the initial sea level;
    `converged` says whether the tolerance was met."""

    state: EarthState = field(repr=False)
    snapshots: list[SeaLevelSnapshot] = field(repr=False)
    internal_state: ViscoelasticState = field(repr=False)
    misfits: list[float]
    updates: list[float]
    converged: bool
    method: str
    shorelines: str


def _record_stride(record: bool | int) -> int:
    """The recording stride: 0 for no recording, 1 for every node
    (record=True), or the given checkpoint stride -- internal states
    kept every `record` nodes (and at the last), the per-node load
    and centrifugal coefficients always kept, and the gaps
    reconstructed by exact backward steps during a reverse sweep.
    Backwards the dissipation amplifies roundoff by exp(h) per
    branch and node, so the accumulated h = dt / tau across a gap
    is the digits spent."""
    if record is True:
        return 1
    if record is False:
        return 0
    stride = int(record)
    if stride < 1:
        raise ValueError("record must be True, False, or a stride >= 1")
    return stride


@dataclass(frozen=True)
class ForwardTrajectory:
    """A recorded forward trajectory for the viscosity-kernel adjoints:
    the internal states, at every node or at sparse checkpoints, and
    the per-node surface load and centrifugal coefficients that let a
    reverse sweep reconstruct the gaps by exact backward steps.  The
    reconstruction amplifies roundoff by exp(h) per branch and node
    across a gap, so the recording stride buys memory at the price of
    trailing digits; see `SeaLevelEvolution.run`."""

    states: list[ViscoelasticState | None]
    loads: Optional[list[np.ndarray]] = None
    tides: Optional[list[np.ndarray]] = None

    @classmethod
    def from_snapshots(cls, snapshots: Sequence[SeaLevelSnapshot]) -> ForwardTrajectory:
        """The trajectory a recording run's snapshots carry."""
        states = [snap.internal_state for snap in snapshots]
        loads = [snap.load_coefficients for snap in snapshots]
        tides = [snap.tide_coefficients for snap in snapshots]
        if any(ld is None for ld in loads) or any(td is None for td in tides):
            if all(st is not None for st in states):
                return cls(states)
            raise ValueError(
                "the snapshots do not carry the recorded loads: run with record"
            )
        return cls(states, loads, tides)


class _TrajectoryCursor:
    """Serves the forward internal states to a reverse sweep, strictly
    backwards: stored states where the recording kept them, exact
    backward steps under the recorded loads between."""

    def __init__(self, evolution, trajectory, dates, scheme) -> None:
        self._evolution = evolution
        self._trajectory = trajectory
        kyr = KYR / evolution.model.parameters.scales.factor(units.TIME)
        self._dt = (dates[:-1] - dates[1:]) * kyr * evolution._to_mat_time
        self._scheme = scheme
        self._node = len(trajectory.states) - 1
        self._state = trajectory.states[-1]

    def state(self, node: int) -> ViscoelasticState:
        """The forward state at a date node, reconstructing as needed."""
        if node > self._node:
            raise ValueError("the trajectory cursor moves backwards")
        while self._node > node:
            target = self._node - 1
            stored = self._trajectory.states[target]
            if stored is not None:
                self._state = stored
            else:
                evolution = self._evolution
                self._state, _ = evolution.operator.step_back(
                    self._state,
                    float(self._dt[target]),
                    scheme=self._scheme,
                    load=evolution._columns(
                        self._trajectory.loads[target], evolution._to_mat_density
                    ),
                    tide=evolution._columns(
                        self._trajectory.tides[target], evolution._to_mat_potential
                    ),
                )
            self._node = target
        return self._state


def _normalise_measurement(
    measurement: dict[float, SHGrid | Measurement],
    dates: np.ndarray,
    model: EarthModel,
) -> dict[int, Measurement]:
    """Measurement duals keyed by date, checked and re-keyed by node
    index; a bare grid means a sea-level dual."""
    duals: dict[int, Measurement] = {}
    for key, value in measurement.items():
        matches = np.flatnonzero(np.isclose(dates, key))
        if matches.size != 1:
            raise ValueError(f"measurement date {key} is not on the date grid")
        if not isinstance(value, Measurement):
            value = Measurement(sea_level=value)
        for grid in (value.sea_level, value.displacement, value.gravity_potential):
            if grid is not None:
                model.check_field(grid)
        duals[int(matches[0])] = value
    return duals


class SeaLevelEvolution:
    """Forward GIA: ice history in, sea level, displacement, potential
    and rotation histories out.

    `operator` is the radial viscoelastic operator, which must carry
    every degree up to the state's lmax; `state` the background
    `EarthState` at the start of the run, whose ocean function the
    frozen-shoreline equation holds fixed.
    """

    def __init__(
        self,
        operator: ViscoelasticOperator,
        state: EarthState,
        *,
        scheme: str = "exponential_trapezoid",
    ) -> None:
        if scheme not in ("exponential_trapezoid", "backward_euler"):
            raise ValueError(
                "the coupled loop steps implicitly: scheme must be "
                f"'exponential_trapezoid' or 'backward_euler', got {scheme!r}"
            )
        model = state.model
        if operator.lmax != model.lmax:
            raise ValueError(
                f"the operator reaches degree {operator.lmax}, "
                f"the model degree {model.lmax}"
            )
        if operator.degrees != list(range(model.lmax + 1)):
            raise ValueError("the operator must carry every degree up to lmax")
        self.operator = operator
        self.state = state
        self.model = model
        self.scheme = scheme
        parameters = model.parameters
        self._water_density = parameters.water_density
        self._rotation_factor = parameters.rotation_factor
        self._axial_rotation_factor = parameters.axial_rotation_factor
        self._uniform_rotation_factor = parameters.uniform_rotation_factor
        # the library's internal frame is one unit system (see
        # docs/source/units.rst): the material must live in the sea
        # level model's scales, so every conversion factor below is
        # exactly one.  The factors are kept written out all the same:
        # they are the dimensional structure of the formulas, and a
        # factor is never dropped because the current scheme makes it
        # one.
        material_scales = operator.material.scales
        sle_scales = parameters.scales
        if not all(
            np.isclose(getattr(material_scales, name), getattr(sle_scales, name))
            for name in ("length", "mass", "time")
        ):
            raise ValueError(
                "the material and the earth model live in different unit "
                "systems; convert the planetmodel model into the model's "
                "frame (model.converted(state.model.parameters.scales)) "
                "before building its mesh and material"
            )

        def ratio(dims):
            return material_scales.factor(dims) / sle_scales.factor(dims)

        self._to_nd_length = ratio(units.LENGTH)
        self._to_nd_potential = ratio(POTENTIAL)
        self._to_nd_moment = ratio(MOMENT)
        self._to_mat_density = 1.0 / ratio(SURFACE_DENSITY)
        self._to_mat_potential = 1.0 / ratio(POTENTIAL)
        self._to_mat_time = 1.0 / ratio(units.TIME)
        self._sle: dict[tuple[str, float], SeaLevelEquation] = {}
        self._fluid_margin: float | None = None

    # -- plumbing ----------------------------------------------------------

    def _sle_of(self, table) -> SeaLevelEquation:
        return SeaLevelEquation(
            EarthModel(
                self.model.lmax,
                parameters=self.model.parameters,
                love_numbers=table,
                grid=self.model.grid_name,
                extend=self.model.extend,
            )
        )

    def _sle_for(self, dt_material: float) -> SeaLevelEquation:
        """The sea level solver of the step's effective Love numbers,
        cached per (scheme, dt)."""
        key = (self.scheme, float(dt_material))
        if key not in self._sle:
            self._sle[key] = self._sle_of(
                self.operator.effective_love_numbers(self.scheme, dt_material)
            )
        return self._sle[key]

    def _sle_elastic(self) -> SeaLevelEquation:
        """The solver of the unrelaxed numbers, for the instantaneous
        response at the start time."""
        key = ("unrelaxed", 0.0)
        if key not in self._sle:
            self._sle[key] = self._sle_of(self.operator.elastic_love_numbers())
        return self._sle[key]

    def _assert_transverse_stable(self) -> None:
        """Refuses a transverse feedback the specified moments cannot
        sustain.  The equilibrium moments of the rotational feedback are
        specified, never computed from the spherical model (Yu et al.
        2025); the evolution is stable only when the oblateness they
        sustain exceeds the model's hydrostatic fluid limit, since over
        long steps the effective tidal number approaches that limit.
        The margin is 1 - k_hyd / k_sustained: for Earth-like moments
        and a model with an elastic lithosphere it is positive; the
        `with_hydrostatic_moments` construction makes it delta / (1 +
        delta) for any rheology."""
        if self._fluid_margin is None:
            k_hyd = fluid_tidal_number(self.operator.material, 2, eps=self.operator.eps)
            parameters = self.model.parameters
            self._fluid_margin = (
                1.0 - parameters.inertia_factor * parameters.rotation_factor * k_hyd
            )
        if self._fluid_margin <= 0.0:
            raise ValueError(
                "the transverse rotational feedback is unstable: the specified "
                "moments of inertia sustain less oblateness than the model's "
                f"hydrostatic fluid limit (margin {self._fluid_margin:.3e}).  "
                "Specify consistent moments -- e.g. parameters."
                "with_hydrostatic_moments(fluid_tidal_number(material)) -- or "
                "pass rotational_feedbacks='axial' or False."
            )

    def _columns(self, coeffs: np.ndarray, factor: float):
        """A degree-column callable over a coefficient array: degree l
        maps to its 2 (l + 1) columns, cosine block then sine block."""

        def channel(l: int, t: float) -> np.ndarray:
            return factor * np.concatenate(
                (coeffs[0, l, : l + 1], coeffs[1, l, : l + 1])
            )

        return channel

    def _coeffs(self, per_degree: dict[int, np.ndarray], factor: float) -> SHCoeffs:
        """The inverse packing: per-degree column arrays to SHCoeffs."""
        lmax = self.model.lmax
        out = np.zeros((2, lmax + 1, lmax + 1))
        for l, cols in per_degree.items():
            out[0, l, : l + 1] = cols[: l + 1]
            out[1, l, : l + 1] = cols[l + 1 :]
        return SHCoeffs.from_array(
            factor * out,
            normalization=self.model.normalization,
            csphase=self.model.csphase,
        )

    def _centrifugal(self, omega: np.ndarray) -> np.ndarray:
        """The centrifugal potential coefficients of an angular velocity
        change: degree 2 orders 0 and 1, and the degree-0 coefficient
        of the (r/a)^2 mean that the l = 0 tide channel carries."""
        psi = np.zeros((2, self.model.lmax + 1, self.model.lmax + 1))
        psi[:, 2, 1] = self._rotation_factor * omega[:2]
        psi[0, 2, 0] = self._axial_rotation_factor * omega[2]
        psi[0, 0, 0] = self._uniform_rotation_factor * omega[2]
        return psi

    # -- the loop ------------------------------------------------------------

    def run(
        self,
        times: Sequence[float],
        ice_load: Callable[[float], SHGrid],
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        verbose: bool = False,
        record: bool | int = False,
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """Integrates from times[0] over the given grid, which should
        include any breakpoints of the ice history; `ice_load(t)` is
        the non-ocean surface load (grounded ice, sediment) at time t,
        an SHGrid in the model's units, continuous in t for t >
        times[0].  The start is the instantaneous elastic response to
        `ice_load(times[0])` -- the right limit, so a load switched on
        at the start is stepped at full order.  Returns the surface
        fields at every time, the first included, and the final
        internal state, whose time is in the MATERIAL's units.

        `rotational_feedbacks` is True for the full feedback, "axial"
        for the axial component alone, or False.  The full feedback
        requires moments of inertia sustaining more oblateness than the
        model's hydrostatic fluid limit -- specified observed values
        for an Earth-like body, or the construction of
        `EarthModelParameters.with_hydrostatic_moments` for any other
        -- and is refused otherwise.

        `record` keeps what a viscosity-kernel adjoint needs of the
        trajectory: True attaches every node's internal state to its
        snapshot; an integer stride keeps the states every that many
        nodes (and at the last), with the per-node load and centrifugal
        coefficients always kept so a reverse sweep reconstructs the
        gaps by exact backward steps.  The reconstruction amplifies
        roundoff by exp(h) per branch and node across a gap, so the
        stride spends trailing digits for memory; the accumulated
        h = dt / tau between checkpoints is the budget."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be increasing, with at least two entries")
        stride = _record_stride(record)
        last_node = times.size - 1
        op = self.operator
        ocean = self.state.ocean_function.data

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        vstate.t = times[0] * self._to_mat_time

        if rotational_feedbacks is True:
            self._assert_transverse_stable()

        load_grid = ice_load(times[0])
        slc, disp, potential, omega = self._sle_elastic().solve_generalised_equation(
            self.state,
            direct_load=load_grid,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
        )
        total = load_grid.data + self._water_density * ocean * slc.data
        sigma = self.model.expand_field(
            SHGrid.from_array(total, grid=self.model.grid, copy=False)
        ).coeffs
        psi = self._centrifugal(omega) if rotational_feedbacks else np.zeros_like(sigma)
        if stride:
            op.prime_strain(
                vstate,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
        snapshots = [
            SeaLevelSnapshot(
                float(times[0]),
                slc,
                disp,
                potential,
                omega,
                internal_state=vstate if stride else None,
                load_coefficients=sigma if stride else None,
                tide_coefficients=psi if stride else None,
            )
        ]

        for node, (t0, t1) in enumerate(zip(times[:-1], times[1:]), start=1):
            dt = (t1 - t0) * self._to_mat_time
            step = op.start_step(
                vstate,
                dt,
                scheme=self.scheme,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            sle = self._sle_for(dt)
            memory_u = self._coeffs(step.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(step.memory_surface("phi"), self._to_nd_potential)
            moment = step.inertia_moment()[0] * self._to_nd_moment
            load_grid = ice_load(t1)
            slc, disp, potential, omega = sle.solve_generalised_equation(
                self.state,
                direct_load=load_grid,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
            )
            total = load_grid.data + self._water_density * ocean * slc.data
            sigma = self.model.expand_field(
                SHGrid.from_array(total, grid=self.model.grid, copy=False)
            ).coeffs
            psi = (
                self._centrifugal(omega)
                if rotational_feedbacks
                else np.zeros_like(sigma)
            )
            vstate, _ = step.finish(
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            keep = bool(stride) and (node % stride == 0 or node == last_node)
            snapshots.append(
                SeaLevelSnapshot(
                    float(t1),
                    slc,
                    disp,
                    potential,
                    omega,
                    internal_state=vstate if keep else None,
                    load_coefficients=sigma if stride else None,
                    tide_coefficients=psi if stride else None,
                )
            )
        return snapshots, vstate

    def run_adaptive(
        self,
        t_start: float,
        t_final: float,
        ice_load: Callable[[float], SHGrid],
        *,
        dt_initial: float,
        rtol_step: float = 1e-4,
        atol_step: float | None = None,
        quantise: bool = True,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_steps: int = 10000,
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """The coupled loop under the adaptive exponential trapezoid.

        As `run`, but the step size follows the embedded ETD1 error
        estimate of the radial state: small through transients, striding
        when the response is quiet.  Each accepted step emits a
        snapshot at its landing time.  Every fresh step size costs new
        effective systems, a Love-number table and a sea level solver,
        so by default the proposed steps snap DOWN onto the ladder
        dt_initial 2^n (`quantise`), holding those costs to a handful
        of flavours; the estimate is the first-order companion's and
        hence conservative.  Times are in the Earth model's units;
        `ice_load(t)` must be continuous.
        """
        if t_final <= t_start:
            raise ValueError("t_final must lie beyond t_start")
        if dt_initial <= 0.0:
            raise ValueError("dt_initial must be positive")
        op = self.operator
        model = self.model
        parameters = model.parameters
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        ocean = self.state.ocean_function.data
        water_density = parameters.water_density
        scheme = self.scheme
        if scheme != "exponential_trapezoid":
            raise ValueError("the adaptive loop is the exponential trapezoid's")

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        vstate.t = t_start * self._to_mat_time

        # the right limit at the start, as in `run`
        load_grid = ice_load(t_start)
        start_sle = self._sle_elastic()
        slc, disp, potential, omega = start_sle.solve_generalised_equation(
            self.state,
            direct_load=load_grid,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
        total = load_grid.data + water_density * ocean * slc.data
        sigma = model.expand_field(
            SHGrid.from_array(total, grid=model.grid, copy=False)
        ).coeffs
        psi = self._centrifugal(omega) if rotational_feedbacks else np.zeros_like(sigma)
        snapshots = [SeaLevelSnapshot(float(t_start), slc, disp, potential, omega)]

        def load_cb(coeffs):
            return self._columns(coeffs, self._to_mat_density)

        def tide_cb(coeffs):
            return self._columns(coeffs, self._to_mat_potential)

        dt = dt_initial * self._to_mat_time
        base = dt
        accepted = 0
        same_rung = 0
        for _ in range(max_steps):
            t_now = vstate.t / self._to_mat_time
            remaining = (t_final - t_now) * self._to_mat_time
            if remaining <= 1e-12 * self._to_mat_time * max(abs(t_final), 1.0):
                break
            trial = min(dt, remaining)
            d_start = op.prime_strain(vstate, load=load_cb(sigma), tide=tide_cb(psi))
            step = op.start_step(
                vstate,
                trial,
                scheme=scheme,
                load=load_cb(sigma),
                tide=tide_cb(psi),
            )
            sle = self._sle_for(trial)
            memory_u = self._coeffs(step.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(step.memory_surface("phi"), self._to_nd_potential)
            moment = step.inertia_moment()[0] * self._to_nd_moment
            t1 = t_now + trial / self._to_mat_time
            grid1 = ice_load(t1)
            slc, disp, potential, omega = sle.solve_generalised_equation(
                self.state,
                direct_load=grid1,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            total = grid1.data + water_density * ocean * slc.data
            sigma_new = model.expand_field(
                SHGrid.from_array(total, grid=model.grid, copy=False)
            ).coeffs
            psi_new = (
                self._centrifugal(omega)
                if rotational_feedbacks
                else np.zeros_like(sigma_new)
            )
            candidate, _ = step.finish(load=load_cb(sigma_new), tide=tide_cb(psi_new))
            # the embedded ETD1 estimate on the radial state
            weights = op.scheme_weights(scheme, trial)
            if atol_step is None:
                top = max(
                    (float(np.max(np.abs(candidate.m[l]))) for l in vstate.degrees),
                    default=0.0,
                )
                floor = 1e-6 * rtol_step * top if top > 0.0 else 1e-300
            else:
                floor = atol_step
            total_sq = 0.0
            count = 0
            for l in vstate.degrees:
                deg = op.degree(l)
                for k, (e, _, _) in enumerate(weights):
                    ek = e[deg.e0 :][None, :, :, None]
                    companion = ek * vstate.m[l][k] + (1.0 - ek) * d_start[l]
                    scale = floor + rtol_step * np.maximum(
                        np.abs(vstate.m[l][k]), np.abs(candidate.m[l][k])
                    )
                    ratio = (candidate.m[l][k] - companion) / scale
                    total_sq += float(np.sum(ratio * ratio))
                    count += ratio.size
            err = np.sqrt(total_sq / count) if count else 0.0
            factor = 4.0 if err == 0.0 else 0.9 / np.sqrt(err)
            dt_next = trial * float(np.clip(factor, 0.2, 4.0))
            raw = dt_next
            if quantise and dt_next < remaining:
                # floor for ladder stability, with a stagnation promoter
                # climbing a rung after eight healthy accepts
                dt_next = base * 2.0 ** np.floor(np.log2(dt_next / base))
            if err > 1.0:
                same_rung = 0
                dt = dt_next
                continue
            vstate = candidate
            sigma = sigma_new
            psi = psi_new
            accepted += 1
            snapshots.append(SeaLevelSnapshot(float(t1), slc, disp, potential, omega))
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
        return snapshots, vstate

    def run_ice_history(
        self,
        dates: Sequence[float],
        ice_model: BaseIceModel,
        *,
        shorelines: str = "frozen",
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        verbose: bool = False,
        record: bool | int = False,
        final_sea_level: Optional[SHGrid] = None,
        calibration: str = "fixed_point",
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """Integrates an ice history given by any ice model.

        `dates` are in kiloyears before present, decreasing (the past
        first), and should include the history's breakpoints; the model
        interpolates between them, per the `BaseIceModel` contract.
        The snapshots carry their dates.

        With `shorelines="frozen"` the equation is linear: the direct
        load is the grounded ice change since the first date, grounded
        meaning outside the background state's fixed ocean function.
        With `shorelines="migrating"` each step solves the nonlinear
        equation instead: the ocean function follows the flotation
        condition, grounding and mass conservation are re-evaluated as
        the shorelines move, and every snapshot records its ocean
        function -- the C(t) history an adjoint calculation reads
        back.  The background state must then be the state at the
        first date, whose ice thickness is the history's.

        `record` keeps the internal states for a viscosity-kernel
        adjoint, every node or on a checkpoint stride; see `run`.

        Given `final_sea_level`, the sea level at the last date, the
        initial sea level is first calibrated so that the run reaches
        it, by `calibrate_initial_sea_level` with the method named by
        `calibration` ("fixed_point" or "gradient") and its defaults
        otherwise; the background state's sea level is the starting
        guess, and the evolution adopts the calibrated state.  The run
        returned is the one from that state.
        """
        if shorelines not in ("frozen", "migrating"):
            raise ValueError(
                f"shorelines must be 'frozen' or 'migrating', got {shorelines!r}"
            )
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the "
                "past first), with at least two entries"
            )
        parameters = self.model.parameters
        if not np.isclose(ice_model.length_scale, parameters.length_scale):
            raise ValueError(
                "the ice model's length scale differs from the Earth model's"
            )
        if final_sea_level is not None:
            calibrated = self.calibrate_initial_sea_level(
                dates,
                ice_model,
                final_sea_level,
                shorelines=shorelines,
                method=calibration,
                rotational_feedbacks=rotational_feedbacks,
                solver_rtol=rtol,
                verbose=verbose,
            )
            if not record:
                return calibrated.snapshots, calibrated.internal_state
            # the calibration's runs keep no trajectory, so a recording
            # run is repeated from the calibrated state
            return self.run_ice_history(
                dates,
                ice_model,
                shorelines=shorelines,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
                record=record,
            )
        if shorelines == "migrating":
            return self._run_migrating(
                dates,
                ice_model,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
                record=record,
            )
        kyr = KYR / parameters.scales.factor(units.TIME)
        date0 = float(dates[0])
        grounded = 1.0 - self.state.ocean_function.data
        rho_ice = parameters.ice_density

        def thickness(date: float) -> np.ndarray:
            return ice_model.get_ice_thickness(
                date,
                self.model.lmax,
                grid=self.model.grid_name,
                sampling=self.model.sampling,
                extend=self.model.extend,
            ).data

        first = thickness(date0)

        def ice_load(t: float) -> SHGrid:
            change = thickness(date0 - t / kyr) - first
            return SHGrid.from_array(
                rho_ice * grounded * change, grid=self.model.grid, copy=False
            )

        snapshots, vstate = self.run(
            (date0 - dates) * kyr,
            ice_load,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
            record=record,
        )
        snapshots = [replace(snap, date=date0 - snap.t / kyr) for snap in snapshots]
        return snapshots, vstate

    def _thickness(self, ice_model: BaseIceModel, date: float) -> np.ndarray:
        return ice_model.get_ice_thickness(
            date,
            self.model.lmax,
            grid=self.model.grid_name,
            sampling=self.model.sampling,
            extend=self.model.extend,
        ).data

    def _total_load_coeffs(self, final_state: EarthState) -> np.ndarray:
        """The surface load the nonlinear solve's final state carries,
        relative to the background: grounded ice and ocean water, the
        construction of the solver's own iteration, so the radial step
        is finished under the load its sea level solution saw."""
        parameters = self.model.parameters
        start = self.state
        grounded = parameters.ice_density * (
            (1.0 - final_state.ocean_function.data) * final_state.ice_thickness.data
            - (1.0 - start.ocean_function.data) * start.ice_thickness.data
        )
        ocean = parameters.water_density * (
            final_state.ocean_function.data * final_state.sea_level.data
            - start.ocean_function.data * start.sea_level.data
        )
        return self.model.expand_field(
            SHGrid.from_array(grounded + ocean, grid=self.model.grid, copy=False)
        ).coeffs

    def _run_migrating(
        self, dates, ice_model, *, rotational_feedbacks, rtol, verbose, record=False
    ):
        """The migrating-shoreline loop: each step's equilibrium is the
        nonlinear equation, with the memory field through the response
        channels, and the converged state's load finishes the radial
        step.  The solutions and the state are consistent to the
        nonlinear iteration's tolerance (the fixed point is no longer
        eliminated exactly, as it is in the linear path)."""
        op = self.operator
        model = self.model
        parameters = model.parameters
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        kyr = KYR / parameters.scales.factor(units.TIME)
        date0 = float(dates[0])

        start_ice = self._thickness(ice_model, date0)
        scale = np.max(np.abs(start_ice))
        if np.max(np.abs(self.state.ice_thickness.data - start_ice)) > 1e-6 * scale:
            raise ValueError(
                "the background state's ice thickness is not the history's "
                "at the first date; build the state at dates[0]"
            )

        stride = _record_stride(record)
        last_node = len(dates) - 1
        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        sigma = np.zeros((2, model.lmax + 1, model.lmax + 1))
        psi = np.zeros_like(sigma)
        if stride:
            # the start is the undisturbed background, so the cache is
            # the zero solve's
            op.prime_strain(
                vstate,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
        zero = model.zero_grid()
        snapshots = [
            SeaLevelSnapshot(
                0.0,
                zero,
                zero,
                zero,
                np.zeros(3),
                date=date0,
                ocean_function=self.state.ocean_function,
                internal_state=vstate if stride else None,
                load_coefficients=sigma if stride else None,
                tide_coefficients=psi if stride else None,
            )
        ]
        for node, (a, b) in enumerate(zip(dates[:-1], dates[1:]), start=1):
            # the step grid is in dates; the operator works in the
            # material's units
            dt = (a - b) * kyr * self._to_mat_time
            step = op.start_step(
                vstate,
                dt,
                scheme=self.scheme,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            sle = self._sle_for(dt)
            memory_u = self._coeffs(step.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(step.memory_surface("phi"), self._to_nd_potential)
            moment = step.inertia_moment()[0] * self._to_nd_moment
            change = SHGrid.from_array(
                self._thickness(ice_model, float(b)) - start_ice,
                grid=model.grid,
                copy=False,
            )
            final_state, slc, disp, potential, omega = sle.solve_nonlinear_equation(
                self.state,
                ice_thickness_change=change,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
            )
            sigma = self._total_load_coeffs(final_state)
            psi = (
                self._centrifugal(omega)
                if rotational_feedbacks
                else np.zeros_like(sigma)
            )
            vstate, _ = step.finish(
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            keep = bool(stride) and (node % stride == 0 or node == last_node)
            snapshots.append(
                SeaLevelSnapshot(
                    float(vstate.t / self._to_mat_time),
                    slc,
                    disp,
                    potential,
                    omega,
                    date=float(b),
                    ocean_function=final_state.ocean_function,
                    internal_state=vstate if keep else None,
                    load_coefficients=sigma if stride else None,
                    tide_coefficients=psi if stride else None,
                )
            )
        return snapshots, vstate

    # -- the initial sea level ------------------------------------------------

    def _with_state(self, state: EarthState) -> SeaLevelEvolution:
        """The same evolution over another background state.  The
        caches of effective Love numbers and solvers depend on the
        operator and the Earth model only, so they are shared."""
        if state.model is not self.model:
            raise ValueError("the state must use the evolution's Earth model")
        sibling = copy.copy(self)
        sibling.state = state
        return sibling

    def final_sea_level_misfit(
        self,
        dates: Sequence[float],
        ice_model: BaseIceModel,
        final_sea_level: SHGrid,
        *,
        shorelines: str = "frozen",
        weights: Optional[SHGrid] = None,
        gradient: bool = True,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        verbose: bool = False,
    ) -> FinalSeaLevelMisfit:
        """The misfit of the run's final sea level to a target, and its
        gradient with respect to the initial sea level.

        The background state is the initial state, so its sea level is
        S0 and, under migrating shorelines, its ice thickness must be
        the history's at the first date.  The objective is

            J = (1/2) int w (S0 + xi(t_f) - S_f)^2 dS,

        with S_f the target at the last date and w the `weights` (one
        everywhere by default).  Its gradient, following Yu et al.
        (2025, eq. 87), is the weighted residual plus the initial sea
        level kernel of a sea-level measurement of that residual at the
        last date: the initial sea level enters the absolute sea level
        directly, and the load at each date through rho_w (C_n - C_0)
        S0.  Under frozen shorelines the ocean function never changes,
        the kernel vanishes identically and no adjoint sweep is run;
        the gradient is then the weighted residual, and a unit step
        down it is the classical update.  Under migrating shorelines
        the kernel costs one adjoint sweep, about the cost of the
        forward run.  The gradient is exact away from changes of the
        initial ocean function itself, where J is not differentiable.

        Args:
            dates: Kiloyears before present, decreasing; the last is
                the date of the target.
            ice_model: The ice history.
            final_sea_level: The target sea level at the last date.
            shorelines: "frozen" or "migrating", as for
                `run_ice_history`.
            weights: Non-negative weights on the residual, an SHGrid;
                None weights every point equally.
            gradient: Whether to compute the gradient.
            rotational_feedbacks, rtol, verbose: As for
                `run_ice_history`.

        Returns:
            The misfit, its residual and gradient, and the forward run.
        """
        model = self.model
        model.check_field(final_sea_level)
        if weights is not None:
            model.check_field(weights)
            if np.any(weights.data < 0.0):
                raise ValueError("the weights must be non-negative")
        snapshots, vstate = self.run_ice_history(
            dates,
            ice_model,
            shorelines=shorelines,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
        )
        residual = (
            self.state.sea_level.data
            + snapshots[-1].sea_level_change.data
            - final_sea_level.data
        )
        w = 1.0 if weights is None else weights.data
        value = 0.5 * model.integrate(
            SHGrid.from_array(w * residual**2, grid=model.grid, copy=False)
        )
        grad = None
        if gradient:
            g = w * residual
            if shorelines == "migrating":
                kernels = self.adjoint_ice_history(
                    dates,
                    {float(dates[-1]): SHGrid.from_array(g, grid=model.grid)},
                    ocean_functions=[snap.ocean_function for snap in snapshots],
                    rotational_feedbacks=rotational_feedbacks,
                    rtol=rtol,
                )
                g = g + kernels.initial_sea_level.data
            grad = SHGrid.from_array(g, grid=model.grid)
        return FinalSeaLevelMisfit(
            float(value),
            SHGrid.from_array(residual, grid=model.grid),
            grad,
            snapshots,
            vstate,
        )

    def calibrate_initial_sea_level(
        self,
        dates: Sequence[float],
        ice_model: BaseIceModel,
        final_sea_level: SHGrid,
        *,
        shorelines: str = "frozen",
        method: str = "fixed_point",
        weights: Optional[SHGrid] = None,
        step: float = 1.0,
        iterations: int = 10,
        rtol: float = 1e-4,
        anderson_memory: int = 5,
        rotational_feedbacks: bool = True,
        solver_rtol: float = 1e-9,
        verbose: bool = False,
    ) -> InitialSeaLevelCalibration:
        """Calibrates the initial sea level so that the run reaches a
        given final sea level, and adopts the calibrated state.

        A thickness-only ice history leaves the initial sea level S0 to
        be found, and the standard device is to choose it so that the
        run reproduces the observed present-day sea level S_f.  Writing
        the final sea level as S0 + xi(t_f; S0), two methods are
        offered:

        - "fixed_point": the classical update S0 <- S0 - r, with r the
          residual S0 + xi(t_f) - S_f; that is, S0 <- S_f - xi(t_f).
          It is Newton's method for S0 + xi(t_f; S0) = S_f with the
          Jacobian I + d xi / d S0 replaced by the identity, which is
          exact under frozen shorelines, where xi depends on S0 only
          through the initial ocean function; it converges at the rate
          set by the dependence of xi on S0 through the migrating
          shorelines.  With `weights` it updates only where they are
          positive.
        - "gradient": steepest descent on the weighted misfit of
          `final_sea_level_misfit`, S0 <- S0 - step grad J, with the
          gradient from the adjoint (Yu et al. 2025, Section 3.9).
          Under frozen shorelines with unit weights and step it is the
          classical update exactly.  It is the method for an objective
          the classical update does not serve: a weighted or partial
          comparison with the present.

        Either update is accelerated by Anderson mixing, a multisecant
        quasi-Newton method built from the iterates.  The iteration
        stops when the largest update falls below `rtol` times the
        largest value of S0.

        The evolution's background state supplies the starting guess
        and the Caspian option; its ice thickness is replaced by the
        history's at the first date.  On return the evolution's state
        is the calibrated one, so that later runs and adjoints use it.

        Args:
            dates: Kiloyears before present, decreasing; the last is
                the date of the target.
            ice_model: The ice history; its thickness alone is used.
            final_sea_level: The target sea level at the last date, an
                SHGrid in the Earth model's units.
            shorelines: "frozen" or "migrating", as for
                `run_ice_history`.
            method: "fixed_point" or "gradient".
            weights: Non-negative weights on the misfit, an SHGrid; None
                weights every point equally.
            step: The step length of either update; one is the natural
                value.
            iterations: The most forward runs taken.
            rtol: The convergence tolerance on the update.
            anderson_memory: The depth of the acceleration; zero gives
                the plain iteration.
            rotational_feedbacks: As for `run_ice_history`.
            solver_rtol: The tolerance of the sea level solves.
            verbose: Prints each iteration's misfit and update.

        Returns:
            The calibration, with the calibrated state and its run.
        """
        if method not in ("fixed_point", "gradient"):
            raise ValueError(
                f"method must be 'fixed_point' or 'gradient', got {method!r}"
            )
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        model = self.model
        dates = np.asarray(dates, dtype=float)
        ice_start = ice_model.get_ice_thickness(
            float(dates[0]),
            model.lmax,
            grid=model.grid_name,
            sampling=model.sampling,
            extend=model.extend,
        )
        if weights is None:
            w = np.ones_like(final_sea_level.data)
        else:
            model.check_field(weights)
            w = weights.data
        support = (w > 0.0).astype(float)
        total_weight = model.integrate(SHGrid.from_array(w, grid=model.grid))

        mixer = AndersonMixer(anderson_memory)
        guess = self.state.sea_level.data.copy()
        misfits: list[float] = []
        updates: list[float] = []
        converged = False
        for n in range(iterations):
            state = EarthState(
                ice_start,
                SHGrid.from_array(guess, grid=model.grid),
                model,
                exclude_caspian=self.state.exclude_caspian,
            )
            misfit = self._with_state(state).final_sea_level_misfit(
                dates,
                ice_model,
                final_sea_level,
                shorelines=shorelines,
                weights=weights,
                gradient=method == "gradient",
                rotational_feedbacks=rotational_feedbacks,
                rtol=solver_rtol,
            )
            misfits.append(float(np.sqrt(2.0 * misfit.value / total_weight)))
            if method == "fixed_point":
                direction = support * misfit.residual.data
            else:
                direction = misfit.gradient.data
            updated = mixer.update(guess, guess - step * direction)
            change = float(np.max(np.abs(updated - guess)))
            scale = float(np.max(np.abs(updated)))
            updates.append(change / scale)
            if verbose:
                print(
                    f"calibration iteration {n + 1}: misfit {misfits[-1]:.3e}, "
                    f"relative update {updates[-1]:.3e}"
                )
            if change <= rtol * scale:
                converged = True
                break
            guess = updated
        if not converged:
            warnings.warn(
                f"the initial sea level calibration did not converge in "
                f"{iterations} iterations (relative update {updates[-1]:.3e}, "
                f"tolerance {rtol:.1e}); the last state run is returned",
                RuntimeWarning,
                stacklevel=2,
            )
        self.state = state
        return InitialSeaLevelCalibration(
            state,
            misfit.snapshots,
            misfit.internal_state,
            misfits,
            updates,
            converged,
            method,
            shorelines,
        )

    def _as_trajectory(self, internal_states, dates) -> ForwardTrajectory:
        """Normalises the recorded forward trajectory an adjoint was
        given: a ForwardTrajectory, a recording run's snapshots, or a
        plain sequence of states when every node was kept."""
        if isinstance(internal_states, ForwardTrajectory):
            trajectory = internal_states
        else:
            seq = list(internal_states)
            if seq and isinstance(seq[0], SeaLevelSnapshot):
                trajectory = ForwardTrajectory.from_snapshots(seq)
            else:
                trajectory = ForwardTrajectory(seq)
        states = trajectory.states
        if len(states) != dates.size:
            raise ValueError("internal_states must give one entry per date node")
        last = states[-1]
        if last is None or last.cached_strain is None:
            raise ValueError(
                "internal_states must be recorded forward states " "(a run with record)"
            )
        if any(st is None for st in states):
            if trajectory.loads is None or trajectory.tides is None:
                raise ValueError(
                    "a checkpointed trajectory needs the recorded load and "
                    "centrifugal coefficients: pass the recording run's "
                    "snapshots or a ForwardTrajectory"
                )
        elif any(st.cached_strain is None for st in states):
            raise ValueError(
                "internal_states must be recorded forward states " "(a run with record)"
            )
        return trajectory

    def continuous_adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid],
        *,
        ocean_functions: Optional[Sequence[SHGrid]] = None,
        internal_states: Optional[Sequence[ViscoelasticState]] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
    ) -> IceKernels:
        """Ice-history kernels by the continuous adjoint, discretised.

        The adjoint equations have the same form as the forward problem
        (Yu et al. 2025, Section 3.3), so between observation times the
        adjoint state is stepped by the SAME machinery as a forward run
        with no load, in reversed time; at each observation time the
        instantaneous (unrelaxed) fingerprint problem supplies the
        delta amplitudes and the adjoint internal variables take the
        finite jump of their eq. (80),

            m-dagger += (1 / tau_k) d-tilde-dagger,

        with d-tilde-dagger the strain of the instantaneous adjoint
        solution.  Unlike `adjoint_ice_history` (the exact discrete
        transpose) this formulation never sees the time stepper's
        internals, so it survives a change of scheme.  The two differ
        by the discretise/transpose commute error, which vanishes as
        the run resolves its rheology: a discrete adjoint is exactly
        faithful to the discretised functional, which in an
        under-resolved run is a poor stand-in for the continuum one,
        so DISAGREEMENT between the two gradients is a resolution
        diagnostic for the forward calculation itself.  Returns the
        same `IceKernels` views, the node duals assembled from the
        density and the lumps, so the two methods run the same gates.

        With `internal_states` -- the recorded forward trajectory, from
        a run with `record=True` (frozen, or migrating together with
        its `ocean_functions`) -- the kernels also carry the
        continuous sensitivity to the branch inverse relaxation times:
        the multiplier of the internal-variable dynamics is the
        branch-weighted adjoint state over itau, scaled by the ratio of
        the material's load pairing to the model's sea-level pairing
        (see `pairing` below), so

            dJ / d itau(r) = int W_k m-dagger / itau (d - m) dt,

        accumulated over the reversed nodes with the density's split
        trapezoid (the adjoint state jumps at observations; the forward
        integrand is continuous).  As with the ice kernels, the result
        converges to the discrete one as the run resolves its rheology.
        """
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the "
                "past first), with at least two entries"
            )
        duals = _normalise_measurement(measurement, dates, self.model)
        op = self.operator
        model = self.model
        parameters = model.parameters
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        kyr = KYR / parameters.scales.factor(units.TIME)
        C = self.state.ocean_function.data
        water_density = parameters.water_density
        g_surface = parameters.gravitational_acceleration
        r2 = parameters.mean_sea_floor_radius**2
        if ocean_functions is not None:
            if len(ocean_functions) != dates.size:
                raise ValueError("ocean_functions must give one grid per date node")
            for grid in ocean_functions:
                model.check_field(grid)

        def C_at(node: int) -> np.ndarray:
            return ocean_functions[node].data if ocean_functions is not None else C

        def C_grid(node: int) -> Optional[SHGrid]:
            return ocean_functions[node] if ocean_functions is not None else None

        def grounded_at(node: int) -> np.ndarray:
            return parameters.ice_density * (1.0 - C_at(node))

        elastic_sle = self._sle_elastic()

        def psi_star(potential_dual: SHGrid) -> np.ndarray:
            """The adjoint of the centrifugal potential operator: a
            potential dual to angular velocity space (the formula of
            `pyslfp.linear_operators.centrifugal_potential_operator`)."""
            coeffs = model.expand_field(potential_dual).coeffs
            return r2 * np.array(
                [
                    parameters.rotation_factor * coeffs[0, 2, 1],
                    parameters.rotation_factor * coeffs[1, 2, 1],
                    parameters.axial_rotation_factor * coeffs[0, 2, 0]
                    + parameters.uniform_rotation_factor * coeffs[0, 0, 0],
                ]
            )

        def jump(m: Measurement, node: int):
            """The instantaneous adjoint fingerprint of a measurement,
            through the forward channel mapping of the elastic adjoint
            (zeta_d = g_xi, zeta^u = -g_u, zeta^phi = -g g_phi,
            L = -g (g_omega - Psi* g_phi)): its sea level (the delta
            amplitude and the kernel lump), and the converged channel
            coefficients that drive the radial jump."""
            avc = None
            if rotational_feedbacks:
                avc = np.zeros(3)
                if m.angular_velocity is not None:
                    avc = avc - g_surface * np.asarray(m.angular_velocity, float)
                if m.gravity_potential is not None:
                    avc = avc + g_surface * psi_star(m.gravity_potential)
            zeta_u = None if m.displacement is None else -1.0 * m.displacement
            zeta_phi = (
                None
                if m.gravity_potential is None
                else -g_surface * m.gravity_potential
            )
            xi_a, _, _, omega_a = elastic_sle.solve_generalised_equation(
                self.state,
                direct_load=m.sea_level,
                displacement_load=zeta_u,
                gravitational_potential_load=zeta_phi,
                angular_momentum_change=avc,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            direct = m.sea_level.data if m.sea_level is not None else np.zeros_like(C)
            lam = model.expand_field(
                SHGrid.from_array(
                    direct + water_density * C_at(node) * xi_a.data,
                    grid=model.grid,
                    copy=False,
                )
            ).coeffs
            psi_a = (
                self._centrifugal(omega_a)
                if rotational_feedbacks
                else np.zeros_like(lam)
            )
            zu = None if zeta_u is None else model.expand_field(zeta_u).coeffs
            zphi = None if zeta_phi is None else model.expand_field(zeta_phi).coeffs
            return xi_a, lam, psi_a, zu, zphi

        def apply_jump(vstate: ViscoelasticState, lam, psi_a, zu, zphi) -> None:
            """Yu et al. eq. (80): the strain of the instantaneous
            adjoint solution, over the relaxation time, jumps the
            adjoint internal variables; elastic branches do not move.
            The split load channels carry displacement and potential
            measurements into the radial field."""
            lam_cols = self._columns(lam, self._to_mat_density)
            psi_cols = self._columns(psi_a, self._to_mat_potential)
            zu_cols = None if zu is None else self._columns(zu, self._to_mat_density)
            zphi_cols = (
                None if zphi is None else self._columns(zphi, self._to_mat_density)
            )
            for l in vstate.degrees:
                deg = op.degree(l)
                units_lv = op.unit_responses(None, 0.0, l)
                x = (
                    units_lv[:, 0][:, None] * lam_cols(l, 0.0)[None, :]
                    + units_lv[:, 1][:, None] * psi_cols(l, 0.0)[None, :]
                )
                if zu_cols is not None or zphi_cols is not None:
                    split = op.split_unit_responses(None, 0.0, l)
                    if zu_cols is not None:
                        x = x + split[:, 0][:, None] * zu_cols(l, 0.0)[None, :]
                    if zphi_cols is not None:
                        x = x + split[:, 1][:, None] * zphi_cols(l, 0.0)[None, :]
                d_tilde = deg.strain(x)
                for k, itau in enumerate(deg.itau):
                    vstate.m[l][k] += itau[None, :, :, None] * d_tilde
            vstate.invalidate()

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        sigma_c = np.zeros((2, model.lmax + 1, model.lmax + 1))
        psi_c = np.zeros_like(sigma_c)
        N = dates.size - 1
        # the node weights of the date quadrature, split across the
        # observation discontinuities: 'late' towards the present,
        # 'early' towards t0, in kiloyears
        w_early = np.zeros(dates.size)
        w_late = np.zeros(dates.size)
        w_early[1:] = 0.5 * (dates[:-1] - dates[1:])
        w_late[:-1] = 0.5 * (dates[:-1] - dates[1:])
        per_ka = kyr * self._to_mat_time

        itau_kernel = None
        cursor = None
        if internal_states is not None:
            trajectory = self._as_trajectory(internal_states, dates)
            cursor = _TrajectoryCursor(self, trajectory, dates, self.scheme)
            ngll = op.material.mesh.ngll
            K = op.rheology.num_branches
            itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]

        # the radial state the jumps seed is NOT the true multiplier of
        # the internal-variable dynamics: `apply_jump` drives the
        # forward unit-response machinery, whose load pairing lives on
        # the material's surface (g a^2, in the material's units),
        # while the functional pairs the dual with sea level on the
        # model's surface (r^2, in the model's units).  The true
        # multiplier is the state times the ratio of the two pairings,
        # which carries dimensions (mass x length), so it is unity only
        # when the two unit systems agree -- on an SI material its
        # omission inflates the kernels by ~thirty orders of magnitude
        pairing = (
            r2
            * self._to_nd_length
            / (op.material.radius**2 * self._to_mat_density)
            / op.material.surface_gravity
        )

        def accumulate_itau(node: int, weight: float) -> None:
            """One side's contribution at a node: the branch-weighted
            adjoint state, scaled to the true multiplier of the
            internal-variable dynamics by `pairing`, over itau, paired
            with the forward d - m."""
            if itau_kernel is None or weight == 0.0:
                return
            fwd = cursor.state(node)
            for l in vstate.degrees:
                deg = op.degree(l)
                dm = fwd.cached_strain[l][None] - fwd.m[l]
                weighted = deg.apply_weights(vstate.m[l])
                for k in range(K):
                    itau_k = deg.itau[k]
                    inv = np.where(itau_k > 0.0, itau_k, 1.0)
                    g = (weighted[k] * dm[k]).sum(axis=(0, 3))
                    g = np.where(itau_k > 0.0, pairing * g / inv, 0.0)
                    itau_kernel[k][deg.e0 :] += weight * g

        lump_load: dict[int, np.ndarray] = {}
        instantaneous: dict[float, SHGrid] = {}
        # the density is discontinuous at each observation (a finite
        # jump), so both one-sided values are kept: 'late' is the side
        # towards the present, 'early' the side towards t0
        density_late = [np.zeros_like(C) for _ in range(dates.size)]
        density_early = [None] * dates.size

        def smooth_equilibrium(node: int):
            """The memory-only adjoint equilibrium of the current state:
            its sea level (the smooth kernel density) and the load and
            centrifugal coefficients the next reversed step starts
            from.  Between steps the instantaneous response is the
            unrelaxed one, with the node's shorelines."""
            solutions = op.solve_elastic(vstate)
            memory_u = self._coeffs(op.surface(solutions, "U"), self._to_nd_length)
            memory_phi = self._coeffs(
                op.surface(solutions, "phi"), self._to_nd_potential
            )
            moment = (
                float(op.degree(0).moment_row @ solutions[0][:, 0]) * self._to_nd_moment
            )
            xi, _, _, omega = elastic_sle.solve_generalised_equation(
                self.state,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            sig = model.expand_field(
                SHGrid.from_array(
                    water_density * C_at(node) * xi.data, grid=model.grid, copy=False
                )
            ).coeffs
            psi_new = (
                self._centrifugal(omega) if rotational_feedbacks else np.zeros_like(sig)
            )
            return xi.data, sig, psi_new

        def observe(node: int) -> None:
            """Any measurement jump at a reversed node; the smooth
            density is then re-read on the side the sweep continues
            from, since the measurement's legacy lives entirely in the
            jumped internal variables, not in a persistent load."""
            nonlocal sigma_c, psi_c
            if node in duals:
                xi_a, lam, psi_a, zu, zphi = jump(duals[node], node)
                lump_load[node] = xi_a.data
                instantaneous[float(dates[node])] = SHGrid.from_array(
                    grounded_at(node) * xi_a.data, grid=model.grid
                )
                apply_jump(vstate, lam, psi_a, zu, zphi)
                early, sigma_c, psi_c = smooth_equilibrium(node)
                if node > 0:
                    density_early[node] = early

        # the reversed sweep, nodes N (the latest date) down to 0; the
        # state starts undisturbed, so an unmeasured terminal node has
        # zero density
        observe(N)
        accumulate_itau(N, per_ka * w_early[N])
        for n in range(N, 0, -1):
            dt = (dates[n - 1] - dates[n]) * kyr * self._to_mat_time
            step = op.start_step(
                vstate,
                dt,
                scheme=self.scheme,
                load=self._columns(sigma_c, self._to_mat_density),
                tide=self._columns(psi_c, self._to_mat_potential),
            )
            sle = self._sle_for(dt)
            memory_u = self._coeffs(step.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(step.memory_surface("phi"), self._to_nd_potential)
            moment = step.inertia_moment()[0] * self._to_nd_moment
            xi, _, _, omega = sle.solve_generalised_equation(
                self.state,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                ocean_function=C_grid(n - 1),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            density_late[n - 1] = xi.data
            sigma_c = model.expand_field(
                SHGrid.from_array(
                    water_density * C_at(n - 1) * xi.data, grid=model.grid, copy=False
                )
            ).coeffs
            psi_c = (
                self._centrifugal(omega)
                if rotational_feedbacks
                else np.zeros_like(sigma_c)
            )
            vstate, _ = step.finish(
                load=self._columns(sigma_c, self._to_mat_density),
                tide=self._columns(psi_c, self._to_mat_potential),
            )
            accumulate_itau(n - 1, per_ka * w_late[n - 1])
            observe(n - 1)
            accumulate_itau(n - 1, per_ka * w_early[n - 1])

        # assemble the discrete-equivalent node duals from the density
        # and the lumps, so every gate runs on both methods alike.  The
        # jumps divide by relaxation times in the MATERIAL's units, so
        # the smooth fields are densities per material time; the date
        # quadrature converts them to per kiloyear, and at an
        # observation node the weight splits across the discontinuity:
        # the half-interval towards the present pairs with the late
        # side, the half-interval towards t0 with the early side
        density_late = [d * per_ka for d in density_late]
        density_early = [None if d is None else d * per_ka for d in density_early]
        load_duals = []
        density_load = []
        for n in range(dates.size):
            early = (
                density_early[n] if density_early[n] is not None else density_late[n]
            )
            load_duals.append(
                w_early[n] * early + w_late[n] * density_late[n] + lump_load.get(n, 0.0)
            )
            total_w = w_early[n] + w_late[n]
            density_load.append(
                (w_early[n] * early + w_late[n] * density_late[n]) / total_w
            )
        raw = [grounded_at(n) * load_duals[n] for n in range(dates.size)]
        baseline = -parameters.ice_density * (1.0 - C) * sum(load_duals)
        thickness = list(raw)
        thickness[0] = thickness[0] + baseline
        initial_sea_level = water_density * sum(
            (C_at(n) - C) * load_duals[n] for n in range(dates.size)
        )
        rate = []
        tail = np.zeros_like(C)
        for n in range(dates.size - 1, 0, -1):
            tail = tail + raw[n]
            rate.append(tail.copy())
        rate.reverse()
        return IceKernels(
            dates,
            [SHGrid.from_array(ld, grid=model.grid) for ld in load_duals],
            [SHGrid.from_array(t, grid=model.grid) for t in thickness],
            SHGrid.from_array(baseline, grid=model.grid),
            SHGrid.from_array(initial_sea_level, grid=model.grid),
            instantaneous,
            [
                SHGrid.from_array(grounded_at(n) * density_load[n], grid=model.grid)
                for n in range(dates.size)
            ],
            [SHGrid.from_array(r, grid=model.grid) for r in rate],
            itau_kernel,
        )

    def adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid],
        *,
        ocean_functions: Optional[Sequence[SHGrid]] = None,
        internal_states: Optional[Sequence[ViscoelasticState]] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
    ) -> IceKernels:
        """Ice-history sensitivity kernels by the discrete adjoint.

        For the frozen-shoreline forward run over `dates` (an ice
        history enters linearly, so no forward run is needed), and an
        objective J = sum_i int g_i xi(date_i) dS given by the
        `measurement` duals g_i keyed by their dates (which must lie on
        the date grid), this computes the exact transpose of the
        time-stepped forward map: the kernels satisfy

            delta J = sum_n int load_duals[n] delta f_n dS
                    = sum_n int thickness[n] delta I(date_n) dS

        to the solvers' tolerance, for ANY perturbation on the date
        grid.  One reversed sweep: per step, the transposed radial
        update (one symmetric solve per degree) chained with the
        engine's `adjoint_generalised_solve`.

        With `ocean_functions` -- one grid per date node, read off a
        migrating-shoreline forward run's snapshots -- the kernels are
        instead the linearisation about that trajectory: the adjoint
        stays linear with the time-reversed C(t) as its coefficients
        (Yu et al. 2025, Section 3.3; the shoreline-motion singular
        terms cancel), and the grounding factors follow C(t).  The
        gradient identity then holds in the linearised sense.

        A measurement is a sea-level dual grid, or a `Measurement`
        carrying duals of any of the four outputs.

        With `internal_states` -- the recorded forward trajectory, one
        state per date node from a run with `record=True` (frozen, or
        migrating together with its `ocean_functions`), under the same
        scheme and feedbacks -- the kernels also carry the
        exact discrete sensitivity to the branch inverse relaxation
        times, dJ / d itau(r): per step the derivative of the scheme
        weights in h = dt itau pairs the reverse sweep's duals with the
        forward internal variables and strains, and the effective
        moduli's dependence pairs the transposed and forward solutions
        through the branch bilinear.  `viscosity` maps the result to
        dJ / d eta(r).
        """
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the "
                "past first), with at least two entries"
            )
        duals = _normalise_measurement(measurement, dates, self.model)
        op = self.operator
        model = self.model
        parameters = model.parameters
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        kyr = KYR / parameters.scales.factor(units.TIME)
        scheme = self.scheme
        implicit = scheme == "backward_euler"
        C = self.state.ocean_function.data
        if ocean_functions is not None:
            if len(ocean_functions) != dates.size:
                raise ValueError("ocean_functions must give one grid per date node")
            for grid in ocean_functions:
                model.check_field(grid)

        def C_at(node: int) -> np.ndarray:
            return ocean_functions[node].data if ocean_functions is not None else C

        def C_grid(node: int) -> Optional[SHGrid]:
            return ocean_functions[node] if ocean_functions is not None else None

        water_density = parameters.water_density
        r2 = parameters.mean_sea_floor_radius**2
        lmax = model.lmax
        zero_grid = np.zeros_like(C)
        ngll = op.material.mesh.ngll
        K = op.rheology.num_branches

        mdual = {}
        ddual = {}
        for l in op.degrees:
            deg = op.degree(l)
            ncols = 2 * (l + 1)
            mdual[l] = np.zeros((K, 3, deg.ne, ngll, ncols))
            ddual[l] = np.zeros((3, deg.ne, ngll, ncols))

        itau_kernel = None
        cursor = None
        if internal_states is not None:
            trajectory = self._as_trajectory(internal_states, dates)
            cursor = _TrajectoryCursor(self, trajectory, dates, scheme)
            itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]

        def synth(coeffs: np.ndarray) -> np.ndarray:
            return model.expand_coefficient(
                SHCoeffs.from_array(
                    coeffs,
                    normalization=model.normalization,
                    csphase=model.csphase,
                )
            ).data

        def pack(values: dict[int, np.ndarray], factor: float) -> np.ndarray:
            out = np.zeros((2, lmax + 1, lmax + 1))
            for l, cols in values.items():
                out[0, l, : l + 1] = cols[: l + 1]
                out[1, l, : l + 1] = cols[l + 1 :]
            return factor * out

        def sle_dual_inputs(sigma_dual, psi_dual, node):
            """The engine-adjoint inputs at a node: the measurement's
            duals plus those carried by the step's converged load and
            centrifugal coefficients."""
            m = duals.get(node)
            g_xi = (
                m.sea_level.data
                if m is not None and m.sea_level is not None
                else zero_grid
            ) + water_density * C_at(node) * (synth(sigma_dual) / r2)
            g_omega = np.array(
                [
                    parameters.rotation_factor * psi_dual[0, 2, 1],
                    parameters.rotation_factor * psi_dual[1, 2, 1],
                    parameters.axial_rotation_factor * psi_dual[0, 2, 0]
                    + parameters.uniform_rotation_factor * psi_dual[0, 0, 0],
                ]
            )
            if m is not None and m.angular_velocity is not None:
                g_omega = g_omega + np.asarray(m.angular_velocity, dtype=float)
            extra = {}
            if m is not None:
                extra["displacement_dual"] = m.displacement
                extra["potential_dual"] = m.gravity_potential
            return g_xi, g_omega, extra

        load_duals = [None] * dates.size
        for n in range(dates.size - 1, 0, -1):
            dt = (dates[n - 1] - dates[n]) * kyr * self._to_mat_time
            weights = op.scheme_weights(scheme, dt)
            x_duals = {}
            sig_cols = {}
            psi_cols = {}
            if itau_kernel is not None:
                fwd_n = cursor.state(n)
                fwd_prev = cursor.state(n - 1)
                dweights = op.scheme_weight_derivatives(scheme, dt)
            for l in op.degrees:
                deg = op.degree(l)
                summed = ddual[l].copy()
                for k, wk in enumerate(weights):
                    b = 1.0 - wk[0][deg.e0 :] if implicit else wk[2][deg.e0 :]
                    summed += b[None, :, :, None] * mdual[l][k]
                xd = deg.strain_transpose(summed)
                x_duals[l] = xd
                units_lv = op.unit_responses(scheme, dt, l)
                sig_cols[l] = units_lv[:, 0] @ xd
                psi_cols[l] = units_lv[:, 1] @ xd
            sigma_dual = pack(sig_cols, self._to_mat_density)
            psi_dual = pack(psi_cols, self._to_mat_potential)
            g_xi, g_omega, extra = sle_dual_inputs(sigma_dual, psi_dual, n)
            sle = self._sle_for(dt)
            f_dual, su_dual, sphi_dual, j_dual = sle.adjoint_generalised_solve(
                self.state,
                sea_level_dual=SHGrid.from_array(g_xi, grid=model.grid, copy=False),
                angular_velocity_dual=g_omega,
                ocean_function=C_grid(n),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                **extra,
            )
            load_duals[n] = f_dual.data + synth(sigma_dual) / r2
            su_cols = self._columns(su_dual.coeffs, self._to_nd_length)
            sphi_cols = self._columns(sphi_dual.coeffs, self._to_nd_potential)
            for l in op.degrees:
                deg = op.degree(l)
                rhs = (
                    x_duals[l]
                    + deg.surface_dual("U", su_cols(l, 0.0))
                    + deg.surface_dual("phi", sphi_cols(l, 0.0))
                )
                if l == 0:
                    rhs[:, 0] += deg.moment_row * (j_dual * self._to_nd_moment)
                y = op.effective_system(scheme, dt, l).solve(rhs)
                sy = deg.strain(y)
                mtilde_dual = mdual[l] + deg.apply_weights(sy)
                if itau_kernel is not None:
                    m_prev = fwd_prev.m[l]
                    d0 = fwd_prev.cached_strain[l]
                    d1 = fwd_n.cached_strain[l]
                    Wd1 = deg.apply_weights(d1)
                    for k, dwk in enumerate(dweights):
                        if implicit:
                            dw = dwk[0][deg.e0 :][None, :, :, None]
                            g = (
                                (
                                    mtilde_dual[k] * m_prev[k]
                                    - mdual[l][k] * d1
                                    - Wd1[k] * sy
                                )
                                * dw
                            ).sum(axis=(0, 3))
                        else:
                            de = dwk[0][deg.e0 :][None, :, :, None]
                            dalpha = dwk[1][deg.e0 :][None, :, :, None]
                            dbeta = dwk[2][deg.e0 :][None, :, :, None]
                            g = (
                                mtilde_dual[k] * (de * m_prev[k] + dalpha * d0)
                                + (mdual[l][k] * d1 + Wd1[k] * sy) * dbeta
                            ).sum(axis=(0, 3))
                        itau_kernel[k][deg.e0 :] += dt * g
                for k, wk in enumerate(weights):
                    if implicit:
                        w = wk[0][deg.e0 :][None, :, :, None]
                        mdual[l][k] = w * mtilde_dual[k]
                    else:
                        e = wk[0][deg.e0 :][None, :, :, None]
                        mdual[l][k] = e * mtilde_dual[k]
                if implicit:
                    ddual[l] = np.zeros_like(ddual[l])
                else:
                    ddual[l] = sum(
                        wk[1][deg.e0 :][None, :, :, None] * mtilde_dual[k]
                        for k, wk in enumerate(weights)
                    )
        # the start: the instantaneous elastic solve at dates[0]
        sig_cols = {}
        psi_cols = {}
        for l in op.degrees:
            deg = op.degree(l)
            xd = deg.strain_transpose(ddual[l])
            units_lv = op.unit_responses(None, 0.0, l)
            sig_cols[l] = units_lv[:, 0] @ xd
            psi_cols[l] = units_lv[:, 1] @ xd
        sigma_dual = pack(sig_cols, self._to_mat_density)
        psi_dual = pack(psi_cols, self._to_mat_potential)
        g_xi, g_omega, extra = sle_dual_inputs(sigma_dual, psi_dual, 0)
        f_dual, _, _, _ = self._sle_elastic().adjoint_generalised_solve(
            self.state,
            sea_level_dual=SHGrid.from_array(g_xi, grid=model.grid, copy=False),
            angular_velocity_dual=g_omega,
            ocean_function=C_grid(0),
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            **extra,
        )
        load_duals[0] = f_dual.data + synth(sigma_dual) / r2

        # the grounding factor follows the trajectory's shorelines; the
        # BASELINE grounds with the initial C0, since every load depends
        # on the initial thickness through rho_i (1 - C0) I0
        raw = [
            parameters.ice_density * (1.0 - C_at(n)) * load_duals[n]
            for n in range(dates.size)
        ]
        baseline = -parameters.ice_density * (1.0 - C) * sum(load_duals)
        thickness = list(raw)
        thickness[0] = thickness[0] + baseline
        # the initial sea level enters each step's load as
        # rho_w (C_n - C0) delta SL0 -- the discrete analogue of Yu et
        # al.'s eq. (65); identically zero under frozen shorelines
        initial_sea_level = water_density * sum(
            (C_at(n) - C) * load_duals[n] for n in range(dates.size)
        )

        # the distributional view: the delta amplitude at each
        # observation time is the instantaneous (unrelaxed) fingerprint
        # adjoint of that measurement, independent of the date grid
        instantaneous = {}
        elastic_sle = self._sle_elastic()
        lumps = {}
        for node, m in duals.items():
            f_dual, _, _, _ = elastic_sle.adjoint_generalised_solve(
                self.state,
                sea_level_dual=m.sea_level,
                displacement_dual=m.displacement,
                potential_dual=m.gravity_potential,
                angular_velocity_dual=m.angular_velocity,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            lumps[node] = parameters.ice_density * (1.0 - C_at(node)) * f_dual.data
            instantaneous[float(dates[node])] = SHGrid.from_array(
                lumps[node], grid=model.grid
            )
        # trapezoidal time weights of the date grid, in kiloyears
        w = np.empty(dates.size)
        w[0] = 0.5 * (dates[0] - dates[1])
        w[-1] = 0.5 * (dates[-2] - dates[-1])
        w[1:-1] = 0.5 * (dates[:-2] - dates[2:])
        density = [(raw[n] - lumps.get(n, 0.0)) / w[n] for n in range(dates.size)]

        # the quelled kernel: with respect to the thickness RATE, the
        # cumulative tail of the node duals -- the deltas integrate to
        # finite jumps, and the initial-thickness dual vanishes
        # identically (sum of all node duals plus the baseline is zero)
        rate = []
        tail = np.zeros_like(C)
        for n in range(dates.size - 1, 0, -1):
            tail = tail + raw[n]
            rate.append(tail.copy())
        rate.reverse()

        return IceKernels(
            dates,
            [SHGrid.from_array(ld, grid=model.grid) for ld in load_duals],
            [SHGrid.from_array(th, grid=model.grid) for th in thickness],
            SHGrid.from_array(baseline, grid=model.grid),
            SHGrid.from_array(initial_sea_level, grid=model.grid),
            instantaneous,
            [SHGrid.from_array(d, grid=model.grid) for d in density],
            [SHGrid.from_array(r, grid=model.grid) for r in rate],
            itau_kernel,
        )


def calibrate_initial_sea_level(
    operator: ViscoelasticOperator,
    ice_model: BaseIceModel,
    dates: Sequence[float],
    present_sea_level: SHGrid,
    model: EarthModel,
    *,
    scheme: str = "exponential_trapezoid",
    exclude_caspian: bool = True,
    iterations: int = 4,
    rtol: float = 1e-3,
    anderson_memory: int = 5,
    rotational_feedbacks: bool = True,
    verbose: bool = False,
) -> EarthState:
    """The background state at the first date whose forward run reaches
    the observed present-day sea level, under frozen shorelines.

    A generic ice model carries no sea level, so the initial state must
    be found. Starting from the present-day sea level as the guess, the
    classical fixed point

        SL(t0) <- SL(present, observed) - xi(present; SL(t0))

    is iterated with Anderson acceleration.  This is a shorthand for
    `SeaLevelEvolution.calibrate_initial_sea_level`, which also offers
    migrating shorelines, weights and the adjoint gradient.  The ICE-NG
    models carry their own (inverted) initial sea level and do not need
    it.

    Args:
        operator: The radial viscoelastic operator.
        ice_model: Any ice model; thickness alone is used.
        dates: Kiloyears before present, decreasing, ending at the
            present-day observation's date.
        present_sea_level: The observed present-day sea level, an
            SHGrid in the Earth model's units.
        model: The Earth model of the run.
        scheme, rotational_feedbacks: As for `SeaLevelEvolution`.
        exclude_caspian: Passed to each trial state.
        iterations: The most passes taken.
        rtol: Convergence, the largest update of the initial sea level
            relative to its largest value.
        anderson_memory: The depth of the quasi-Newton acceleration;
            0 is the plain fixed-point iteration.
        verbose: Prints each pass's misfit and update.

    Returns:
        The calibrated background `EarthState` at the first date.
    """
    model.check_field(present_sea_level)
    ice_start = ice_model.get_ice_thickness(
        float(np.asarray(dates, dtype=float)[0]),
        model.lmax,
        grid=model.grid_name,
        sampling=model.sampling,
        extend=model.extend,
    )
    guess = EarthState(
        ice_start, present_sea_level, model, exclude_caspian=exclude_caspian
    )
    evolution = SeaLevelEvolution(operator, guess, scheme=scheme)
    return evolution.calibrate_initial_sea_level(
        dates,
        ice_model,
        present_sea_level,
        iterations=iterations,
        rtol=rtol,
        anderson_memory=anderson_memory,
        rotational_feedbacks=rotational_feedbacks,
        verbose=verbose,
    ).state
