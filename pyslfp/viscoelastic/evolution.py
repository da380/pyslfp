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
from typing import Optional, Union
from dataclasses import dataclass, field, replace

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from planetmodel import units

from ..core import EarthModel
from ..core.model import SECONDS_PER_KILOYEAR
from ..core.response import RELATIVE_SEA_LEVEL_CHANGE, GRDResponse, GRDSpace
from ..core.observables import centrifugal_potential_operator
from ..core.spaces import lebesgue_response_space
from ..ice import BaseIceModel, BaseIceSeaLevelModel
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
    "KernelDerivatives",
    "Measurement",
    "SeaLevelSnapshot",
    "SeaLevelEvolution",
    "calibrate_initial_sea_level",
]

#: Seconds per kiloyear (Julian years), for the date conversions.
KYR = SECONDS_PER_KILOYEAR

POTENTIAL = units.LENGTH**2 * units.TIME**-2
SURFACE_DENSITY = units.DENSITY * units.LENGTH
MOMENT = units.MASS * units.LENGTH**2


@dataclass(frozen=True)
class Measurement:
    """The duals of one observation time on the scalar members of the
    response: the objective gains

        int g_xi xi dS + int g_u U dS + int g_phi phi dS
        + g_omega . omega

    at that date, with the grids paired by the surface integral and U
    the vertical displacement.  The fields are named after the
    response's members.  A bare SHGrid given as a measurement is
    shorthand for a sea level dual alone, and a `GRDResponse` stands for
    its inner product with the response in its own space, which is how
    the horizontal displacement is observed."""

    relative_sea_level_change: Optional[SHGrid] = None
    displacement: Optional[SHGrid] = None
    potential_change: Optional[SHGrid] = None
    angular_velocity_change: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _Duals:
    """A measurement in the form the adjoints take: the scalar members'
    duals as grids paired by the surface integral, the angular velocity
    change's, and `horizontal`, the dual s of the coefficients of V in
    the model's normalisation, J = sum s_lm V_lm."""

    relative_sea_level_change: Optional[SHGrid] = None
    displacement: Optional[SHGrid] = None
    potential_change: Optional[SHGrid] = None
    angular_velocity_change: Optional[np.ndarray] = None
    horizontal: Optional[np.ndarray] = None

    @classmethod
    def of_response(cls, dual: GRDResponse) -> _Duals:
        """The duals of J = <dual, response> in the dual's own space.

        With lambda the space's metric and c the components, J = sum
        lambda g c, so the duals are the metric-weighted components:
        the scalar members' as fields, which then pair by the surface
        integral since the L2 metric of the components is one, and V's,
        whose components are b V_lm, as s_lm = b lambda g."""
        space = dual.space
        weighted = space.from_components(
            space.metric_values * space.to_components(dual)
        )
        field_space = space.field_space
        horizontal = None
        if space.has_horizontal:
            degrees = field_space.degrees
            orders = field_space.orders
            mask = degrees >= 1
            horizontal = np.zeros((2, field_space.lmax + 1, field_space.lmax + 1))
            horizontal[
                (orders[mask] < 0).astype(int), degrees[mask], np.abs(orders[mask])
            ] = field_space.radius * weighted.displacement.horizontal
        return cls(
            field_space.from_components(weighted.relative_sea_level_change),
            field_space.from_components(weighted.displacement.vertical),
            field_space.from_components(weighted.potential_change),
            weighted.angular_velocity_change,
            horizontal,
        )

    def tangential_load(self, model: EarthModel) -> Optional[SHGrid]:
        """The tangential traction whose forward response is the adjoint of
        the horizontal dual: zeta^v_lm = -s_lm / (b^2 l(l + 1)), matching
        the scale of the traction channel's zeta^u = -g_u."""
        if self.horizontal is None:
            return None
        b = model.parameters.mean_sea_floor_radius
        ell = np.arange(model.lmax + 1, dtype=float)
        factor = np.zeros_like(ell)
        factor[1:] = -1.0 / (b**2 * ell[1:] * (ell[1:] + 1.0))
        return model.expand_coefficient(
            SHCoeffs.from_array(
                factor[None, :, None] * self.horizontal,
                normalization=model.normalization,
                csphase=model.csphase,
            )
        )


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
    construction.  The Sobolev kernel, the derivative's Riesz
    representer in a space of ice histories, is the adjoint of
    `IceHistoryOperator`, built from the per-node duals.

    The kernels with respect to the ice history and the initial sea
    level are fields on the surface grid and pair by int . dS.  The
    kernels with respect to the inverse relaxation times, `itau`, are
    fields in the Earth: each is a DENSITY PER UNIT VOLUME, held as its
    samples K(r) at the nodes of the material's mesh, (nspec, ngll) per
    branch, so that for a perturbation delta(r) of the branch's inverse
    relaxation time

        dJ = int K delta dV = 4 pi int K(r) delta(r) r^2 dr
           ~ sum W K delta,

    W the nodes' volume weights (`Material.volume_weights`, and the sum
    is `Material.volume_pairing`).  Neither the r^2 nor the 4 pi is in
    the kernel: both belong to the volume measure.  The sweeps
    accumulate the gradient with respect to the nodal values, W K, and
    divide the weights out on return (`Material.kernel_samples`); a
    node at the centre has no volume, and its sample is the element's
    polynomial through the other nodes.  The values are in the
    material's units, the objective's unit per unit of inverse time per
    unit of volume; converted through the material's `scales`, the
    objective's unit per s^-1 per m^3."""

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
    #: densities per unit volume sampled at the nodes of the material's
    #: mesh, (nspec, ngll) arrays in its units (see above), when the
    #: adjoint was given the forward trajectory (`internal_states`);
    #: None otherwise.
    itau: Optional[list[np.ndarray]] = field(default=None, repr=False)

    def viscosity(self, rheology: Rheology) -> list[np.ndarray]:
        """The viscosity kernels dJ / d eta(r) per branch from the
        inverse-relaxation-time kernels, densities per unit volume
        sampled at the nodes as `itau` is: itau = mu / eta gives
        d itau / d eta = -itau^2 / mu, so an elastic branch (itau = 0)
        has no viscosity sensitivity in this parameterisation.  They
        pair with a viscosity perturbation through
        `Material.volume_pairing`."""
        if self.itau is None:
            raise ValueError("no itau kernels: run the adjoint with internal_states")
        out = []
        for kernel, mu_k, itau_k in zip(self.itau, rheology.mu, rheology.itau):
            safe = np.where(mu_k > 0.0, mu_k, 1.0)
            out.append(np.where(mu_k > 0.0, -(itau_k**2) / safe, 0.0) * kernel)
        return out


@dataclass(frozen=True)
class KernelDerivatives:
    """The derivative of an objective's kernels along a perturbation of the
    model parameters: what a second-order adjoint returns, the components
    of the Hessian's action on that perturbation.  `thickness` and
    `initial_sea_level` are the derivatives of the discrete node duals of
    `IceKernels`, in the same pairing, and `itau` that of the
    inverse-relaxation-time kernels, likewise a density per unit volume
    sampled at the nodes, which pairs with a perturbation of the inverse
    relaxation times through `Material.volume_pairing`."""

    dates: np.ndarray
    thickness: list[SHGrid] = field(repr=False)
    initial_sea_level: SHGrid = field(repr=False)
    itau: list[np.ndarray] = field(repr=False)


@dataclass(frozen=True)
class SeaLevelSnapshot:
    """The response at one time of an evolution, in the Earth model's
    non-dimensional units: a `GRDResponse` of the run's response space,
    holding the sea level change, the displacement (with its horizontal
    part when the space holds it), the potential change since the start
    and, with rotational feedback, the angular velocity change.  Fields
    are read by name, `response.field("relative_sea_level_change")`.  A
    run driven by an ice history also records `date`, in kiloyears
    before present."""

    t: float
    response: GRDResponse = field(repr=False)
    date: float | None = None
    #: The step's ocean function under migrating shorelines (None when
    #: the shorelines are frozen at the background state's): the C(t)
    #: record an adjoint calculation reads back.
    ocean_function: SHGrid | None = field(default=None, repr=False)
    #: The ocean function the derivatives of the loads see at this time,
    #: C + Lambda (`EarthState.effective_ocean_function`), and its derivative
    #: per unit of sea level (`EarthState.flotation_curvature`), which the
    #: second derivative of the loads carries; the ocean function itself
    #: under the sharp flotation rule, and zero, which None stands for.
    effective_ocean_function: SHGrid | None = field(default=None, repr=False)
    flotation_curvature: SHGrid | None = field(default=None, repr=False)
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


def _effective_ocean_function(snapshot: SeaLevelSnapshot) -> Optional[SHGrid]:
    """The ocean function the derivatives see at a snapshot: the effective
    one when recorded, else the ocean function itself."""
    if snapshot.effective_ocean_function is not None:
        return snapshot.effective_ocean_function
    return snapshot.ocean_function


def _effective_ocean_functions(snapshots: Sequence[SeaLevelSnapshot]) -> list[SHGrid]:
    return [_effective_ocean_function(snap) for snap in snapshots]


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


class _ForwardCursor:
    """Serves the forward internal states to a forward sweep, in order:
    stored states where the recording kept them, exact forward steps
    under the recorded loads between, which amplify nothing."""

    def __init__(self, evolution, trajectory, dates, scheme) -> None:
        self._evolution = evolution
        self._trajectory = trajectory
        kyr = KYR / evolution.model.parameters.scales.factor(units.TIME)
        self._dt = (dates[:-1] - dates[1:]) * kyr * evolution._to_mat_time
        self._scheme = scheme
        self._node = 0
        self._state = trajectory.states[0]

    def state(self, node: int) -> ViscoelasticState:
        """The forward state at a date node, reconstructing as needed."""
        if node < self._node:
            raise ValueError("the trajectory cursor moves forwards")
        while self._node < node:
            target = self._node + 1
            stored = self._trajectory.states[target]
            if stored is not None:
                self._state = stored
            else:
                evolution = self._evolution
                self._state, _ = evolution.operator.step(
                    self._state,
                    float(self._dt[self._node]),
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
    measurement: dict[float, SHGrid | Measurement | GRDResponse],
    dates: np.ndarray,
    model: EarthModel,
) -> dict[int, _Duals]:
    """Measurement duals keyed by date, checked, re-keyed by node index
    and put in the form the adjoints take; a bare grid means a sea level
    dual, and a response its inner product in its own space."""
    duals: dict[int, _Duals] = {}
    b = model.parameters.mean_sea_floor_radius
    for key, value in measurement.items():
        matches = np.flatnonzero(np.isclose(dates, key))
        if matches.size != 1:
            raise ValueError(f"measurement date {key} is not on the date grid")
        if isinstance(value, GRDResponse):
            field_space = value.space.field_space
            if field_space.lmax != model.lmax or not np.isclose(
                field_space.radius, b, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    f"the measurement at {key} is not over the model's sphere"
                )
            value = _Duals.of_response(value)
        elif isinstance(value, Measurement):
            value = _Duals(
                value.relative_sea_level_change,
                value.displacement,
                value.potential_change,
                value.angular_velocity_change,
            )
        else:
            value = _Duals(relative_sea_level_change=value)
        for grid in (
            value.relative_sea_level_change,
            value.displacement,
            value.potential_change,
        ):
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
        self._spaces: dict[bool, GRDSpace] = {}

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

    def _kernel_samples(
        self, gradients: Optional[list[np.ndarray]]
    ) -> Optional[list[np.ndarray]]:
        """The kernels per unit volume, per branch, from the gradients
        with respect to the nodal values the sweeps accumulate."""
        if gradients is None:
            return None
        material = self.operator.material
        return [material.kernel_samples(g) for g in gradients]

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

    def _space(
        self, space: Optional[GRDSpace], rotational_feedbacks: bool | str
    ) -> GRDSpace:
        """The response space of a run: the one given, or the L2 space
        without the horizontal part, holding the angular velocity change
        exactly when there is rotational feedback.  The solves check that
        a given space is over the model's sphere and matches the
        feedback."""
        if space is not None:
            return space
        rotation = rotational_feedbacks is not False
        if rotation not in self._spaces:
            self._spaces[rotation] = lebesgue_response_space(
                self.model, rotation=rotation
            )
        return self._spaces[rotation]

    def _memory_horizontal(self, step, space: GRDSpace) -> Optional[SHCoeffs]:
        """The memory field's horizontal displacement potential V, when the
        response space holds the horizontal part."""
        if not space.has_horizontal:
            return None
        return self._coeffs(step.memory_surface("V"), self._to_nd_length)

    def _load_and_tide(
        self,
        load_grid: SHGrid,
        response: GRDResponse,
        rotational_feedbacks: bool | str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The coefficients of the total surface load the response carries
        under frozen shorelines, the direct load and the ocean's water
        rho_w C xi, and of the centrifugal potential of its angular
        velocity change: the forcings that finish a radial step."""
        xi = response.field(RELATIVE_SEA_LEVEL_CHANGE).data
        total = (
            load_grid.data + self._water_density * self.state.ocean_function.data * xi
        )
        sigma = self.model.expand_field(
            SHGrid.from_array(total, grid=self.model.grid, copy=False)
        ).coeffs
        psi = (
            self._centrifugal(response.angular_velocity_change)
            if rotational_feedbacks
            else np.zeros_like(sigma)
        )
        return sigma, psi

    # -- the loop ------------------------------------------------------------

    def run(
        self,
        times: Sequence[float],
        ice_load: Callable[[float], SHGrid],
        *,
        space: Optional[GRDSpace] = None,
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
        at the start is stepped at full order.  Returns the response at
        every time, the first included, and the final internal state,
        whose time is in the MATERIAL's units.

        `space` is the `GRDSpace` the responses belong to, over the
        model's sphere, holding the angular velocity change exactly when
        there is rotational feedback; when it holds the horizontal part,
        each step adds the memory field's V to the response's.  None
        gives the L2 space without the horizontal part.  The space sets
        the inner product alone: the components are the same in every
        space over the sphere.

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
        space = self._space(space, rotational_feedbacks)

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        vstate.t = times[0] * self._to_mat_time

        if rotational_feedbacks is True:
            self._assert_transverse_stable()

        load_grid = ice_load(times[0])
        response = self._sle_elastic().solve_response(
            self.state,
            load_grid,
            space,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
        )
        sigma, psi = self._load_and_tide(load_grid, response, rotational_feedbacks)
        if stride:
            op.prime_strain(
                vstate,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
        snapshots = [
            SeaLevelSnapshot(
                float(times[0]),
                response,
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
            response = sle.solve_response(
                self.state,
                load_grid,
                space,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                horizontal_response=self._memory_horizontal(step, space),
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
            )
            sigma, psi = self._load_and_tide(load_grid, response, rotational_feedbacks)
            vstate, _ = step.finish(
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            keep = bool(stride) and (node % stride == 0 or node == last_node)
            snapshots.append(
                SeaLevelSnapshot(
                    float(t1),
                    response,
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
        space: Optional[GRDSpace] = None,
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
        `ice_load(t)` must be continuous.  `space` is as for `run`.
        """
        if t_final <= t_start:
            raise ValueError("t_final must lie beyond t_start")
        if dt_initial <= 0.0:
            raise ValueError("dt_initial must be positive")
        op = self.operator
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        scheme = self.scheme
        if scheme != "exponential_trapezoid":
            raise ValueError("the adaptive loop is the exponential trapezoid's")
        space = self._space(space, rotational_feedbacks)

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        vstate.t = t_start * self._to_mat_time

        # the right limit at the start, as in `run`
        load_grid = ice_load(t_start)
        response = self._sle_elastic().solve_response(
            self.state,
            load_grid,
            space,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
        sigma, psi = self._load_and_tide(load_grid, response, rotational_feedbacks)
        snapshots = [SeaLevelSnapshot(float(t_start), response)]

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
            response = sle.solve_response(
                self.state,
                grid1,
                space,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                horizontal_response=self._memory_horizontal(step, space),
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            sigma_new, psi_new = self._load_and_tide(
                grid1, response, rotational_feedbacks
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
            snapshots.append(SeaLevelSnapshot(float(t1), response))
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
        space: Optional[GRDSpace] = None,
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
        The snapshots carry their dates, and their responses belong to
        `space`, as for `run`.

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
                space=space,
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
                space=space,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
                record=record,
            )
        if shorelines == "migrating":
            return self._run_migrating(
                dates,
                ice_model,
                space=space,
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
            space=space,
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
        self,
        dates,
        ice_model,
        *,
        space,
        rotational_feedbacks,
        rtol,
        verbose,
        record=False,
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
        space = self._space(space, rotational_feedbacks)
        kyr = KYR / parameters.scales.factor(units.TIME)
        date0 = float(dates[0])

        if self.state.has_prescribed_ocean_function:
            raise ValueError(
                "migrating shorelines follow the flotation condition from the "
                "first date, so the background state must not prescribe its "
                "ocean function"
            )
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
        snapshots = [
            SeaLevelSnapshot(
                0.0,
                space.zero(),
                date=date0,
                ocean_function=self.state.ocean_function,
                effective_ocean_function=self.state.effective_ocean_function,
                flotation_curvature=self.state.flotation_curvature,
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
            final_state, response = sle.solve_nonlinear_response(
                self.state,
                space,
                ice_thickness_change=change,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                horizontal_response=self._memory_horizontal(step, space),
                inertia_moment_change=moment,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
            )
            sigma = self._total_load_coeffs(final_state)
            psi = (
                self._centrifugal(response.angular_velocity_change)
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
                    response,
                    date=float(b),
                    ocean_function=final_state.ocean_function,
                    effective_ocean_function=final_state.effective_ocean_function,
                    flotation_curvature=final_state.flotation_curvature,
                    internal_state=vstate if keep else None,
                    load_coefficients=sigma if stride else None,
                    tide_coefficients=psi if stride else None,
                )
            )
        return snapshots, vstate

    # -- the tangent-linear runs -----------------------------------------------

    def _first_order_drive(
        self,
        method: str,
        dt: float,
        prev: ViscoelasticState,
        nxt: ViscoelasticState,
        ditau: Sequence[np.ndarray],
    ) -> dict[int, np.ndarray]:
        """The memory increment of a step from `prev` to `nxt`, two states
        of a trajectory with their strains, for a perturbation `ditau` of
        the inverse relaxation times, bilinear in the two: the forward mode
        of the scheme's weights ("discrete"), or the continuous drive
        q_k = (d - m_k) delta itau_k / itau_k discretised by the scheme
        ("continuous")."""
        op = self.operator
        implicit = self.scheme == "backward_euler"
        if method == "discrete":
            dweights = op.scheme_weight_derivatives(self.scheme, dt)
        else:
            weights = op.scheme_weights(self.scheme, dt)
        out = {}
        for l in op.degrees:
            deg = op.degree(l)
            m_prev = prev.m[l]
            d0 = prev.cached_strain[l]
            d1 = nxt.cached_strain[l]
            extra = np.zeros_like(m_prev)
            for k in range(op.rheology.num_branches):
                dh = (dt * ditau[k][deg.e0 :])[None, :, :, None]
                if method == "discrete":
                    if implicit:
                        dw = dweights[k][0][deg.e0 :][None, :, :, None]
                        extra[k] = dw * dh * (m_prev[k] - d1)
                    else:
                        de, dalpha, dbeta = (
                            w[deg.e0 :][None, :, :, None] for w in dweights[k]
                        )
                        extra[k] = dh * (de * m_prev[k] + dalpha * d0 + dbeta * d1)
                else:
                    itau_k = deg.itau[k][None, :, :, None]
                    ratio = np.where(
                        itau_k > 0.0,
                        dh / (dt * np.where(itau_k > 0.0, itau_k, 1.0)),
                        0.0,
                    )
                    q0 = (d0 - m_prev[k]) * ratio
                    q1 = (d1 - nxt.m[l][k]) * ratio
                    if implicit:
                        w = weights[k][0][deg.e0 :][None, :, :, None]
                        extra[k] = (1.0 - w) * q1
                    else:
                        alpha = weights[k][1][deg.e0 :][None, :, :, None]
                        beta = weights[k][2][deg.e0 :][None, :, :, None]
                        extra[k] = alpha * q0 + beta * q1
            out[l] = extra
        return out

    def _second_weight_drive(
        self,
        dt: float,
        prev: ViscoelasticState,
        nxt: ViscoelasticState,
        ditau_u: Sequence[np.ndarray],
        ditau_v: Sequence[np.ndarray],
    ) -> dict[int, np.ndarray]:
        """The discrete method's term from the second derivatives of the
        scheme's weights, paired with the forward trajectory and the two
        perturbations of the inverse relaxation times."""
        op = self.operator
        implicit = self.scheme == "backward_euler"
        d2weights = op.scheme_weight_second_derivatives(self.scheme, dt)
        out = {}
        for l in op.degrees:
            deg = op.degree(l)
            m_prev = prev.m[l]
            d0 = prev.cached_strain[l]
            d1 = nxt.cached_strain[l]
            extra = np.zeros_like(m_prev)
            for k in range(op.rheology.num_branches):
                dh2 = (dt * dt * ditau_u[k][deg.e0 :] * ditau_v[k][deg.e0 :])[
                    None, :, :, None
                ]
                if implicit:
                    d2w = d2weights[k][0][deg.e0 :][None, :, :, None]
                    extra[k] = d2w * dh2 * (m_prev[k] - d1)
                else:
                    d2e, d2alpha, d2beta = (
                        w[deg.e0 :][None, :, :, None] for w in d2weights[k]
                    )
                    extra[k] = dh2 * (d2e * m_prev[k] + d2alpha * d0 + d2beta * d1)
            out[l] = extra
        return out

    def _check_dates_and_snapshots(self, dates, snapshots):
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the "
                "past first), with at least two entries"
            )
        snapshots = list(snapshots)
        if len(snapshots) != dates.size:
            raise ValueError("the snapshots must give one entry per date node")
        migrating = snapshots[0].ocean_function is not None
        if migrating and any(snap.ocean_function is None for snap in snapshots):
            raise ValueError(
                "every snapshot of a migrating run carries its ocean function"
            )
        return dates, snapshots, migrating

    def _check_itau(self, itau: Sequence[np.ndarray]) -> list[np.ndarray]:
        op = self.operator
        ditau = [np.asarray(a, dtype=float) for a in itau]
        shape = op.material.mesh.r.shape
        K = op.rheology.num_branches
        if len(ditau) != K or any(a.shape != shape for a in ditau):
            raise ValueError(
                f"itau must give {K} nodal arrays of shape {shape}, one per branch"
            )
        return ditau

    def _full_trajectory(self, snapshots, dates, what: str) -> ForwardTrajectory:
        """A tangent-linear run's trajectory, which must be recorded at every
        node."""
        trajectory = self._as_trajectory(snapshots, dates)
        if any(state is None for state in trajectory.states):
            raise ValueError(f"{what} must be recorded at every node")
        return trajectory

    def _tangent_run(
        self,
        dates: np.ndarray,
        snapshots: list[SeaLevelSnapshot],
        migrating: bool,
        direct_load: Callable[[int], Optional[SHGrid]],
        drive: Optional[Callable[[int, float], dict[int, np.ndarray]]],
        *,
        space: Optional[GRDSpace],
        rotational_feedbacks: bool | str,
        rtol: float,
        record: bool,
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """The loop the tangent-linear runs share: the forward problem with
        the trajectory's ocean functions prescribed, the direct load given
        per node, and a memory increment per step."""
        op = self.operator
        model = self.model
        parameters = model.parameters
        if rotational_feedbacks is True:
            self._assert_transverse_stable()
        space = self._space(space, rotational_feedbacks)
        kyr = KYR / parameters.scales.factor(units.TIME)
        rho_w = parameters.water_density
        C0 = self.state.effective_ocean_function.data

        def C_at(node: int) -> np.ndarray:
            return _effective_ocean_function(snapshots[node]).data if migrating else C0

        def C_grid(node: int) -> Optional[SHGrid]:
            return _effective_ocean_function(snapshots[node]) if migrating else None

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        op.prime_strain(vstate)
        sigma = np.zeros((2, model.lmax + 1, model.lmax + 1))
        psi = np.zeros_like(sigma)
        # the direct load vanishes at the first date, so the start, the
        # instantaneous response to it, is zero
        out = [
            SeaLevelSnapshot(
                0.0,
                space.zero(),
                date=float(dates[0]),
                ocean_function=C_grid(0),
                internal_state=vstate if record else None,
                load_coefficients=sigma if record else None,
                tide_coefficients=psi if record else None,
            )
        ]
        for node, (a, b) in enumerate(zip(dates[:-1], dates[1:]), start=1):
            dt = (a - b) * kyr * self._to_mat_time
            step = op.start_step(
                vstate,
                dt,
                scheme=self.scheme,
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            if drive is not None:
                step.add_memory(drive(node, dt))
            sle = self._sle_for(dt)
            memory_u = self._coeffs(step.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(step.memory_surface("phi"), self._to_nd_potential)
            moment = step.inertia_moment()[0] * self._to_nd_moment
            load_grid = direct_load(node)
            response = sle.solve_response(
                self.state,
                load_grid,
                space,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                horizontal_response=self._memory_horizontal(step, space),
                inertia_moment_change=moment,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            xi = response.field(RELATIVE_SEA_LEVEL_CHANGE).data
            total = rho_w * C_at(node) * xi
            if load_grid is not None:
                total = total + load_grid.data
            sigma = model.expand_field(
                SHGrid.from_array(total, grid=model.grid, copy=False)
            ).coeffs
            psi = (
                self._centrifugal(response.angular_velocity_change)
                if rotational_feedbacks
                else np.zeros_like(sigma)
            )
            vstate, _ = step.finish(
                load=self._columns(sigma, self._to_mat_density),
                tide=self._columns(psi, self._to_mat_potential),
            )
            out.append(
                SeaLevelSnapshot(
                    float(vstate.t / self._to_mat_time),
                    response,
                    date=float(b),
                    ocean_function=C_grid(node),
                    internal_state=vstate if record else None,
                    load_coefficients=sigma if record else None,
                    tide_coefficients=psi if record else None,
                )
            )
        return out, vstate

    def tangent_linear_ice_history(
        self,
        dates: Sequence[float],
        snapshots: Sequence[SeaLevelSnapshot],
        *,
        ice_thickness: Optional[Sequence[SHGrid]] = None,
        initial_sea_level: Optional[SHGrid] = None,
        itau: Optional[Sequence[np.ndarray]] = None,
        method: str = "discrete",
        space: Optional[GRDSpace] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        record: bool = False,
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """The tangent-linear run: the derivative of a run's responses
        along a perturbation of the ice history, the initial sea level
        and the inverse relaxation times, about the forward run whose
        `snapshots` are given.

        With the ocean functions of the forward run held, the forward
        problem is linear in its loads, so the derivative is the forward
        problem once more: the shorelines prescribed from the trajectory,
        the direct load the perturbation's,

            rho_i (1 - C_n) delta I_n - rho_i (1 - C_0) delta I_0
            + rho_w (C_n - C_0) delta S_0,

        and, for a perturbation of the relaxation times, the drive of each
        branch shifted by a known term from the trajectory.  Holding the
        ocean functions is exact in the continuum, the singular terms of
        shoreline motion cancelling (Yu et al. 2025); on the grid the
        map is piecewise affine in the ice history and the initial sea
        level, and this is its derivative on each piece.  Under frozen
        shorelines the initial sea level contributes nothing.

        `method` chooses how the relaxation-time perturbation enters:
        "discrete" differentiates the scheme's weights, the forward mode
        of the time-stepped map, whose adjoint is `adjoint_ice_history`;
        "continuous" discretises the continuous tangent-linear problem,
        the drive q_k = (d - m_k) delta itau_k / itau_k entering each
        step like the strain, whose adjoint is
        `continuous_adjoint_ice_history`.  The two coincide in the ice
        history and the initial sea level, and differ in the relaxation
        times by the order of the scheme.

        Args:
            dates: The forward run's dates, kiloyears before present,
                decreasing.
            snapshots: The forward run's snapshots, one per date: their
                ocean functions are the trajectory's under migrating
                shorelines, and with `itau` the run must have recorded
                its internal states (`record`), every node or on a
                checkpoint stride, the gaps being refilled by forward
                steps.
            ice_thickness: The thickness perturbation at each date, one
                grid per node, in the Earth model's units; None for no
                perturbation.
            initial_sea_level: The perturbation of the initial sea level,
                a grid; None for none.
            itau: The perturbation of each branch's inverse relaxation
                times, its values at the nodes of the material's mesh,
                (nspec, ngll) arrays in its units, as `IceKernels.itau`
                is laid out, with which it pairs through
                `Material.volume_pairing`; None for none.
            method: "discrete" or "continuous".
            space, rotational_feedbacks, rtol: As for `run_ice_history`,
                and as the forward run was made.
            record: Whether the snapshots carry the tangent-linear
                internal states and loads at every node, which a
                second-order run about this one needs.

        Returns:
            The perturbed responses at every date, as snapshots carrying
            their dates and the trajectory's ocean functions, and the
            final tangent-linear internal state.
        """
        if method not in ("discrete", "continuous"):
            raise ValueError(
                f"method must be 'discrete' or 'continuous', got {method!r}"
            )
        dates, snapshots, migrating = self._check_dates_and_snapshots(dates, snapshots)
        model = self.model
        parameters = model.parameters
        rho_i = parameters.ice_density
        rho_w = parameters.water_density
        C0 = self.state.effective_ocean_function.data
        if ice_thickness is not None:
            ice_thickness = list(ice_thickness)
            if len(ice_thickness) != dates.size:
                raise ValueError("ice_thickness must give one grid per date node")
            for grid in ice_thickness:
                model.check_field(grid)
            delta_ice = [grid.data for grid in ice_thickness]
        else:
            delta_ice = None
        delta_sea = None
        if initial_sea_level is not None and migrating:
            model.check_field(initial_sea_level)
            delta_sea = initial_sea_level.data

        def C_at(node: int) -> np.ndarray:
            return _effective_ocean_function(snapshots[node]).data if migrating else C0

        def direct_load(node: int) -> Optional[SHGrid]:
            if delta_ice is None and delta_sea is None:
                return None
            load = np.zeros_like(C0)
            if delta_ice is not None:
                load = (
                    rho_i * (1.0 - C_at(node)) * delta_ice[node]
                    - rho_i * (1.0 - C0) * delta_ice[0]
                )
            if delta_sea is not None:
                load = load + rho_w * (C_at(node) - C0) * delta_sea
            return SHGrid.from_array(load, grid=model.grid, copy=False)

        drive = None
        if itau is not None:
            ditau = self._check_itau(itau)
            trajectory = self._as_trajectory(snapshots, dates)
            cursor = _ForwardCursor(self, trajectory, dates, self.scheme)

            def drive(node: int, dt: float) -> dict[int, np.ndarray]:
                return self._first_order_drive(
                    method, dt, cursor.state(node - 1), cursor.state(node), ditau
                )

        return self._tangent_run(
            dates,
            snapshots,
            migrating,
            direct_load,
            drive,
            space=space,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            record=record,
        )

    def _flotation_variations(
        self,
        snapshots: Sequence[SeaLevelSnapshot],
        tangent: Sequence[SeaLevelSnapshot],
        ice: Optional[Sequence[SHGrid]],
        sea_level: Optional[SHGrid],
    ) -> Optional[list[np.ndarray]]:
        """Under a smoothed flotation rule, rho_w K_n delta y_n at every
        node for a direction, with K the snapshots' flotation curvature and
        delta y = delta xi + delta S_0 - (rho_i / rho_w) delta I; None under
        the sharp rule, where K vanishes."""
        curvatures = [snap.flotation_curvature for snap in snapshots]
        if not any(k is not None and np.any(k.data != 0.0) for k in curvatures):
            return None
        parameters = self.model.parameters
        rho_w = parameters.water_density
        ratio = parameters.ice_density / rho_w
        out = []
        for node, (snap, k) in enumerate(zip(tangent, curvatures)):
            y = snap.response.field(RELATIVE_SEA_LEVEL_CHANGE).data.copy()
            if sea_level is not None:
                y += sea_level.data
            if ice is not None:
                y -= ratio * ice[node].data
            out.append(np.zeros_like(y) if k is None else rho_w * k.data * y)
        return out

    def _flotation_second_order_load(
        self, snapshots, tangent_u, tangent_v, ice_u, ice_v, sea_level_u, sea_level_v
    ) -> Callable[[int], Optional[SHGrid]]:
        """The direct load of the second-order run from the shorelines'
        second variation, rho_w K (delta_u y)(delta_v y) less its value at
        the first date, or no load under the sharp rule."""
        a = self._flotation_variations(snapshots, tangent_v, ice_v, sea_level_v)
        if a is None:
            return lambda node: None
        parameters = self.model.parameters
        rho_w = parameters.water_density
        ratio = parameters.ice_density / rho_w
        model = self.model

        def y_u(node: int) -> np.ndarray:
            y = tangent_u[node].response.field(RELATIVE_SEA_LEVEL_CHANGE).data.copy()
            if sea_level_u is not None:
                y += sea_level_u.data
            if ice_u is not None:
                y -= ratio * ice_u[node].data
            return y

        base = a[0] * y_u(0)

        def direct_load(node: int) -> Optional[SHGrid]:
            return SHGrid.from_array(
                a[node] * y_u(node) - base, grid=model.grid, copy=False
            )

        return direct_load

    def second_order_tangent_linear_ice_history(
        self,
        dates: Sequence[float],
        snapshots: Sequence[SeaLevelSnapshot],
        tangent_u: Sequence[SeaLevelSnapshot],
        tangent_v: Sequence[SeaLevelSnapshot],
        *,
        itau_u: Optional[Sequence[np.ndarray]] = None,
        itau_v: Optional[Sequence[np.ndarray]] = None,
        itau_uv: Optional[Sequence[np.ndarray]] = None,
        ice_u: Optional[Sequence[SHGrid]] = None,
        ice_v: Optional[Sequence[SHGrid]] = None,
        sea_level_u: Optional[SHGrid] = None,
        sea_level_v: Optional[SHGrid] = None,
        method: str = "discrete",
        space: Optional[GRDSpace] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        record: bool = False,
    ) -> tuple[list[SeaLevelSnapshot], ViscoelasticState]:
        """The second-order tangent-linear run: the second derivative of a
        run's responses along two perturbations u and v, about the forward
        run whose `snapshots` are given and the tangent-linear runs of u
        and of v about it.

        With the ocean functions held the map is affine in the ice history
        and the initial sea level, so the second derivative is non-zero
        only through the relaxation times: it is the forward problem with
        no direct load, the shorelines prescribed, and the drive of each
        branch shifted by the bilinear term of `tangent_linear_ice_history`
        taken with v's tangent-linear trajectory and u's perturbation,
        with u's trajectory and v's perturbation, with the forward
        trajectory and the second variation `itau_uv` of the
        parameterisation, and, by the discrete method, by the second
        derivatives of the scheme's weights with the forward trajectory
        and both perturbations.  It is symmetric in u and v.

        Args:
            dates, snapshots: As for `tangent_linear_ice_history`; the
                forward run must have recorded its internal states.
            tangent_u, tangent_v: The tangent-linear runs of u and v,
                recorded at every node (`record=True`).
            itau_u, itau_v: The perturbations of the inverse relaxation
                times in u and in v, as for `tangent_linear_ice_history`;
                None for none.
            itau_uv: The second variation of the inverse relaxation times
                along u and v, for a parameterisation that is not linear
                in them, such as itau delta_u chi delta_v chi for the log
                viscosity; None for a linear one.
            ice_u, ice_v, sea_level_u, sea_level_v: The two directions'
                perturbations of the ice history, one grid per node, and of
                the initial sea level, which under a smoothed flotation rule
                enter the second derivative through the shorelines: the
                load gains rho_w K (delta_u y)(delta_v y), with K the
                snapshots' `flotation_curvature` and y = xi + S_0 -
                (rho_i / rho_w) I, less the same at the first date. None
                for none; under the sharp rule they do not enter.
            method, space, rotational_feedbacks, rtol, record: As for
                `tangent_linear_ice_history`.

        Returns:
            The responses of the second derivative at every date, and the
            final internal state.
        """
        if method not in ("discrete", "continuous"):
            raise ValueError(
                f"method must be 'discrete' or 'continuous', got {method!r}"
            )
        dates, snapshots, migrating = self._check_dates_and_snapshots(dates, snapshots)
        parts = []
        if itau_u is not None:
            parts.append(("v", self._check_itau(itau_u)))
        if itau_v is not None:
            parts.append(("u", self._check_itau(itau_v)))
        if itau_uv is not None:
            parts.append(("base", self._check_itau(itau_uv)))
        drive = None
        if parts or (itau_u is not None and itau_v is not None):
            base = _ForwardCursor(
                self, self._as_trajectory(snapshots, dates), dates, self.scheme
            )
            u_states = self._full_trajectory(tangent_u, dates, "tangent_u").states
            v_states = self._full_trajectory(tangent_v, dates, "tangent_v").states
            both = (
                (self._check_itau(itau_u), self._check_itau(itau_v))
                if method == "discrete" and itau_u is not None and itau_v is not None
                else None
            )

            def drive(node: int, dt: float) -> dict[int, np.ndarray]:
                base_prev, base_next = base.state(node - 1), base.state(node)
                pairs = {
                    "base": (base_prev, base_next),
                    "u": (u_states[node - 1], u_states[node]),
                    "v": (v_states[node - 1], v_states[node]),
                }
                terms = [
                    self._first_order_drive(method, dt, *pairs[which], ditau)
                    for which, ditau in parts
                ]
                if both is not None:
                    terms.append(
                        self._second_weight_drive(dt, base_prev, base_next, *both)
                    )
                total: dict[int, np.ndarray] = {}
                for term in terms:
                    for l, value in term.items():
                        total[l] = value if l not in total else total[l] + value
                return total

        direct_load = self._flotation_second_order_load(
            snapshots, tangent_u, tangent_v, ice_u, ice_v, sea_level_u, sea_level_v
        )
        return self._tangent_run(
            dates,
            snapshots,
            migrating,
            direct_load,
            drive,
            space=space,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            record=record,
        )

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
        space: Optional[GRDSpace] = None,
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
            space, rotational_feedbacks, rtol, verbose: As for
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
            space=space,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
        )
        residual = (
            self.state.sea_level.data
            + snapshots[-1].response.field(RELATIVE_SEA_LEVEL_CHANGE).data
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
                    ocean_functions=_effective_ocean_functions(snapshots),
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
        space: Optional[GRDSpace] = None,
        rotational_feedbacks: bool = True,
        solver_rtol: float = 1e-9,
        verbose: bool = False,
    ) -> InitialSeaLevelCalibration:
        """Calibrates the initial sea level so that the run reaches a
        given final sea level, and adopts the calibrated state.

        A thickness-only ice history leaves the initial sea level S0 to
        be found, and the standard device is to choose it so that the
        run reproduces the observed present-day sea level S_f.  Write
        the final sea level as S0 + xi(t_f; S0) and the residual as
        r = S0 + xi(t_f) - S_f.

        Under frozen shorelines the problem is linear: the shorelines
        are those of the background state (with the history's ice at
        the first date), held fixed, and the run does not depend on S0
        at all.  One run then gives the exact answer, S0 <- S_f -
        xi(t_f) wherever the weights are positive, by either method.
        The calibrated state keeps those shorelines as a prescribed
        ocean function, so that later frozen runs from it reproduce
        the target.

        Under migrating shorelines the initial shorelines follow from
        S0 by flotation, and the loads at later dates depend on S0
        through the shorelines that move, so the calibration iterates.
        Two methods are offered:

        - "fixed_point": the classical update S0 <- S0 - r, that is
          S0 <- S_f - xi(t_f; S0), restricted to where the weights are
          positive.  It is Newton's method for S0 + xi(t_f; S0) = S_f
          with the Jacobian I + d xi / d S0 replaced by the identity,
          and contracts at the rate set by the dependence of xi on S0.
        - "gradient": steepest descent on the weighted misfit of
          `final_sea_level_misfit`, S0 <- S0 - step grad J, with the
          gradient from the adjoint (Yu et al. 2025, Section 3.9).
          Unit steps contract more slowly than the classical update,
          which approximately inverts the Jacobian where the gradient
          applies its transpose; it is the method for an objective
          the classical update does not serve, a weighted or partial
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
            step: The step length of either update under migrating
                shorelines; one is the natural value.
            iterations: The most forward runs taken under migrating
                shorelines.
            rtol: The convergence tolerance on the update under
                migrating shorelines.
            anderson_memory: The depth of the acceleration; zero gives
                the plain iteration.
            space, rotational_feedbacks: As for `run_ice_history`.
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

        def rms(residual: np.ndarray) -> float:
            integral = model.integrate(
                SHGrid.from_array(w * residual**2, grid=model.grid, copy=False)
            )
            return float(np.sqrt(integral / total_weight))

        guess = self.state.sea_level.data.copy()
        if shorelines == "frozen":
            # with the shorelines fixed the problem is linear, and the run
            # does not depend on S0 at all: one run gives the misfit's
            # exact minimiser, and the run from it is the run already made
            prescribed = (
                self.state.ocean_function
                if self.state.has_prescribed_ocean_function
                else None
            )
            start = EarthState(
                ice_start,
                SHGrid.from_array(guess, grid=model.grid),
                model,
                land=self.state.land,
                flotation_width=self.state.flotation_width,
                ocean_function=prescribed,
            )
            misfit = self._with_state(start).final_sea_level_misfit(
                dates,
                ice_model,
                final_sea_level,
                weights=weights,
                gradient=False,
                space=space,
                rotational_feedbacks=rotational_feedbacks,
                rtol=solver_rtol,
            )
            residual = misfit.residual.data
            calibrated = guess - support * residual
            state = EarthState(
                ice_start,
                SHGrid.from_array(calibrated, grid=model.grid),
                model,
                land=self.state.land,
                flotation_width=self.state.flotation_width,
                ocean_function=start.ocean_function,
            )
            update = float(np.max(np.abs(support * residual)))
            if verbose:
                print(
                    f"calibration under frozen shorelines: misfit "
                    f"{rms(residual):.3e} removed in one step"
                )
            self.state = state
            return InitialSeaLevelCalibration(
                state,
                misfit.snapshots,
                misfit.internal_state,
                [rms(residual), rms((1.0 - support) * residual)],
                [update / float(np.max(np.abs(calibrated)))],
                True,
                method,
                shorelines,
            )

        mixer = AndersonMixer(anderson_memory)
        misfits: list[float] = []
        updates: list[float] = []
        converged = False
        for n in range(iterations):
            state = EarthState(
                ice_start,
                SHGrid.from_array(guess, grid=model.grid),
                model,
                land=self.state.land,
                flotation_width=self.state.flotation_width,
            )
            misfit = self._with_state(state).final_sea_level_misfit(
                dates,
                ice_model,
                final_sea_level,
                shorelines=shorelines,
                weights=weights,
                gradient=method == "gradient",
                space=space,
                rotational_feedbacks=rotational_feedbacks,
                rtol=solver_rtol,
            )
            misfits.append(rms(misfit.residual.data))
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

    def second_order_adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid | Measurement | GRDResponse],
        tangent: Sequence[SeaLevelSnapshot],
        *,
        itau: Optional[Sequence[np.ndarray]] = None,
        ocean_functions: Optional[Sequence[SHGrid]] = None,
        internal_states: Sequence[ViscoelasticState],
        flotation_curvatures: Optional[Sequence[SHGrid]] = None,
        ice_thickness: Optional[Sequence[SHGrid]] = None,
        initial_sea_level: Optional[SHGrid] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        method: str = "discrete",
    ) -> tuple[IceKernels, KernelDerivatives]:
        """The second-order adjoint: the kernels of an objective and their
        derivative along a perturbation v of the model parameters, the
        action of the objective's Hessian on v.

        The adjoint sweep that gives the kernels is itself a function of
        the parameters, through the forward trajectory and the
        relaxation times; differentiating it along v gives a second sweep
        of the same form, forced not by the measurement but by the first
        sweep's state and v's perturbation of the relaxation times, with
        v's tangent-linear trajectory entering where the forward
        trajectory enters the kernels.  By the discrete method this is
        the exact derivative of the exact gradient, so the Hessian it gives
        is symmetric to rounding; by the continuous method it discretises
        the second-order adjoint equations of Yu et al. (2025), with the
        terms of shoreline motion omitted, the ocean functions being held.

        Args:
            dates, measurement: As for `adjoint_ice_history`.
            tangent: v's tangent-linear run about the forward run,
                recorded at every node (`tangent_linear_ice_history` with
                `record=True`).
            itau: v's perturbation of the inverse relaxation times, nodal
                arrays per branch as for `tangent_linear_ice_history`;
                None for none.
            ocean_functions: The trajectory's effective ocean functions,
                under migrating shorelines.
            internal_states: The recorded forward trajectory, as for
                `adjoint_ice_history`; required here.
            flotation_curvatures: Under a smoothed flotation rule, the
                trajectory's `flotation_curvature` at every node, with
                `ice_thickness` and `initial_sea_level` v's perturbations
                of those arguments: the shorelines' second variation then
                enters as a sea level dual rho_w K delta_v y lambda_n at
                every node and as terms of the thickness and initial sea
                level blocks. None under the sharp rule.
            rotational_feedbacks, rtol: As for `adjoint_ice_history`.
            method: "discrete" or "continuous".

        Returns:
            The kernels, as `adjoint_ice_history` or
            `continuous_adjoint_ice_history` would give them, and their
            derivatives along v.
        """
        if method not in ("discrete", "continuous"):
            raise ValueError(
                f"method must be 'discrete' or 'continuous', got {method!r}"
            )
        if method == "continuous":
            return self._second_order_continuous_sweep(
                dates,
                measurement,
                tangent,
                itau=itau,
                ocean_functions=ocean_functions,
                internal_states=internal_states,
                flotation_curvatures=flotation_curvatures,
                ice_thickness=ice_thickness,
                initial_sea_level=initial_sea_level,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
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
        C = self.state.effective_ocean_function.data
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
        trajectory = self._as_trajectory(internal_states, dates)
        cursor = _TrajectoryCursor(self, trajectory, dates, scheme)
        v_states = self._full_trajectory(tangent, dates, "tangent").states
        ditau = (
            self._check_itau(itau)
            if itau is not None
            else [np.zeros(op.material.mesh.r.shape) for _ in range(K)]
        )
        a_flotation = (
            None
            if flotation_curvatures is None
            else self._flotation_variations(
                [
                    SeaLevelSnapshot(0.0, None, flotation_curvature=k)
                    for k in flotation_curvatures
                ],
                tangent,
                ice_thickness,
                initial_sea_level,
            )
        )

        mdual, ddual, dmdual, dddual = {}, {}, {}, {}
        for l in op.degrees:
            deg = op.degree(l)
            ncols = 2 * (l + 1)
            mdual[l] = np.zeros((K, 3, deg.ne, ngll, ncols))
            dmdual[l] = np.zeros((K, 3, deg.ne, ngll, ncols))
            ddual[l] = np.zeros((3, deg.ne, ngll, ncols))
            dddual[l] = np.zeros((3, deg.ne, ngll, ncols))
        itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]
        d_itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]

        def synth(coeffs: np.ndarray) -> np.ndarray:
            return model.expand_coefficient(
                SHCoeffs.from_array(
                    coeffs, normalization=model.normalization, csphase=model.csphase
                )
            ).data

        def coefficients(c: Optional[np.ndarray]) -> Optional[SHCoeffs]:
            if c is None:
                return None
            return SHCoeffs.from_array(
                c, normalization=model.normalization, csphase=model.csphase
            )

        def pack(values: dict[int, np.ndarray], factor: float) -> np.ndarray:
            out = np.zeros((2, lmax + 1, lmax + 1))
            for l, cols in values.items():
                out[0, l, : l + 1] = cols[: l + 1]
                out[1, l, : l + 1] = cols[l + 1 :]
            return factor * out

        def sle_dual_inputs(sigma_dual, psi_dual, node, *, measured):
            m = duals.get(node) if measured else None
            g_xi = (
                m.relative_sea_level_change.data
                if m is not None and m.relative_sea_level_change is not None
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
            if m is not None and m.angular_velocity_change is not None:
                g_omega = g_omega + np.asarray(m.angular_velocity_change, dtype=float)
            extra = {}
            if m is not None:
                extra["displacement_dual"] = m.displacement
                extra["potential_dual"] = m.potential_change
                extra["horizontal_dual"] = coefficients(m.horizontal)
            return g_xi, g_omega, extra

        def triples(dt: float, deg):
            """Per branch, the scheme weights (E, A, B) of the step, their
            h-derivatives and their second h-derivatives, over the degree's
            sub-mesh, broadcast over the scalar and column axes."""
            weights = op.scheme_weights(scheme, dt)
            first = op.scheme_weight_derivatives(scheme, dt)
            second = op.scheme_weight_second_derivatives(scheme, dt)

            def cut(a):
                return a[deg.e0 :][None, :, :, None]

            out = []
            for k in range(K):
                if implicit:
                    w, dw, d2w = cut(weights[k][0]), cut(first[k][0]), cut(second[k][0])
                    out.append(((w, 0.0, 1.0 - w), (dw, 0.0, -dw), (d2w, 0.0, -d2w)))
                else:
                    out.append(
                        tuple(
                            tuple(cut(a) for a in group)
                            for group in (weights[k], first[k], second[k])
                        )
                    )
            return out

        load_duals = [None] * dates.size
        dload_duals = [None] * dates.size
        for n in range(dates.size - 1, 0, -1):
            dt = (dates[n - 1] - dates[n]) * kyr * self._to_mat_time
            fwd_n = cursor.state(n)
            fwd_prev = cursor.state(n - 1)
            v_n = v_states[n]
            v_prev = v_states[n - 1]
            x_duals, dx_duals = {}, {}
            sig_cols, psi_cols, dsig_cols, dpsi_cols = {}, {}, {}, {}
            per_degree = {}
            for l in op.degrees:
                deg = op.degree(l)
                T = triples(dt, deg)
                dh = [(dt * ditau[k][deg.e0 :])[None, :, :, None] for k in range(K)]
                per_degree[l] = (T, dh)
                summed = ddual[l].copy()
                for k in range(K):
                    (_, _, B), _, _ = T[k]
                    summed += B * mdual[l][k]
                xd = deg.strain_transpose(summed)
                x_duals[l] = xd
                units_lv = op.unit_responses(scheme, dt, l)
                sig_cols[l] = units_lv[:, 0] @ xd
                psi_cols[l] = units_lv[:, 1] @ xd
            sigma_dual = pack(sig_cols, self._to_mat_density)
            psi_dual = pack(psi_cols, self._to_mat_potential)
            g_xi, g_omega, extra = sle_dual_inputs(
                sigma_dual, psi_dual, n, measured=True
            )
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
            m = duals.get(n)
            sv_cols = (
                None
                if m is None or m.horizontal is None
                else self._columns(m.horizontal, self._to_nd_length)
            )
            # the first sweep's radial solve, and from it the second
            # sweep's dual of the step's solution: the derivative of the
            # effective system acting on the first sweep's solution lands
            # on the full solution, so it reaches the load and the tide
            # and passes through the sea level adjoint with the rest
            solves = {}
            for l in op.degrees:
                deg = op.degree(l)
                T, dh = per_degree[l]
                system = op.effective_system(scheme, dt, l)
                rhs = (
                    x_duals[l]
                    + deg.surface_dual("U", su_cols(l, 0.0))
                    + deg.surface_dual("phi", sphi_cols(l, 0.0))
                )
                if sv_cols is not None:
                    rhs = rhs + deg.surface_dual("V", sv_cols(l, 0.0))
                if l == 0:
                    rhs[:, 0] += deg.moment_row * (j_dual * self._to_nd_moment)
                y = system.solve(rhs)
                sy = deg.strain(y)
                wsy = deg.apply_weights(sy)
                solves[l] = (y, sy, wsy)
                dsummed = dddual[l].copy()
                for k in range(K):
                    (_, _, B), (_, _, Bh), _ = T[k]
                    dsummed += B * dmdual[l][k] + Bh * dh[k] * mdual[l][k]
                source = sum(T[k][1][2] * dh[k] * wsy[k] for k in range(K))
                dxd = deg.strain_transpose(dsummed) + deg.dual_from_scalars(
                    source[0], source[1], source[2]
                )
                dx_duals[l] = dxd
                units_lv = op.unit_responses(scheme, dt, l)
                dsig_cols[l] = units_lv[:, 0] @ dxd
                dpsi_cols[l] = units_lv[:, 1] @ dxd
            dsigma_dual = pack(dsig_cols, self._to_mat_density)
            dpsi_dual = pack(dpsi_cols, self._to_mat_potential)
            dg_xi, dg_omega, _ = sle_dual_inputs(
                dsigma_dual, dpsi_dual, n, measured=False
            )
            if a_flotation is not None:
                dg_xi = dg_xi + a_flotation[n] * load_duals[n]
            df_dual, dsu_dual, dsphi_dual, dj_dual = sle.adjoint_generalised_solve(
                self.state,
                sea_level_dual=SHGrid.from_array(dg_xi, grid=model.grid, copy=False),
                angular_velocity_dual=dg_omega,
                ocean_function=C_grid(n),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            dload_duals[n] = df_dual.data + synth(dsigma_dual) / r2
            dsu_cols = self._columns(dsu_dual.coeffs, self._to_nd_length)
            dsphi_cols = self._columns(dsphi_dual.coeffs, self._to_nd_potential)
            for l in op.degrees:
                deg = op.degree(l)
                T, dh = per_degree[l]
                system = op.effective_system(scheme, dt, l)
                y, sy, wsy = solves[l]
                mtilde_dual = mdual[l] + wsy
                drhs = (
                    dx_duals[l]
                    + deg.surface_dual("U", dsu_cols(l, 0.0))
                    + deg.surface_dual("phi", dsphi_cols(l, 0.0))
                )
                if l == 0:
                    drhs[:, 0] += deg.moment_row * (dj_dual * self._to_nd_moment)
                dy = system.solve(drhs)
                dsy = deg.strain(dy)
                dmtilde_dual = dmdual[l] + deg.apply_weights(dsy)
                m_prev = fwd_prev.m[l]
                d0 = fwd_prev.cached_strain[l]
                d1 = fwd_n.cached_strain[l]
                vm_prev = v_prev.m[l]
                vd0 = v_prev.cached_strain[l]
                vd1 = v_n.cached_strain[l]
                Wd1 = deg.apply_weights(d1)
                Wvd1 = deg.apply_weights(vd1)
                new_m = np.empty_like(mdual[l])
                new_dm = np.empty_like(mdual[l])
                new_d = np.zeros_like(ddual[l])
                new_dd = np.zeros_like(ddual[l])
                for k in range(K):
                    (E, A, B), (Eh, Ah, Bh), (Ehh, Ahh, Bhh) = T[k]
                    g = (
                        mtilde_dual[k] * (Eh * m_prev[k] + Ah * d0)
                        + (mdual[l][k] * d1 + Wd1[k] * sy) * Bh
                    ).sum(axis=(0, 3))
                    itau_kernel[k][deg.e0 :] += dt * g
                    dg = (
                        dmtilde_dual[k] * (Eh * m_prev[k] + Ah * d0)
                        + mtilde_dual[k] * (Eh * vm_prev[k] + Ah * vd0)
                        + mtilde_dual[k] * (Ehh * m_prev[k] + Ahh * d0) * dh[k]
                        + (
                            dmdual[l][k] * d1
                            + mdual[l][k] * vd1
                            + Wvd1[k] * sy
                            + Wd1[k] * dsy
                        )
                        * Bh
                        + (mdual[l][k] * d1 + Wd1[k] * sy) * Bhh * dh[k]
                    ).sum(axis=(0, 3))
                    d_itau_kernel[k][deg.e0 :] += dt * dg
                    new_m[k] = E * mtilde_dual[k]
                    new_dm[k] = E * dmtilde_dual[k] + Eh * dh[k] * mtilde_dual[k]
                    new_d += A * mtilde_dual[k]
                    new_dd += A * dmtilde_dual[k] + Ah * dh[k] * mtilde_dual[k]
                mdual[l], dmdual[l], ddual[l], dddual[l] = new_m, new_dm, new_d, new_dd
        # the start: the instantaneous elastic solve at dates[0], whose
        # system does not depend on the relaxation times
        sig_cols, psi_cols, dsig_cols, dpsi_cols = {}, {}, {}, {}
        for l in op.degrees:
            deg = op.degree(l)
            xd = deg.strain_transpose(ddual[l])
            dxd = deg.strain_transpose(dddual[l])
            units_lv = op.unit_responses(None, 0.0, l)
            sig_cols[l] = units_lv[:, 0] @ xd
            psi_cols[l] = units_lv[:, 1] @ xd
            dsig_cols[l] = units_lv[:, 0] @ dxd
            dpsi_cols[l] = units_lv[:, 1] @ dxd
        elastic_sle = self._sle_elastic()
        for which, (sc, pc) in enumerate(
            ((sig_cols, psi_cols), (dsig_cols, dpsi_cols))
        ):
            sigma_dual = pack(sc, self._to_mat_density)
            psi_dual = pack(pc, self._to_mat_potential)
            g_xi, g_omega, extra = sle_dual_inputs(
                sigma_dual, psi_dual, 0, measured=which == 0
            )
            f_dual, _, _, _ = elastic_sle.adjoint_generalised_solve(
                self.state,
                sea_level_dual=SHGrid.from_array(g_xi, grid=model.grid, copy=False),
                angular_velocity_dual=g_omega,
                ocean_function=C_grid(0),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                **extra,
            )
            (load_duals if which == 0 else dload_duals)[0] = (
                f_dual.data + synth(sigma_dual) / r2
            )

        def views(duals_by_node):
            raw = [
                parameters.ice_density * (1.0 - C_at(n)) * duals_by_node[n]
                for n in range(dates.size)
            ]
            baseline = -parameters.ice_density * (1.0 - C) * sum(duals_by_node)
            thickness = list(raw)
            thickness[0] = thickness[0] + baseline
            initial_sea_level = water_density * sum(
                (C_at(n) - C) * duals_by_node[n] for n in range(dates.size)
            )
            return raw, baseline, thickness, initial_sea_level

        raw, baseline, thickness, initial_sea_level = views(load_duals)
        _, _, dthickness, dinitial_sea_level = views(dload_duals)
        if a_flotation is not None:
            dthickness, dinitial_sea_level = self._flotation_blocks(
                a_flotation, load_duals, dthickness, dinitial_sea_level
            )
        instantaneous = {}
        lumps = {}
        for node, m in duals.items():
            f_dual, _, _, _ = elastic_sle.adjoint_generalised_solve(
                self.state,
                sea_level_dual=m.relative_sea_level_change,
                displacement_dual=m.displacement,
                potential_dual=m.potential_change,
                angular_velocity_dual=m.angular_velocity_change,
                horizontal_dual=coefficients(m.horizontal),
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            lumps[node] = parameters.ice_density * (1.0 - C_at(node)) * f_dual.data
            instantaneous[float(dates[node])] = SHGrid.from_array(
                lumps[node], grid=model.grid
            )
        w = np.empty(dates.size)
        w[0] = 0.5 * (dates[0] - dates[1])
        w[-1] = 0.5 * (dates[-2] - dates[-1])
        w[1:-1] = 0.5 * (dates[:-2] - dates[2:])
        density = [(raw[n] - lumps.get(n, 0.0)) / w[n] for n in range(dates.size)]
        rate = []
        tail = np.zeros_like(C)
        for n in range(dates.size - 1, 0, -1):
            tail = tail + raw[n]
            rate.append(tail.copy())
        rate.reverse()
        kernels = IceKernels(
            dates,
            [SHGrid.from_array(ld, grid=model.grid) for ld in load_duals],
            [SHGrid.from_array(th, grid=model.grid) for th in thickness],
            SHGrid.from_array(baseline, grid=model.grid),
            SHGrid.from_array(initial_sea_level, grid=model.grid),
            instantaneous,
            [SHGrid.from_array(d, grid=model.grid) for d in density],
            [SHGrid.from_array(r, grid=model.grid) for r in rate],
            self._kernel_samples(itau_kernel),
        )
        derivatives = KernelDerivatives(
            dates,
            [SHGrid.from_array(th, grid=model.grid) for th in dthickness],
            SHGrid.from_array(dinitial_sea_level, grid=model.grid),
            self._kernel_samples(d_itau_kernel),
        )
        return kernels, derivatives

    def _flotation_blocks(self, a, node_duals, dthickness, dinitial_sea_level):
        """The shorelines' second variation in the thickness and initial sea
        level blocks of the Hessian's action: the field rho_w K delta_v y
        paired with the node duals, times -rho_i / rho_w for the thickness
        at its node, and summed for the initial sea level, less the first
        date's for both."""
        parameters = self.model.parameters
        ratio = parameters.ice_density / parameters.water_density
        total = sum(node_duals)
        dthickness = list(dthickness)
        for n in range(len(node_duals)):
            dthickness[n] = dthickness[n] - ratio * a[n] * node_duals[n]
        dthickness[0] = dthickness[0] + ratio * a[0] * total
        dinitial_sea_level = (
            dinitial_sea_level
            + sum(a[n] * node_duals[n] for n in range(len(node_duals)))
            - a[0] * total
        )
        return dthickness, dinitial_sea_level

    def _second_order_continuous_sweep(
        self,
        dates,
        measurement,
        tangent,
        *,
        itau,
        ocean_functions,
        internal_states,
        flotation_curvatures=None,
        ice_thickness=None,
        initial_sea_level=None,
        rotational_feedbacks,
        rtol,
    ) -> tuple[IceKernels, KernelDerivatives]:
        """The continuous adjoint sweep carrying its own derivative along
        v: the second-order adjoint equations of Yu et al. (2025),
        discretised by the forward stepper as the first-order ones are.
        The derivative state is stepped with the continuous drive of the
        tangent-linear problem built from the adjoint trajectory, jumps at
        each observation by v's perturbation of the inverse relaxation
        times times the instantaneous strain, and the viscosity pairing is
        differentiated with v's tangent-linear trajectory in place of the
        forward one."""
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
        C = self.state.effective_ocean_function.data
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

        K = op.rheology.num_branches
        ngll = op.material.mesh.ngll
        trajectory = self._as_trajectory(internal_states, dates)
        cursor = _TrajectoryCursor(self, trajectory, dates, self.scheme)
        v_states = self._full_trajectory(tangent, dates, "tangent").states
        ditau = (
            self._check_itau(itau)
            if itau is not None
            else [np.zeros(op.material.mesh.r.shape) for _ in range(K)]
        )
        a_flotation = (
            None
            if flotation_curvatures is None
            else self._flotation_variations(
                [
                    SeaLevelSnapshot(0.0, None, flotation_curvature=k)
                    for k in flotation_curvatures
                ],
                tangent,
                ice_thickness,
                initial_sea_level,
            )
        )
        elastic_sle = self._sle_elastic()
        psi_star = centrifugal_potential_operator(model).adjoint

        def jump(m: _Duals, node: int):
            avc = None
            if rotational_feedbacks:
                avc = np.zeros(3)
                if m.angular_velocity_change is not None:
                    avc = avc - g_surface * np.asarray(m.angular_velocity_change, float)
                if m.potential_change is not None:
                    avc = avc + g_surface * psi_star(m.potential_change)
            zeta_u = None if m.displacement is None else -1.0 * m.displacement
            zeta_phi = (
                None if m.potential_change is None else -g_surface * m.potential_change
            )
            zeta_v = m.tangential_load(model)
            xi_a, _, _, omega_a = elastic_sle.solve_generalised_equation(
                self.state,
                direct_load=m.relative_sea_level_change,
                displacement_load=zeta_u,
                gravitational_potential_load=zeta_phi,
                tangential_load=zeta_v,
                angular_momentum_change=avc,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            direct = (
                m.relative_sea_level_change.data
                if m.relative_sea_level_change is not None
                else np.zeros_like(C)
            )
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
            channels = [
                None if zeta is None else model.expand_field(zeta).coeffs
                for zeta in (zeta_u, zeta_phi, zeta_v)
            ]
            return xi_a, lam, psi_a, channels

        def jump_strains(lam, psi_a, channels) -> dict[int, np.ndarray]:
            """The strain of the instantaneous adjoint solution, per
            degree, which jumps the adjoint internal variables over the
            relaxation time (Yu et al. eq. 80) and their derivative by
            v's perturbation of its inverse."""
            lam_cols = self._columns(lam, self._to_mat_density)
            psi_cols = self._columns(psi_a, self._to_mat_potential)
            channel_cols = [
                None if z is None else self._columns(z, self._to_mat_density)
                for z in channels
            ]
            out = {}
            for l in op.degrees:
                deg = op.degree(l)
                units_lv = op.unit_responses(None, 0.0, l)
                x = (
                    units_lv[:, 0][:, None] * lam_cols(l, 0.0)[None, :]
                    + units_lv[:, 1][:, None] * psi_cols(l, 0.0)[None, :]
                )
                if any(cols is not None for cols in channel_cols):
                    split = op.split_unit_responses(None, 0.0, l)
                    for column, cols in enumerate(channel_cols):
                        if cols is not None:
                            x = x + split[:, column][:, None] * cols(l, 0.0)[None, :]
                out[l] = deg.strain(x)
            return out

        def apply_jump(state: ViscoelasticState, strains, factors) -> None:
            for l in state.degrees:
                deg = op.degree(l)
                for k in range(K):
                    state.m[l][k] += factors[k][deg.e0 :][None, :, :, None] * strains[l]
            state.invalidate()

        def smooth_equilibrium(state: ViscoelasticState, node: int):
            solutions = op.solve_elastic(state)
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

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        dstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        sigma_c = np.zeros((2, model.lmax + 1, model.lmax + 1))
        psi_c = np.zeros_like(sigma_c)
        dsigma_c = np.zeros_like(sigma_c)
        dpsi_c = np.zeros_like(sigma_c)
        N = dates.size - 1
        w_early = np.zeros(dates.size)
        w_late = np.zeros(dates.size)
        w_early[1:] = 0.5 * (dates[:-1] - dates[1:])
        w_late[:-1] = 0.5 * (dates[:-1] - dates[1:])
        per_ka = kyr * self._to_mat_time
        itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]
        d_itau_kernel = [np.zeros((op.material.mesh.nspec, ngll)) for _ in range(K)]
        pairing = (
            r2
            * self._to_nd_length
            / (op.material.radius**2 * self._to_mat_density)
            / op.material.surface_gravity
        )

        def accumulate_itau(node: int, weight: float) -> None:
            """Both pairings at a node: the branch-weighted adjoint state
            over itau against the forward d - m, and its derivative, with
            the derivative state, v's tangent-linear trajectory and the
            perturbation of itau each in their place."""
            if weight == 0.0:
                return
            fwd = cursor.state(node)
            tl = v_states[node]
            for l in vstate.degrees:
                deg = op.degree(l)
                dm = fwd.cached_strain[l][None] - fwd.m[l]
                vdm = tl.cached_strain[l][None] - tl.m[l]
                weighted = deg.apply_weights(vstate.m[l])
                dweighted = deg.apply_weights(dstate.m[l])
                for k in range(K):
                    itau_k = deg.itau[k]
                    live = itau_k > 0.0
                    inv = np.where(live, itau_k, 1.0)
                    ratio = np.where(live, ditau[k][deg.e0 :] / inv, 0.0)
                    g = (weighted[k] * dm[k]).sum(axis=(0, 3))
                    itau_kernel[k][deg.e0 :] += weight * np.where(
                        live, pairing * g / inv, 0.0
                    )
                    dg = (dweighted[k] * dm[k] + weighted[k] * vdm[k]).sum(
                        axis=(0, 3)
                    ) - ratio * g
                    d_itau_kernel[k][deg.e0 :] += weight * np.where(
                        live, pairing * dg / inv, 0.0
                    )

        lump_load: dict[int, np.ndarray] = {}
        instantaneous: dict[float, SHGrid] = {}
        density_late = [np.zeros_like(C) for _ in range(dates.size)]
        density_early = [None] * dates.size
        ddensity_late = [np.zeros_like(C) for _ in range(dates.size)]
        ddensity_early = [None] * dates.size

        dlump_load: dict[int, np.ndarray] = {}

        def observe(node: int) -> None:
            """At a reversed node, the measurement's jump for both states
            and, under a smoothed flotation rule, the derivative state's
            jump from the shorelines' second variation, a sea level dual
            rho_w K delta_v y times the node's dual, which stands for a
            source distributed in time under the date quadrature."""
            nonlocal sigma_c, psi_c, dsigma_c, dpsi_c
            late = density_late[node]
            early = late
            jumped = False
            if node in duals:
                xi_a, lam, psi_a, channels = jump(duals[node], node)
                lump_load[node] = xi_a.data
                instantaneous[float(dates[node])] = SHGrid.from_array(
                    grounded_at(node) * xi_a.data, grid=model.grid
                )
                strains = jump_strains(lam, psi_a, channels)
                apply_jump(vstate, strains, op.rheology.itau)
                apply_jump(dstate, strains, ditau)
                early, sigma_c, psi_c = smooth_equilibrium(vstate, node)
                if node > 0:
                    density_early[node] = early
                jumped = True
            if a_flotation is not None and np.any(a_flotation[node] != 0.0):
                node_dual = per_ka * (
                    w_early[node] * early + w_late[node] * late
                ) + lump_load.get(node, 0.0)
                xi_a, lam, psi_a, channels = jump(
                    _Duals(
                        relative_sea_level_change=SHGrid.from_array(
                            a_flotation[node] * node_dual, grid=model.grid
                        )
                    ),
                    node,
                )
                dlump_load[node] = dlump_load.get(node, 0.0) + xi_a.data
                apply_jump(dstate, jump_strains(lam, psi_a, channels), op.rheology.itau)
                jumped = True
            if jumped:
                dearly, dsigma_c, dpsi_c = smooth_equilibrium(dstate, node)
                if node > 0:
                    ddensity_early[node] = dearly

        observe(N)
        accumulate_itau(N, per_ka * w_early[N])
        for n in range(N, 0, -1):
            dt = (dates[n - 1] - dates[n]) * kyr * self._to_mat_time
            load_cols = self._columns(sigma_c, self._to_mat_density)
            tide_cols = self._columns(psi_c, self._to_mat_potential)
            # the adjoint state before the step, with its strains, is the
            # start of the derivative state's drive
            op.prime_strain(vstate, load=load_cols, tide=tide_cols)
            before = vstate
            step = op.start_step(
                vstate, dt, scheme=self.scheme, load=load_cols, tide=tide_cols
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
            # the derivative state's step: the same reversed step, driven
            # by v's perturbation through the adjoint trajectory
            dstep = op.start_step(
                dstate,
                dt,
                scheme=self.scheme,
                load=self._columns(dsigma_c, self._to_mat_density),
                tide=self._columns(dpsi_c, self._to_mat_potential),
            )
            dstep.add_memory(
                self._first_order_drive("continuous", dt, before, vstate, ditau)
            )
            memory_u = self._coeffs(dstep.memory_surface("U"), self._to_nd_length)
            memory_phi = self._coeffs(
                dstep.memory_surface("phi"), self._to_nd_potential
            )
            moment = dstep.inertia_moment()[0] * self._to_nd_moment
            dxi, _, _, domega = sle.solve_generalised_equation(
                self.state,
                displacement_response=memory_u,
                gravitational_potential_response=memory_phi,
                inertia_moment_change=moment,
                ocean_function=C_grid(n - 1),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            ddensity_late[n - 1] = dxi.data
            dsigma_c = model.expand_field(
                SHGrid.from_array(
                    water_density * C_at(n - 1) * dxi.data, grid=model.grid, copy=False
                )
            ).coeffs
            dpsi_c = (
                self._centrifugal(domega)
                if rotational_feedbacks
                else np.zeros_like(dsigma_c)
            )
            dstate, _ = dstep.finish(
                load=self._columns(dsigma_c, self._to_mat_density),
                tide=self._columns(dpsi_c, self._to_mat_potential),
            )
            accumulate_itau(n - 1, per_ka * w_late[n - 1])
            observe(n - 1)
            accumulate_itau(n - 1, per_ka * w_early[n - 1])

        def assemble(late, early, lumps):
            late = [d * per_ka for d in late]
            early = [None if d is None else d * per_ka for d in early]
            node_duals = []
            density_load = []
            for n in range(dates.size):
                e = early[n] if early[n] is not None else late[n]
                node_duals.append(
                    w_early[n] * e + w_late[n] * late[n] + lumps.get(n, 0.0)
                )
                density_load.append(
                    (w_early[n] * e + w_late[n] * late[n]) / (w_early[n] + w_late[n])
                )
            raw = [grounded_at(n) * node_duals[n] for n in range(dates.size)]
            baseline = -parameters.ice_density * (1.0 - C) * sum(node_duals)
            thickness = list(raw)
            thickness[0] = thickness[0] + baseline
            initial_sea_level = water_density * sum(
                (C_at(n) - C) * node_duals[n] for n in range(dates.size)
            )
            return node_duals, density_load, raw, baseline, thickness, initial_sea_level

        load_duals, density_load, raw, baseline, thickness, initial_sea_level = (
            assemble(density_late, density_early, lump_load)
        )
        _, _, _, _, dthickness, dinitial_sea_level = assemble(
            ddensity_late, ddensity_early, dlump_load
        )
        if a_flotation is not None:
            dthickness, dinitial_sea_level = self._flotation_blocks(
                a_flotation, load_duals, dthickness, dinitial_sea_level
            )
        rate = []
        tail = np.zeros_like(C)
        for n in range(dates.size - 1, 0, -1):
            tail = tail + raw[n]
            rate.append(tail.copy())
        rate.reverse()
        kernels = IceKernels(
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
            self._kernel_samples(itau_kernel),
        )
        derivatives = KernelDerivatives(
            dates,
            [SHGrid.from_array(t, grid=model.grid) for t in dthickness],
            SHGrid.from_array(dinitial_sea_level, grid=model.grid),
            self._kernel_samples(d_itau_kernel),
        )
        return kernels, derivatives

    def continuous_adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid | Measurement | GRDResponse],
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

        Measurements are as for `adjoint_ice_history`.  A horizontal
        dual s enters the instantaneous problem, by reciprocity, as the
        tangential traction zeta^v = -s / (b^2 l(l + 1)), the scale
        that makes it stand to the horizontal displacement as the
        traction channel zeta^u = -g_u stands to the vertical.

        With `internal_states` -- the recorded forward trajectory, from
        a run with `record=True` (frozen, or migrating together with
        its `ocean_functions`) -- the kernels also carry the
        continuous sensitivity to the branch inverse relaxation times:
        the multiplier of the internal-variable dynamics is the
        branch-weighted adjoint state over itau, scaled by the ratio of
        the material's load pairing to the model's sea-level pairing
        (see `pairing` below), so that the gradient with respect to the
        nodal values of itau is

            int W_k m-dagger / itau (d - m) dt,

        W_k the branch's nodal weights of the memory pairing,
        accumulated over the reversed nodes with the density's split
        trapezoid (the adjoint state jumps at observations; the forward
        integrand is continuous), and returned as the kernel
        dJ / d itau(r), a density per unit volume (`IceKernels`).  As
        with the ice kernels, the result converges to the discrete one
        as the run resolves its rheology.
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
        C = self.state.effective_ocean_function.data
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

        # the adjoint of the centrifugal potential operator: a potential
        # dual to angular velocity space
        psi_star = centrifugal_potential_operator(model).adjoint

        def jump(m: _Duals, node: int):
            """The instantaneous adjoint fingerprint of a measurement,
            through the forward channel mapping of the elastic adjoint
            (zeta_d = g_xi, zeta^u = -g_u, zeta^phi = -g g_phi,
            zeta^v = -s / (b^2 l(l + 1)), L = -g (g_omega - Psi* g_phi)):
            its sea level (the delta amplitude and the kernel lump), and
            the converged channel coefficients that drive the radial
            jump."""
            avc = None
            if rotational_feedbacks:
                avc = np.zeros(3)
                if m.angular_velocity_change is not None:
                    avc = avc - g_surface * np.asarray(m.angular_velocity_change, float)
                if m.potential_change is not None:
                    avc = avc + g_surface * psi_star(m.potential_change)
            zeta_u = None if m.displacement is None else -1.0 * m.displacement
            zeta_phi = (
                None if m.potential_change is None else -g_surface * m.potential_change
            )
            zeta_v = m.tangential_load(model)
            xi_a, _, _, omega_a = elastic_sle.solve_generalised_equation(
                self.state,
                direct_load=m.relative_sea_level_change,
                displacement_load=zeta_u,
                gravitational_potential_load=zeta_phi,
                tangential_load=zeta_v,
                angular_momentum_change=avc,
                ocean_function=C_grid(node),
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            direct = (
                m.relative_sea_level_change.data
                if m.relative_sea_level_change is not None
                else np.zeros_like(C)
            )
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
            channels = [
                None if zeta is None else model.expand_field(zeta).coeffs
                for zeta in (zeta_u, zeta_phi, zeta_v)
            ]
            return xi_a, lam, psi_a, channels

        def apply_jump(vstate: ViscoelasticState, lam, psi_a, channels) -> None:
            """Yu et al. eq. (80): the strain of the instantaneous
            adjoint solution, over the relaxation time, jumps the
            adjoint internal variables; elastic branches do not move.
            The split load channels and the tangential traction carry
            displacement, potential and horizontal displacement
            measurements into the radial field."""
            lam_cols = self._columns(lam, self._to_mat_density)
            psi_cols = self._columns(psi_a, self._to_mat_potential)
            channel_cols = [
                None if z is None else self._columns(z, self._to_mat_density)
                for z in channels
            ]
            for l in vstate.degrees:
                deg = op.degree(l)
                units_lv = op.unit_responses(None, 0.0, l)
                x = (
                    units_lv[:, 0][:, None] * lam_cols(l, 0.0)[None, :]
                    + units_lv[:, 1][:, None] * psi_cols(l, 0.0)[None, :]
                )
                if any(cols is not None for cols in channel_cols):
                    split = op.split_unit_responses(None, 0.0, l)
                    for column, cols in enumerate(channel_cols):
                        if cols is not None:
                            x = x + split[:, column][:, None] * cols(l, 0.0)[None, :]
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
                xi_a, lam, psi_a, channels = jump(duals[node], node)
                lump_load[node] = xi_a.data
                instantaneous[float(dates[node])] = SHGrid.from_array(
                    grounded_at(node) * xi_a.data, grid=model.grid
                )
                apply_jump(vstate, lam, psi_a, channels)
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
            self._kernel_samples(itau_kernel),
        )

    def adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid | Measurement | GRDResponse],
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

        A measurement is a sea level dual grid, a `Measurement` carrying
        duals of the scalar members and the angular velocity change, or a
        `GRDResponse` standing for its inner product with the response in
        its own space, which carries a dual on the horizontal displacement
        too.  The memory field's V joins V with no Love number, so the
        horizontal dual enters each reversed step's radial solve at the
        surface V, beside the engine's duals of the memory's U and phi.

        With `internal_states` -- the recorded forward trajectory, one
        state per date node from a run with `record=True` (frozen, or
        migrating together with its `ocean_functions`), under the same
        scheme and feedbacks -- the kernels also carry the
        exact discrete sensitivity to the branch inverse relaxation
        times, dJ / d itau(r): per step the derivative of the scheme
        weights in h = dt itau pairs the reverse sweep's duals with the
        forward internal variables and strains, and the effective
        moduli's dependence pairs the transposed and forward solutions
        through the branch bilinear.  The result is a density per unit
        volume sampled at the nodes (`IceKernels`), whose volume pairing
        with a nodal perturbation of itau (`Material.volume_pairing`)
        is the exact derivative of the discrete model.  `viscosity` maps
        it to dJ / d eta(r).
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
        C = self.state.effective_ocean_function.data
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

        def coefficients(c: Optional[np.ndarray]) -> Optional[SHCoeffs]:
            if c is None:
                return None
            return SHCoeffs.from_array(
                c, normalization=model.normalization, csphase=model.csphase
            )

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
                m.relative_sea_level_change.data
                if m is not None and m.relative_sea_level_change is not None
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
            if m is not None and m.angular_velocity_change is not None:
                g_omega = g_omega + np.asarray(m.angular_velocity_change, dtype=float)
            extra = {}
            if m is not None:
                extra["displacement_dual"] = m.displacement
                extra["potential_dual"] = m.potential_change
                extra["horizontal_dual"] = coefficients(m.horizontal)
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
            # the memory's V joins V with no Love number, so the dual of
            # its surface read is the measurement's horizontal dual itself
            m = duals.get(n)
            sv_cols = (
                None
                if m is None or m.horizontal is None
                else self._columns(m.horizontal, self._to_nd_length)
            )
            for l in op.degrees:
                deg = op.degree(l)
                rhs = (
                    x_duals[l]
                    + deg.surface_dual("U", su_cols(l, 0.0))
                    + deg.surface_dual("phi", sphi_cols(l, 0.0))
                )
                if sv_cols is not None:
                    rhs = rhs + deg.surface_dual("V", sv_cols(l, 0.0))
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
                sea_level_dual=m.relative_sea_level_change,
                displacement_dual=m.displacement,
                potential_dual=m.potential_change,
                angular_velocity_dual=m.angular_velocity_change,
                horizontal_dual=coefficients(m.horizontal),
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
            self._kernel_samples(itau_kernel),
        )


def calibrate_initial_sea_level(
    operator: ViscoelasticOperator,
    ice_model: BaseIceModel,
    dates: Sequence[float],
    present_sea_level: SHGrid,
    model: EarthModel,
    *,
    scheme: str = "exponential_trapezoid",
    land: Union[SHGrid, None, str] = "model",
    iterations: int = 4,
    rtol: float = 1e-3,
    anderson_memory: int = 5,
    rotational_feedbacks: bool = True,
    verbose: bool = False,
) -> EarthState:
    """The background state at the first date whose forward run reaches
    the observed present-day sea level, under frozen shorelines.

    A generic ice model carries no sea level, so the initial state must
    be found.  With the shorelines frozen at those of the starting
    state, the present-day sea level under the ice of the first date,
    the problem is linear and one run gives

        SL(t0) = SL(present, observed) - xi(present),

    the returned state keeping those shorelines as a prescribed ocean
    function.  This is a shorthand for
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
        land: The regions held as land in each trial state (see
            `EarthState`): "model", the default, for the ice model's own
            when it has a sea level history (ICE-NG holds the present-day
            landlocked depressions) and none otherwise, a grid of its own,
            or None for flotation alone.
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
    if isinstance(land, str):
        if land != "model":
            raise ValueError(f"land must be 'model', a grid or None, got {land!r}")
        land = (
            ice_model.get_land(
                model.lmax,
                grid=model.grid_name,
                sampling=model.sampling,
                extend=model.extend,
            )
            if isinstance(ice_model, BaseIceSeaLevelModel)
            else None
        )
    guess = EarthState(ice_start, present_sea_level, model, land=land)
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
