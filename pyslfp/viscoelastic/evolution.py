"""The coupled forward GIA time loop under frozen shorelines.

Each implicit or trapezoid step opens on the radial side
(`ViscoelasticOperator.start_step`), which fixes the step's memory
state; its instantaneous response to loads is then an elastic one with
the step's effective Love numbers, so the sea level equation closes
with the existing solver: the memory field enters through the response
channels, its degree-0 inertia moment joins the axial feedback, and
the ocean and rotational couplings converge inside the one linear SLE
solve.  The converged load and centrifugal potential then finish the
radial step, updating the internal variables.  The shorelines are the
background state's throughout (the linear equation); migrating
shorelines are a later rung.

The driver's public quantities -- times, loads, the returned fields --
are in the Earth model's non-dimensional units, like the rest of the
library; the radial operator works in its material's units, and the
driver converts at the boundary between the two.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional
from dataclasses import dataclass, field, replace

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from planetmodel import units

from ..core import EarthModel
from ..ice import BaseIceModel
from ..love_numbers import fluid_tidal_number
from ..core import SeaLevelEquation
from ..core import EarthState
from .operator import ViscoelasticOperator, ViscoelasticState

__all__ = [
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
    instantaneous: dict[float, SHGrid] = field(repr=False)
    density: list[SHGrid] = field(repr=False)
    rate: list[SHGrid] = field(repr=False)


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
    `EarthState` whose ocean function the linear equation freezes.
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
        # conversion factors, material units -> the model's units
        material_scales = operator.material.scales
        sle_scales = parameters.scales

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
        -- and is refused otherwise."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be increasing, with at least two entries")
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
        snapshots = [SeaLevelSnapshot(float(times[0]), slc, disp, potential, omega)]

        for t0, t1 in zip(times[:-1], times[1:]):
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
            snapshots.append(SeaLevelSnapshot(float(t1), slc, disp, potential, omega))
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
        if shorelines == "migrating":
            return self._run_migrating(
                dates,
                ice_model,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
                verbose=verbose,
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

    def _run_migrating(self, dates, ice_model, *, rotational_feedbacks, rtol, verbose):
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

        vstate = op.zero_state(ncols=lambda l: 2 * (l + 1))
        sigma = np.zeros((2, model.lmax + 1, model.lmax + 1))
        psi = np.zeros_like(sigma)
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
            )
        ]
        for a, b in zip(dates[:-1], dates[1:]):
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
            snapshots.append(
                SeaLevelSnapshot(
                    float(vstate.t / self._to_mat_time),
                    slc,
                    disp,
                    potential,
                    omega,
                    date=float(b),
                    ocean_function=final_state.ocean_function,
                )
            )
        return snapshots, vstate

    def continuous_adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid],
        *,
        ocean_functions: Optional[Sequence[SHGrid]] = None,
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
            observe(n - 1)

        # assemble the discrete-equivalent node duals from the density
        # and the lumps, so every gate runs on both methods alike.  The
        # jumps divide by relaxation times in the MATERIAL's units, so
        # the smooth fields are densities per material time; the date
        # quadrature converts them to per kiloyear, and at an
        # observation node the weight splits across the discontinuity:
        # the half-interval towards the present pairs with the late
        # side, the half-interval towards t0 with the early side
        per_ka = kyr * self._to_mat_time
        density_late = [d * per_ka for d in density_late]
        density_early = [None if d is None else d * per_ka for d in density_early]
        w_early = np.zeros(dates.size)
        w_late = np.zeros(dates.size)
        w_early[1:] = 0.5 * (dates[:-1] - dates[1:])
        w_late[:-1] = 0.5 * (dates[:-1] - dates[1:])
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
        baseline = -sum(raw)
        thickness = list(raw)
        thickness[0] = thickness[0] + baseline
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
            instantaneous,
            [
                SHGrid.from_array(grounded_at(n) * density_load[n], grid=model.grid)
                for n in range(dates.size)
            ],
            [SHGrid.from_array(r, grid=model.grid) for r in rate],
        )

    def adjoint_ice_history(
        self,
        dates: Sequence[float],
        measurement: dict[float, SHGrid],
        *,
        ocean_functions: Optional[Sequence[SHGrid]] = None,
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
                mtilde_dual = mdual[l] + deg.apply_weights(deg.strain(y))
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

        # the grounding factor follows the trajectory's shorelines
        raw = [
            parameters.ice_density * (1.0 - C_at(n)) * load_duals[n]
            for n in range(dates.size)
        ]
        baseline = -sum(raw)
        thickness = list(raw)
        thickness[0] = thickness[0] + baseline

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
            instantaneous,
            [SHGrid.from_array(d, grid=model.grid) for d in density],
            [SHGrid.from_array(r, grid=model.grid) for r in rate],
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
    rotational_feedbacks: bool = True,
    verbose: bool = False,
) -> EarthState:
    """The background state at the first date whose forward run reaches
    the observed present-day sea level.

    A generic ice model carries no sea level, so the initial state must
    be found: the standard device is iteration against the present.
    Starting from the present-day sea level as the guess, each pass
    runs the history forward and replaces the initial guess by the
    observed present-day field minus the modelled change,

        SL(t0) <- SL(present, observed) - xi(present; SL(t0)),

    which converges quickly because xi depends on the initial state
    only through the shorelines and the grounding of the load.  The
    ICE-NG models carry their own (inverted) initial sea level and do
    not need this.

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
        verbose: Prints each pass's update.

    Returns:
        The converged background `EarthState` at the first date.
    """
    model.check_field(present_sea_level)
    ice_start = ice_model.get_ice_thickness(
        float(np.asarray(dates, dtype=float)[0]),
        model.lmax,
        grid=model.grid_name,
        sampling=model.sampling,
        extend=model.extend,
    )
    initial = present_sea_level
    state = None
    for n in range(iterations):
        state = EarthState(ice_start, initial, model, exclude_caspian=exclude_caspian)
        evolution = SeaLevelEvolution(operator, state, scheme=scheme)
        snapshots, _ = evolution.run_ice_history(
            dates,
            ice_model,
            rotational_feedbacks=rotational_feedbacks,
            verbose=verbose,
        )
        updated = SHGrid.from_array(
            present_sea_level.data - snapshots[-1].sea_level_change.data,
            grid=model.grid,
        )
        change = np.max(np.abs(updated.data - initial.data))
        scale = np.max(np.abs(updated.data))
        if verbose:
            print(f"calibration pass {n + 1}: relative update {change / scale:.3e}")
        initial = updated
        if change <= rtol * scale:
            return EarthState(
                ice_start, initial, model, exclude_caspian=exclude_caspian
            )
    return EarthState(ice_start, initial, model, exclude_caspian=exclude_caspian)
