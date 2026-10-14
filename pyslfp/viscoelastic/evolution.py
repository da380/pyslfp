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
