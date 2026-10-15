"""The coupled time loop: elastic fingerprint recovery, mass
conservation, convergence order, and the rotational coupling."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyshtools import SHCoeffs, SHGrid

from pyslfp.core import EarthModel
from pyslfp.ice import AnalyticalIceModel, BaseIceModel
from pyslfp.core import SeaLevelEquation
from pyslfp.core import EarthState
from pyslfp.love_numbers import Material, fluid_tidal_number, love_numbers
from pyslfp.viscoelastic import (
    Measurement,
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
    calibrate_initial_sea_level,
)

# the test body lives in the geophysical non-dimensionalisation: unit
# radius, near-reference density, Earth-like moduli, G equal to one
SCALES = units.Scales.geophysical(6.371e6)
RHO, VS, VP = 1.0, 1.3, 3.1
LMAX = 4
TAU = 6.0e6  # the Maxwell time, in the material's units (about 310 years)


@pytest.fixture(scope="module")
def setting():
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[RHO], vp=[VP], vs=[VS], scales=SCALES
    )
    mesh = RadialMesh(model, ngll=5, lmax=LMAX)
    material = Material(mesh, model)
    # the homogeneous body sustains far less oblateness than Earth's
    # moments imply (its fluid tidal number is Kelvin's 3/2), so the
    # transverse feedback needs the hydrostatic-moment construction
    base = EarthModel.from_planet_model(model, LMAX, mesh=mesh)
    parameters = base.parameters.with_hydrostatic_moments(
        fluid_tidal_number(material, 2)
    )
    earth = EarthModel(
        LMAX, parameters=parameters, love_numbers=love_numbers(material, LMAX)
    )
    ice = earth.zero_grid()
    sea = earth.constant_grid(3000.0 / earth.parameters.length_scale)
    state = EarthState(ice, sea, earth, exclude_caspian=False)
    lats = state.model.lats()[:, None] * np.ones_like(state.model.lons())[None, :]
    thickness = 100.0 / earth.parameters.length_scale
    cap = np.where(lats > 45.0, earth.parameters.ice_density * thickness, 0.0)
    load = SHGrid.from_array(cap * np.ones_like(ice.data), grid=earth.grid, copy=False)
    # band-limit the sharp cap so every solver sees the same field
    load = state.model.expand_coefficient(state.model.expand_field(load))
    # the Maxwell time in the earth model's time unit: a pure ratio of
    # the two non-dimensionalisations
    tau_nd = (
        TAU
        * material.scales.factor(units.TIME)
        / earth.parameters.scales.factor(units.TIME)
    )
    return material, earth, state, load, tau_nd


def _maxwell_operator(material):
    return ViscoelasticOperator(Rheology.from_relaxation_time(material, TAU), LMAX)


def _elastic_rheology(material):
    return Rheology(
        material,
        [
            (
                np.asarray(material.moduli.kappa_mu()[1]).real,
                np.zeros(material.moduli.shape),
            )
        ],
    )


def test_elastic_fingerprint_is_recovered(setting):
    """With an infinite relaxation time everywhere the evolution's every
    snapshot is the elastic fingerprint of the same load."""
    material, earth, state, load, _ = setting
    rheo = _elastic_rheology(material)
    evo = SeaLevelEvolution(ViscoelasticOperator(rheo, LMAX), state)
    snapshots, _ = evo.run([0.0, 0.5, 1.0], lambda t: load)
    want_slc, want_disp, want_phi, want_omega = SeaLevelEquation(
        earth
    ).solve_sea_level_equation(state, load)
    for snap in snapshots:
        scale = np.max(np.abs(want_slc.data))
        assert np.max(np.abs(snap.sea_level_change.data - want_slc.data)) < 1e-8 * scale
        scale = np.max(np.abs(want_disp.data))
        assert np.max(np.abs(snap.displacement.data - want_disp.data)) < 1e-8 * scale
        assert np.allclose(
            snap.angular_velocity_change, want_omega, rtol=1e-8, atol=1e-30
        )


def test_mass_is_conserved_and_the_response_creeps(setting):
    material, earth, state, load, tau_nd = setting
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    times = np.linspace(0.0, 10.0 * tau_nd, 6)
    snapshots, vstate = evo.run(times, lambda t: load)
    ice_mass = earth.integrate(load)
    water = earth.parameters.water_density
    ocean = state.ocean_function.data
    grid = earth.grid
    for snap in snapshots:
        added = water * earth.integrate(
            SHGrid.from_array(ocean * snap.sea_level_change.data, grid=grid)
        )
        assert abs(added + ice_mass) < 1e-9 * abs(ice_mass)
    # the solid flows: subsidence under the held load deepens with time
    first = np.max(np.abs(snapshots[0].displacement.data))
    last = np.max(np.abs(snapshots[-1].displacement.data))
    assert last > 1.5 * first
    assert np.all(np.isfinite(vstate.m[2]))


def test_coupled_loop_is_second_order(setting):
    material, _, state, load, tau_nd = setting
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    T = 5.0 * tau_nd

    def final_slc(n):
        snapshots, _ = evo.run(np.linspace(0.0, T, n + 1), lambda t: load)
        return snapshots[-1].sea_level_change.data

    reference = final_slc(64)
    e1 = np.max(np.abs(final_slc(4) - reference))
    e2 = np.max(np.abs(final_slc(8) - reference))
    order = np.log2(e1 / e2)
    assert abs(order - 2.0) < 0.4, order


class _MeltingCap(BaseIceModel):
    """A thickness-only ice history: a 5 km polar cap, complete at 10 ka
    and melting linearly to nothing at present."""

    START = 10.0  # ka
    THICKNESS = 5000.0  # m

    def get_ice_thickness(self, date, lmax, /, *, grid="DH", sampling=1, extend=True):
        grid, sampling = self.resolve_grid(grid, sampling)
        cap = SHGrid.from_cap(
            15.0, 60.0, -85.0, lmax=lmax, grid=grid, sampling=sampling, extend=extend
        )
        fraction = np.clip(date / self.START, 0.0, 1.0)
        return (self.THICKNESS / self.length_scale) * fraction * cap


def _history_state(material, earth):
    """The analytical testing geometry at 10 ka: smooth continents and
    a melting mid-latitude ice sheet (the house surface state for
    low-degree tests, Gibbs-free by construction), over whatever
    radial body the caller chose."""
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    state = EarthState.from_ice_model(ice_model, 10.0, earth, exclude_caspian=False)
    return ice_model, state


def _marine_history_state(material, earth):
    """The marine-grounded cap on a uniform ocean: non-smooth by
    design, so that melting actually flips flotation cells -- the
    scenario the migrating-shoreline gates need, which the smooth
    analytical geometry cannot produce at these degrees."""
    ice_model = _MeltingCap(length_scale=earth.parameters.length_scale)
    ice = ice_model.get_ice_thickness(
        _MeltingCap.START, LMAX, grid=earth.grid_name, extend=earth.extend
    )
    sea = earth.constant_grid(3000.0 / earth.parameters.length_scale)
    return ice_model, EarthState(ice, sea, earth, exclude_caspian=False)


def test_ice_history_run(setting):
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    assert not state.ocean_function.data.all()  # the cap is grounded
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 6)
    snapshots, _ = evo.run_ice_history(dates, ice_model)
    assert np.allclose([snap.date for snap in snapshots], dates)
    assert snapshots[0].t == 0.0
    # melting grounded ice raises the ocean mean by its eustatic
    # equivalent, up to the solid earth's and rotation's redistribution
    parameters = earth.parameters
    melted = (
        ice_model.get_ice_thickness(
            10.0, LMAX, grid=earth.grid_name, extend=earth.extend
        ).data
        - ice_model.get_ice_thickness(
            0.0, LMAX, grid=earth.grid_name, extend=earth.extend
        ).data
    )
    expected = (
        parameters.ice_density
        / parameters.water_density
        * earth.integrate(
            SHGrid.from_array(
                (1.0 - state.ocean_function.data) * melted, grid=earth.grid
            )
        )
        / state.ocean_area
    )
    mean = (
        earth.integrate(snapshots[-1].sea_level_change * state.ocean_function)
        / state.ocean_area
    )
    assert mean > 0.0
    assert abs(mean / expected - 1.0) < 0.3
    with pytest.raises(ValueError, match="decreasing"):
        evo.run_ice_history(dates[::-1], ice_model)
    with pytest.raises(ValueError, match="length scale"):
        evo.run_ice_history(dates, _MeltingCap(length_scale=1.0))


def test_inconsistent_moments_are_refused(setting):
    """Earth's observed moments sustain less oblateness than the
    homogeneous body's fluid limit: the transverse feedback is refused
    with a pointer at the remedies, and the axial-only mode runs."""
    material, earth, _, _, _ = setting
    model = material.model
    raw_earth = EarthModel.from_planet_model(model, LMAX, mesh=material.mesh)
    ice_model, state = _marine_history_state(material, raw_earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 3)
    with pytest.raises(ValueError, match="oblateness"):
        evo.run_ice_history(dates, ice_model)
    snapshots, _ = evo.run_ice_history(dates, ice_model, rotational_feedbacks="axial")
    omega = snapshots[-1].angular_velocity_change
    assert omega[0] == 0.0 and omega[1] == 0.0 and omega[2] != 0.0
    assert np.max(np.abs(snapshots[-1].sea_level_change.data)) < 1.0  # bounded


def test_initial_sea_level_calibration(setting):
    """The round trip: a forward run from a known initial sea level
    gives the 'observed' present; the calibration recovers the initial
    field from that observation alone."""
    material, earth, _, _, _ = setting
    ice_model, truth = _history_state(material, earth)
    operator = _maxwell_operator(material)
    dates = np.linspace(10.0, 0.0, 6)
    snapshots, _ = SeaLevelEvolution(operator, truth).run_ice_history(dates, ice_model)
    observed = SHGrid.from_array(
        truth.sea_level.data + snapshots[-1].sea_level_change.data, grid=earth.grid
    )
    calibrated = calibrate_initial_sea_level(
        operator,
        ice_model,
        dates,
        observed,
        earth,
        exclude_caspian=False,
        iterations=5,
        rtol=1e-6,
    )
    worst = np.max(np.abs(calibrated.sea_level.data - truth.sea_level.data))
    scale = np.max(np.abs(truth.sea_level.data))
    assert worst < 1e-5 * scale


class _ThinningCap(BaseIceModel):
    """The 5 km cap losing just one metre over the history: every cell
    stays grounded, so the shorelines cannot move."""

    def get_ice_thickness(self, date, lmax, /, *, grid="DH", sampling=1, extend=True):
        grid, sampling = self.resolve_grid(grid, sampling)
        cap = SHGrid.from_cap(
            15.0, 60.0, -85.0, lmax=lmax, grid=grid, sampling=sampling, extend=extend
        )
        fraction = np.clip(date / _MeltingCap.START, 0.0, 1.0)
        thickness = _MeltingCap.THICKNESS - (1.0 - fraction)
        return (thickness / self.length_scale) * cap


def test_migrating_matches_frozen_when_shorelines_cannot_move(setting):
    material, earth, _, _, _ = setting
    scale = earth.parameters.length_scale
    ice_model = _ThinningCap(length_scale=scale)
    ice = earth.zero_grid()
    ice.data[:, :] = ice_model.get_ice_thickness(
        _MeltingCap.START, LMAX, grid=earth.grid_name, extend=earth.extend
    ).data
    sea = earth.constant_grid(3000.0 / scale)
    state = EarthState(ice, sea, earth, exclude_caspian=False)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 5)
    frozen, _ = evo.run_ice_history(dates, ice_model, shorelines="frozen")
    moving, _ = evo.run_ice_history(dates, ice_model, shorelines="migrating")
    for f, m in zip(frozen, moving):
        assert np.array_equal(
            m.ocean_function.data, state.ocean_function.data
        )  # nothing moved
        top = np.max(np.abs(f.sea_level_change.data))
        diff = np.max(np.abs(f.sea_level_change.data - m.sea_level_change.data))
        assert diff <= 1e-6 * max(top, 1e-30), f.date


def test_migrating_shorelines_move_and_conserve_mass(setting):
    """The melting 5 km cap floods: its grounded region becomes ocean,
    and the explicit water budget balances the grounded ice loss."""
    material, earth, _, _, _ = setting
    ice_model, state = _marine_history_state(material, earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 5)
    snapshots, _ = evo.run_ice_history(dates, ice_model, shorelines="migrating")
    C0 = state.ocean_function.data
    Cf = snapshots[-1].ocean_function.data
    assert earth.integrate(SHGrid.from_array(Cf - C0, grid=earth.grid)) > 0.0
    parameters = earth.parameters
    water = parameters.water_density * earth.integrate(
        SHGrid.from_array(
            Cf * (state.sea_level.data + snapshots[-1].sea_level_change.data)
            - C0 * state.sea_level.data,
            grid=earth.grid,
        )
    )
    grounded = parameters.ice_density * earth.integrate(
        SHGrid.from_array(
            (1.0 - Cf) * 0.0 - (1.0 - C0) * state.ice_thickness.data,
            grid=earth.grid,
        )
    )
    assert abs(water + grounded) < 1e-8 * abs(grounded)
    # and the frozen run misses the marine ingress
    frozen, _ = evo.run_ice_history(dates, ice_model, shorelines="frozen")
    diff = np.max(
        np.abs(frozen[-1].sea_level_change.data - snapshots[-1].sea_level_change.data)
    )
    assert diff > 1e-3 * np.max(np.abs(snapshots[-1].sea_level_change.data))


def test_migrating_elastic_limit(setting):
    """With an infinite relaxation time the migrating evolution is the
    elastic nonlinear solver applied date by date."""
    material, earth, _, _, _ = setting
    rheo = _elastic_rheology(material)
    ice_model, state = _history_state(material, earth)
    evo = SeaLevelEvolution(ViscoelasticOperator(rheo, LMAX), state)
    dates = np.linspace(10.0, 0.0, 4)
    snapshots, _ = evo.run_ice_history(dates, ice_model, shorelines="migrating")
    sle = SeaLevelEquation(earth)
    for snap in snapshots[1:]:
        change = SHGrid.from_array(
            ice_model.get_ice_thickness(
                snap.date, LMAX, grid=earth.grid_name, extend=earth.extend
            ).data
            - state.ice_thickness.data,
            grid=earth.grid,
        )
        _, want, _, _, _ = sle.solve_nonlinear_equation(
            state, ice_thickness_change=change
        )
        top = np.max(np.abs(want.data))
        assert np.max(np.abs(snap.sea_level_change.data - want.data)) < 1e-7 * top


def test_rotation_feeds_back_and_evolves(setting):
    material, _, state, load, tau_nd = setting
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    times = np.linspace(0.0, 6.0 * tau_nd, 4)
    snapshots, _ = evo.run(times, lambda t: load)
    omega_first = snapshots[0].angular_velocity_change
    omega_last = snapshots[-1].angular_velocity_change
    assert np.all(np.isfinite(omega_first)) and np.any(omega_first != 0.0)
    # the axial rate evolves as the load compensates isostatically
    assert abs(omega_last[2] - omega_first[2]) > 1e-3 * abs(omega_first[2])
    off, _ = evo.run(times, lambda t: load, rotational_feedbacks=False)
    assert np.all(off[-1].angular_velocity_change == 0.0)
    difference = np.max(
        np.abs(off[-1].sea_level_change.data - snapshots[-1].sea_level_change.data)
    )
    assert difference > 1e-6 * np.max(np.abs(snapshots[-1].sea_level_change.data))


def test_discrete_adjoint_gradient_identity(setting):
    """The discrete adjoint's kernels predict the change of a sea-level
    functional under ANY ice perturbation exactly (frozen shorelines:
    the forward map is linear), feedbacks on and axial-only."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    operator = _maxwell_operator(material)
    evo = SeaLevelEvolution(operator, state)
    dates = np.linspace(10.0, 0.0, 4)
    rng = np.random.default_rng(11)
    scale = earth.parameters.length_scale

    def smooth_grid(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        from pyshtools import SHCoeffs

        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    deltas = {float(d): smooth_grid(10.0 / scale) for d in dates}

    class _Perturbed(BaseIceModel):
        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            return SHGrid.from_array(
                base.data + deltas[float(date)].data, grid=base.grid
            )

    perturbed = _Perturbed(length_scale=scale)
    measurement = {
        float(dates[-1]): smooth_grid(1.0),
        float(dates[1]): smooth_grid(0.5),
    }

    for mode in (True, "axial"):

        def functional(model_in):
            snapshots, _ = evo.run_ice_history(
                dates, model_in, rotational_feedbacks=mode, rtol=1e-12
            )
            total = 0.0
            for snap in snapshots:
                for key, g in measurement.items():
                    if np.isclose(snap.date, key):
                        total += earth.integrate(
                            SHGrid.from_array(
                                g.data * snap.sea_level_change.data, grid=earth.grid
                            )
                        )
            return total

        exact = functional(perturbed) - functional(ice_model)
        kernels = evo.adjoint_ice_history(
            dates, measurement, rotational_feedbacks=mode, rtol=1e-12
        )
        predicted = sum(
            earth.integrate(
                SHGrid.from_array(
                    kernels.thickness[n].data * deltas[float(dates[n])].data,
                    grid=earth.grid,
                )
            )
            for n in range(dates.size)
        )
        assert abs(predicted - exact) < 1e-6 * abs(exact), (mode, exact, predicted)


def test_kernel_singular_parts_are_separated(setting):
    """The distributional view of the ice kernels (Yu et al. 2025,
    Section 3.7): under date-grid refinement the continuous density
    converges, the raw node duals merely scale with the time weights,
    and the delta amplitudes (the instantaneous lumps and the t0
    baseline) are grid-independent."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    rng = np.random.default_rng(2)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    g = earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )
    measurement = {0.0: g}
    coarse = evo.adjoint_ice_history(np.linspace(10.0, 0.0, 5), measurement, rtol=1e-11)
    fine = evo.adjoint_ice_history(np.linspace(10.0, 0.0, 9), measurement, rtol=1e-11)
    # the shared date 5.0 ka sits at index 2 (coarse) and 4 (fine)
    scale = np.max(np.abs(fine.density[4].data))
    assert np.max(np.abs(coarse.density[2].data - fine.density[4].data)) < 0.12 * scale
    ratio = np.max(np.abs(fine.thickness[4].data)) / np.max(
        np.abs(coarse.thickness[2].data)
    )
    assert 0.4 < ratio < 0.6  # the raw node dual scales with the weight
    lump = np.max(np.abs(fine.instantaneous[0.0].data))
    assert (
        np.max(np.abs(coarse.instantaneous[0.0].data - fine.instantaneous[0.0].data))
        < 1e-12 * lump
    )
    top = np.max(np.abs(fine.baseline.data))
    assert np.max(np.abs(coarse.baseline.data - fine.baseline.data)) < 0.05 * top


def test_quelled_rate_kernels(setting):
    """Backus quelling: the kernels with respect to the thickness RATE
    are the cumulative tails of the node duals -- bounded, convergent
    under refinement, exact against rate perturbations (which generate
    thickness histories continuous in time), and blind to a uniform
    thickness shift."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    rng = np.random.default_rng(13)
    scale = earth.parameters.length_scale
    dates = np.linspace(10.0, 0.0, 5)

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    measurement = {0.0: smooth(1.0)}
    kernels = evo.adjoint_ice_history(dates, measurement, rtol=1e-12)

    # blind to a uniform shift: the node duals sum to zero exactly
    total = sum(t.data for t in kernels.thickness)
    assert np.max(np.abs(total)) <= 1e-12 * np.max(np.abs(kernels.thickness[-1].data))

    # exact against per-interval rate perturbations
    h = dates[:-1] - dates[1:]
    rates = [smooth(2.0 / scale) for _ in range(h.size)]
    cumulative = [np.zeros_like(rates[0].data)]
    for k in range(h.size):
        cumulative.append(cumulative[-1] + h[k] * rates[k].data)

    class _RatePerturbed(BaseIceModel):
        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            node = int(np.flatnonzero(np.isclose(dates, date))[0])
            return SHGrid.from_array(base.data + cumulative[node], grid=base.grid)

    def functional(model_in):
        snapshots, _ = evo.run_ice_history(dates, model_in, rtol=1e-12)
        g = measurement[0.0]
        return earth.integrate(
            SHGrid.from_array(
                g.data * snapshots[-1].sea_level_change.data, grid=earth.grid
            )
        )

    exact = functional(_RatePerturbed(length_scale=scale)) - functional(ice_model)
    predicted = sum(
        h[k]
        * earth.integrate(
            SHGrid.from_array(kernels.rate[k].data * rates[k].data, grid=earth.grid)
        )
        for k in range(h.size)
    )
    assert abs(predicted - exact) < 1e-6 * abs(exact), (exact, predicted)

    # refinement: a coarse interval's value is the mean of its halves
    fine = evo.adjoint_ice_history(np.linspace(10.0, 0.0, 9), measurement, rtol=1e-12)
    pair_mean = 0.5 * (fine.rate[2].data + fine.rate[3].data)
    top = np.max(np.abs(pair_mean))
    assert np.max(np.abs(kernels.rate[1].data - pair_mean)) < 0.05 * top


def test_continuous_adjoint_against_the_discrete(setting):
    """The two adjoints compared (the deliverable of the comparison):
    the discrete transpose is exact at any step; the continuous
    adjoint, discretised, carries the commute error -- negligible and
    high-order once the steps resolve the relaxation, first order at
    the per-cent level when they do not.  Their delta amplitudes agree
    although computed by different realisations (the engine transpose
    against the forward channel mapping), cross-validating both."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rng = np.random.default_rng(4)
    scale = earth.parameters.length_scale

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    measurement = {0.0: smooth(1.0), 5.0: smooth(0.5)}
    deltas = {float(d): smooth(10.0 / scale) for d in np.linspace(10.0, 0.0, 9)}

    class _Perturbed(BaseIceModel):
        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            return SHGrid.from_array(
                base.data + deltas[float(date)].data, grid=base.grid
            )

    perturbed = _Perturbed(length_scale=scale)

    def gradient_errors(operator, npts):
        evo = SeaLevelEvolution(operator, state)
        dates = np.linspace(10.0, 0.0, npts)

        def functional(model_in):
            snapshots, _ = evo.run_ice_history(dates, model_in, rtol=1e-12)
            total = 0.0
            for snap in snapshots:
                for key, g in measurement.items():
                    if np.isclose(snap.date, key):
                        total += earth.integrate(
                            SHGrid.from_array(
                                g.data * snap.sea_level_change.data, grid=earth.grid
                            )
                        )
            return total

        exact = functional(perturbed) - functional(ice_model)
        errors = {}
        kernel_sets = {}
        for name, method in (
            ("discrete", evo.adjoint_ice_history),
            ("continuous", evo.continuous_adjoint_ice_history),
        ):
            kernels = method(dates, measurement, rtol=1e-12)
            predicted = sum(
                earth.integrate(
                    SHGrid.from_array(
                        kernels.thickness[n].data * deltas[float(dates[n])].data,
                        grid=earth.grid,
                    )
                )
                for n in range(dates.size)
            )
            errors[name] = abs(predicted - exact) / abs(exact)
            kernel_sets[name] = kernels
        return errors, kernel_sets

    # the stiff regime: steps of several relaxation times
    stiff = _maxwell_operator(material)
    coarse, kernel_sets = gradient_errors(stiff, 5)
    fine, _ = gradient_errors(stiff, 9)
    assert coarse["discrete"] < 1e-9 and fine["discrete"] < 1e-9
    assert coarse["continuous"] < 0.5
    assert coarse["continuous"] / fine["continuous"] > 1.8  # converging

    # the delta amplitudes cross-validate the two realisations
    for key in measurement:
        a = kernel_sets["discrete"].instantaneous[key].data
        b = kernel_sets["continuous"].instantaneous[key].data
        assert np.max(np.abs(a - b)) < 1e-8 * np.max(np.abs(a)), key

    # the resolved regime: relaxation ten times slower; the commute
    # error is small and falls at least quadratically under refinement
    slow = ViscoelasticOperator(
        Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX
    )
    resolved, _ = gradient_errors(slow, 5)
    finer, _ = gradient_errors(slow, 9)
    assert resolved["discrete"] < 1e-9 and finer["discrete"] < 1e-9
    assert resolved["continuous"] < 1e-2
    assert resolved["continuous"] / finer["continuous"] > 2.5


def test_migrating_trajectory_kernels(setting):
    """Kernels about a migrating-shoreline trajectory: the adjoint
    reads the stored C(t) (Yu et al. 2025, Section 3.3) and stays
    linear.  With a constant history both methods reduce exactly to
    the frozen kernels; about the moving trajectory the discrete
    transpose matches central differences to solver precision (the
    forward is the linearised map while no cell flips), and the
    continuous method converges at second order once the rheology is
    resolved -- their disagreement on an under-resolved run is the
    resolution diagnostic, not an error of either."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _marine_history_state(material, earth)
    rng = np.random.default_rng(6)
    scale = earth.parameters.length_scale

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    measurement = {0.0: smooth(1.0)}
    dates = np.linspace(10.0, 0.0, 5)
    stiff = SeaLevelEvolution(_maxwell_operator(material), state)

    # a constant ocean-function history is the frozen calculation
    frozen = stiff.adjoint_ice_history(dates, measurement, rtol=1e-12)
    constant = stiff.adjoint_ice_history(
        dates,
        measurement,
        ocean_functions=[state.ocean_function] * dates.size,
        rtol=1e-12,
    )
    for a, b in zip(frozen.thickness, constant.thickness):
        assert np.array_equal(a.data, b.data)

    deltas = {float(d): smooth(1.0 / scale) for d in dates}
    deltas[float(dates[0])] = 0.0 * deltas[float(dates[0])]

    class _Perturbed(BaseIceModel):
        sign = 1.0

        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            return SHGrid.from_array(
                base.data + _Perturbed.sign * deltas[float(date)].data,
                grid=base.grid,
            )

    def gradient_check(evolution, npts):
        grid_dates = np.linspace(10.0, 0.0, npts)
        snapshots, _ = evolution.run_ice_history(
            grid_dates, ice_model, shorelines="migrating", rtol=1e-12
        )
        history = [snap.ocean_function for snap in snapshots]

        def functional(model_in):
            snaps, _ = evolution.run_ice_history(
                grid_dates, model_in, shorelines="migrating", rtol=1e-12
            )
            g = measurement[0.0]
            return earth.integrate(
                SHGrid.from_array(
                    g.data * snaps[-1].sea_level_change.data, grid=earth.grid
                )
            )

        _Perturbed.sign = 1.0
        plus = functional(_Perturbed(length_scale=scale))
        _Perturbed.sign = -1.0
        minus = functional(_Perturbed(length_scale=scale))
        fd = 0.5 * (plus - minus)
        errors = {}
        for name, method in (
            ("discrete", evolution.adjoint_ice_history),
            ("continuous", evolution.continuous_adjoint_ice_history),
        ):
            kernels = method(
                grid_dates, measurement, ocean_functions=history, rtol=1e-12
            )
            predicted = sum(
                earth.integrate(
                    SHGrid.from_array(
                        kernels.thickness[n].data * deltas[float(grid_dates[n])].data,
                        grid=earth.grid,
                    )
                )
                for n in range(grid_dates.size)
            )
            errors[name] = abs(predicted - fd) / abs(fd)
        return errors

    # deltas exist only on the coarse date set; reuse them on refinement
    for d in np.linspace(10.0, 0.0, 9):
        deltas.setdefault(float(d), 0.0 * deltas[float(dates[1])])

    stiff_errors = gradient_check(stiff, 5)
    assert stiff_errors["discrete"] < 1e-8

    slow = SeaLevelEvolution(
        ViscoelasticOperator(Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX),
        state,
    )
    coarse = gradient_check(slow, 5)
    fine = gradient_check(slow, 9)
    assert coarse["discrete"] < 1e-8 and fine["discrete"] < 1e-8
    assert coarse["continuous"] < 3e-2
    assert coarse["continuous"] / fine["continuous"] > 2.0


def test_full_measurement_kernels(setting):
    """Measurements of all four outputs -- sea level, displacement,
    potential and rotation -- drive both adjoints: the discrete
    transpose is exact (the engine adjoint carries the duals; no new
    radial channels), and the continuous method's jump realises the
    channel mapping of the elastic adjoint with the split radial load
    responses, converging in the resolved regime."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rng = np.random.default_rng(9)
    scale = earth.parameters.length_scale

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    measurement = {
        0.0: Measurement(
            sea_level=smooth(1.0),
            displacement=smooth(0.7),
            gravity_potential=smooth(0.4),
            angular_velocity=rng.normal(size=3) * 1e3,
        ),
        5.0: smooth(0.5),
    }
    deltas = {float(d): smooth(10.0 / scale) for d in np.linspace(10.0, 0.0, 5)}

    class _Perturbed(BaseIceModel):
        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            return SHGrid.from_array(
                base.data + deltas[float(date)].data, grid=base.grid
            )

    def objective(snapshots):
        total = 0.0
        for snap in snapshots:
            for key, m in measurement.items():
                if not np.isclose(snap.date, key):
                    continue
                if not isinstance(m, Measurement):
                    m = Measurement(sea_level=m)
                pairs = (
                    (m.sea_level, snap.sea_level_change),
                    (m.displacement, snap.displacement),
                    (m.gravity_potential, snap.gravity_potential_change),
                )
                for dual, field_grid in pairs:
                    if dual is not None:
                        total += earth.integrate(
                            SHGrid.from_array(
                                dual.data * field_grid.data, grid=earth.grid
                            )
                        )
                if m.angular_velocity is not None:
                    total += float(
                        np.asarray(m.angular_velocity) @ snap.angular_velocity_change
                    )
        return total

    dates = np.linspace(10.0, 0.0, 5)
    for operator, continuous_bound in (
        (_maxwell_operator(material), None),
        (
            ViscoelasticOperator(
                Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX
            ),
            2e-3,
        ),
    ):
        evo = SeaLevelEvolution(operator, state)

        def functional(model_in):
            snaps, _ = evo.run_ice_history(dates, model_in, rtol=1e-12)
            return objective(snaps)

        exact = functional(_Perturbed(length_scale=scale)) - functional(ice_model)
        for name, method in (
            ("discrete", evo.adjoint_ice_history),
            ("continuous", evo.continuous_adjoint_ice_history),
        ):
            kernels = method(dates, measurement, rtol=1e-12)
            predicted = sum(
                earth.integrate(
                    SHGrid.from_array(
                        kernels.thickness[n].data * deltas[float(dates[n])].data,
                        grid=earth.grid,
                    )
                )
                for n in range(dates.size)
            )
            error = abs(predicted - exact) / abs(exact)
            if name == "discrete":
                assert error < 1e-8, error
            elif continuous_bound is not None:
                assert error < continuous_bound, error


def test_initial_state_kernels(setting):
    """The initial-state duals about a migrating trajectory: the
    initial-sea-level kernel (the discrete analogue of Yu et al. 2025,
    eq. 65, assembled from the per-node load duals at no extra cost)
    and the initial-thickness dual (node 0 plus the baseline, which
    grounds with the INITIAL shorelines) both match central
    differences with rebuilt initial states exactly, while no
    flotation cell flips."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _marine_history_state(material, earth)
    operator = _maxwell_operator(material)
    rng = np.random.default_rng(12)
    scale = earth.parameters.length_scale

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    measurement = {0.0: smooth(1.0)}
    dates = np.linspace(10.0, 0.0, 5)
    evo = SeaLevelEvolution(operator, state)
    snapshots, _ = evo.run_ice_history(
        dates, ice_model, shorelines="migrating", rtol=1e-12
    )
    history = [snap.ocean_function for snap in snapshots]
    kernels = evo.adjoint_ice_history(
        dates, measurement, ocean_functions=history, rtol=1e-12
    )
    g = measurement[0.0]
    ice0 = state.ice_thickness

    def functional(state_in, model_in):
        run = SeaLevelEvolution(operator, state_in)
        snaps, _ = run.run_ice_history(
            dates, model_in, shorelines="migrating", rtol=1e-12
        )
        return earth.integrate(
            SHGrid.from_array(g.data * snaps[-1].sea_level_change.data, grid=earth.grid)
        )

    # the initial sea level
    delta_sl = smooth(0.5 / scale)
    plus = functional(
        EarthState(
            ice0,
            SHGrid.from_array(state.sea_level.data + delta_sl.data, grid=earth.grid),
            earth,
            exclude_caspian=False,
        ),
        ice_model,
    )
    minus = functional(
        EarthState(
            ice0,
            SHGrid.from_array(state.sea_level.data - delta_sl.data, grid=earth.grid),
            earth,
            exclude_caspian=False,
        ),
        ice_model,
    )
    fd = 0.5 * (plus - minus)
    predicted = earth.integrate(
        SHGrid.from_array(
            kernels.initial_sea_level.data * delta_sl.data, grid=earth.grid
        )
    )
    assert abs(predicted - fd) < 1e-6 * abs(fd), (fd, predicted)

    # the initial thickness: the history's I0 and the state's ice move
    # together
    delta_ice = smooth(0.5 / scale)

    class _Shifted(BaseIceModel):
        sign = 1.0

        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            base = ice_model.get_ice_thickness(
                date, lmax, grid=grid, sampling=sampling, extend=extend
            )
            add = _Shifted.sign * delta_ice.data if np.isclose(date, 10.0) else 0.0
            return SHGrid.from_array(base.data + add, grid=base.grid)

    def thickness_run(sign):
        _Shifted.sign = sign
        shifted_state = EarthState(
            SHGrid.from_array(ice0.data + sign * delta_ice.data, grid=earth.grid),
            state.sea_level,
            earth,
            exclude_caspian=False,
        )
        return functional(shifted_state, _Shifted(length_scale=scale))

    fd = 0.5 * (thickness_run(+1.0) - thickness_run(-1.0))
    predicted = earth.integrate(
        SHGrid.from_array(kernels.thickness[0].data * delta_ice.data, grid=earth.grid)
    )
    assert abs(predicted - fd) < 1e-8 * abs(fd), (fd, predicted)


def test_kernels_at_the_ice_models_dates(setting):
    """`IceKernels.at_dates`: the transpose of the model's linear
    interpolation maps the run-grid duals onto the model's own slices,
    exactly -- the inversion-facing parameterisation."""
    from pyshtools import SHCoeffs

    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    rng = np.random.default_rng(15)
    scale = earth.parameters.length_scale

    def smooth(amplitude):
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return amplitude * earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    model_dates = np.array([10.0, 5.0, 0.0])
    slice_deltas = [0.0 * smooth(1.0).data] + [
        smooth(10.0 / scale).data for _ in model_dates[1:]
    ]

    class _Sliced(BaseIceModel):
        """Linear interpolation between perturbed copies of the
        analytical model's slices: exactly the BaseIceModel contract."""

        perturbed = False

        def get_ice_thickness(
            self, date, lmax, /, *, grid="DH", sampling=1, extend=True
        ):
            j = int(np.searchsorted(-model_dates, -date, side="right")) - 1
            j = min(max(j, 0), model_dates.size - 2)
            lam = (model_dates[j] - date) / (model_dates[j] - model_dates[j + 1])
            values = []
            for jj in (j, j + 1):
                base = ice_model.get_ice_thickness(
                    model_dates[jj], lmax, grid=grid, sampling=sampling, extend=extend
                ).data
                if _Sliced.perturbed:
                    base = base + slice_deltas[jj]
                values.append(base)
            return SHGrid.from_array(
                (1.0 - lam) * values[0] + lam * values[1],
                grid=self.resolve_grid(grid, sampling)[0],
            )

    run_dates = np.linspace(10.0, 0.0, 9)  # finer than the slices
    measurement = {0.0: smooth(1.0)}
    g = measurement[0.0]

    def functional(perturbed):
        _Sliced.perturbed = perturbed
        snaps, _ = evo.run_ice_history(
            run_dates, _Sliced(length_scale=scale), rtol=1e-12
        )
        return earth.integrate(
            SHGrid.from_array(g.data * snaps[-1].sea_level_change.data, grid=earth.grid)
        )

    exact = functional(True) - functional(False)
    kernels = evo.adjoint_ice_history(run_dates, measurement, rtol=1e-12)
    slice_duals = kernels.at_dates(model_dates)
    predicted = sum(
        earth.integrate(
            SHGrid.from_array(slice_duals[j].data * slice_deltas[j], grid=earth.grid)
        )
        for j in range(model_dates.size)
    )
    assert abs(predicted - exact) < 1e-8 * abs(exact), (exact, predicted)
    with pytest.raises(ValueError, match="bracket"):
        kernels.at_dates(np.array([8.0, 0.0]))


def test_coupled_adaptive_trapezoid(setting):
    """The adaptive coupled loop: tolerance ordering against a fine
    fixed-step reference, striding through the relaxation after a
    Heaviside unload, and the quantised ladder holding the step sizes
    -- and so the per-size Love tables and solvers -- to a handful."""
    material, earth, _, _, _ = setting
    _, state = _marine_history_state(material, earth)
    evo = SeaLevelEvolution(_maxwell_operator(material), state)
    parameters = earth.parameters
    kyr = 1e3 * 365.25 * 86400.0 / parameters.scales.factor(units.TIME)
    unload = SHGrid.from_array(
        -0.25
        * parameters.ice_density
        * (1.0 - state.ocean_function.data)
        * state.ice_thickness.data,
        grid=earth.grid,
    )

    def ice_load(t):
        return unload

    t_end = 10.0 * kyr
    reference, _ = evo.run(np.linspace(0.0, t_end, 401), ice_load)
    xi_ref = reference[-1].sea_level_change.data
    scale = np.max(np.abs(xi_ref))

    errors = {}
    for rtol_step in (1e-1, 1e-2):
        snapshots, _ = evo.run_adaptive(
            0.0, t_end, ice_load, dt_initial=0.02 * kyr, rtol_step=rtol_step
        )
        steps = np.diff([snap.t for snap in snapshots])
        assert np.all(steps > 0.0)
        assert abs(snapshots[-1].t - t_end) < 1e-9 * t_end
        assert len(set(np.round(steps / kyr, 10))) <= 12
        errors[rtol_step] = (
            np.max(np.abs(snapshots[-1].sea_level_change.data - xi_ref)) / scale
        )
        if rtol_step == 1e-1:
            assert len(steps) < 80
            assert max(steps) > 10.0 * (0.02 * kyr)
    assert errors[1e-1] < 2e-3
    assert errors[1e-2] < 1e-4
    assert errors[1e-2] < 0.1 * errors[1e-1]


def test_viscosity_kernels(setting):
    """The discrete inverse-relaxation-time kernels: a reverse sweep
    given the recorded forward trajectory pairs the scheme-weight
    derivatives with the duals, and the FD residual against a smooth
    nodal itau perturbation falls at second order in the step -- the
    kernel itself is exact.  Both implicit schemes, and the viscosity
    mapping -itau^2 / mu."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    mesh = material.mesh
    rng = np.random.default_rng(7)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    g = earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )
    dates = np.linspace(10.0, 0.0, 6)
    measurement = {0.0: g}
    ditau = (0.3 + 0.7 * (mesh.r / mesh.r.max()) ** 2) / TAU

    for scheme, tol in (("exponential_trapezoid", 1e-6), ("backward_euler", 1e-6)):
        op = _maxwell_operator(material)
        evo = SeaLevelEvolution(op, state, scheme=scheme)
        snapshots, _ = evo.run_ice_history(dates, ice_model, record=True)
        kernels = evo.adjoint_ice_history(
            dates,
            measurement,
            internal_states=[snap.internal_state for snap in snapshots],
        )
        predicted = float(np.sum(kernels.itau[0] * ditau))

        def functional(eps):
            rheology = Rheology(
                material, [(op.rheology.mu[0], op.rheology.itau[0] + eps * ditau)]
            )
            evolution = SeaLevelEvolution(
                ViscoelasticOperator(rheology, LMAX), state, scheme=scheme
            )
            snaps, _ = evolution.run_ice_history(dates, ice_model)
            return earth.integrate(
                SHGrid.from_array(
                    g.data * snaps[-1].sea_level_change.data, grid=earth.grid
                )
            )

        fd = {
            eps: (functional(eps) - functional(-eps)) / (2.0 * eps)
            for eps in (0.02, 0.01)
        }
        # second-order truncation, and Richardson extrapolation lands
        # on the kernel's prediction to solver precision
        resid = {eps: abs(f - predicted) for eps, f in fd.items()}
        assert 3.0 < resid[0.02] / resid[0.01] < 5.0, scheme
        richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
        assert abs(richardson - predicted) < tol * abs(predicted), scheme

        # the viscosity mapping, and the refusal without the trajectory
        eta = kernels.viscosity(op.rheology)
        want = -(op.rheology.itau[0] ** 2) / op.rheology.mu[0] * kernels.itau[0]
        assert np.allclose(eta[0], want)
    bare = evo.adjoint_ice_history(dates, measurement)
    assert bare.itau is None
    with pytest.raises(ValueError, match="internal_states"):
        bare.viscosity(op.rheology)


def test_continuous_viscosity_kernels(setting):
    """The continuous-adjoint viscosity kernels converge to the exact
    discrete ones at second order as the run resolves its rheology,
    and the disagreement on a stiff body is the resolution diagnostic
    firing -- the two-method doctrine applied to dJ / d itau."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rng = np.random.default_rng(7)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    g = earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )
    measurement = {0.0: g}

    def disagreement(op, nsteps):
        evo = SeaLevelEvolution(op, state)
        dates = np.linspace(10.0, 0.0, nsteps)
        snapshots, _ = evo.run_ice_history(dates, ice_model, record=True)
        states = [snap.internal_state for snap in snapshots]
        discrete = evo.adjoint_ice_history(
            dates, measurement, internal_states=states
        ).itau[0]
        continuous = evo.continuous_adjoint_ice_history(
            dates, measurement, internal_states=states
        ).itau[0]
        return np.linalg.norm(continuous - discrete) / np.linalg.norm(discrete)

    slow = ViscoelasticOperator(
        Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX
    )
    coarse = disagreement(slow, 11)
    fine = disagreement(slow, 21)
    assert coarse < 1e-2
    assert fine < 3e-3
    assert 3.0 < coarse / fine < 5.0

    stiff = disagreement(_maxwell_operator(material), 6)
    assert stiff > 0.2


def test_migrating_viscosity_kernels(setting):
    """Viscosity kernels about a migrating-shoreline trajectory: the
    discrete linearisation matches central FD of the full nonlinear
    forward at second order in the perturbation (Richardson lands on
    the kernel), and the continuous kernels converge to the discrete
    at second order when resolved."""
    material, earth, _, _, _ = setting
    mesh = material.mesh
    ice_model, state = _marine_history_state(material, earth)
    rng = np.random.default_rng(7)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    g = earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )
    dates = np.linspace(_MeltingCap.START, 0.0, 6)
    measurement = {0.0: g}
    ditau = (0.3 + 0.7 * (mesh.r / mesh.r.max()) ** 2) / TAU

    op = _maxwell_operator(material)
    evo = SeaLevelEvolution(op, state)
    snapshots, _ = evo.run_ice_history(
        dates, ice_model, shorelines="migrating", record=True
    )
    oceans = [snap.ocean_function for snap in snapshots]
    states = [snap.internal_state for snap in snapshots]
    kernels = evo.adjoint_ice_history(
        dates, measurement, ocean_functions=oceans, internal_states=states
    )
    predicted = float(np.sum(kernels.itau[0] * ditau))

    def functional(eps):
        rheology = Rheology(
            material, [(op.rheology.mu[0], op.rheology.itau[0] + eps * ditau)]
        )
        evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
        snaps, _ = evolution.run_ice_history(dates, ice_model, shorelines="migrating")
        return earth.integrate(
            SHGrid.from_array(g.data * snaps[-1].sea_level_change.data, grid=earth.grid)
        )

    fd = {
        eps: (functional(eps) - functional(-eps)) / (2.0 * eps) for eps in (0.02, 0.01)
    }
    resid = {eps: abs(f - predicted) for eps, f in fd.items()}
    assert 3.0 < resid[0.02] / resid[0.01] < 5.0
    richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
    assert abs(richardson - predicted) < 1e-5 * abs(predicted)

    # the continuous kernels, in the resolved regime
    slow = ViscoelasticOperator(
        Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX
    )
    evolution = SeaLevelEvolution(slow, state)

    def disagreement(nsteps):
        grid_dates = np.linspace(_MeltingCap.START, 0.0, nsteps)
        snaps, _ = evolution.run_ice_history(
            grid_dates, ice_model, shorelines="migrating", record=True
        )
        oc = [snap.ocean_function for snap in snaps]
        st = [snap.internal_state for snap in snaps]
        discrete = evolution.adjoint_ice_history(
            grid_dates, measurement, ocean_functions=oc, internal_states=st
        ).itau[0]
        continuous = evolution.continuous_adjoint_ice_history(
            grid_dates, measurement, ocean_functions=oc, internal_states=st
        ).itau[0]
        return np.linalg.norm(continuous - discrete) / np.linalg.norm(discrete)

    coarse = disagreement(11)
    fine = disagreement(21)
    assert coarse < 3e-3
    assert fine < 1e-3
    assert 3.0 < coarse / fine < 5.5


def test_checkpointed_viscosity_kernels(setting):
    """A strided recording reproduces the full-record viscosity
    kernels through exact backward reconstruction, the residual at the
    exp(sum h) roundoff budget of the gaps; migrating trajectories
    reconstruct too, and a bare state list with gaps is refused."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rng = np.random.default_rng(7)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    g = earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )
    measurement = {0.0: g}
    dates = np.linspace(10.0, 0.0, 11)

    # resolved: tiny amplification; stiff: exp(6.4) per gap, still fine
    for tau_factor, stride, tol in ((10.0, 3, 1e-10), (1.0, 2, 1e-9)):
        op = ViscoelasticOperator(
            Rheology.from_relaxation_time(material, tau_factor * TAU), LMAX
        )
        evo = SeaLevelEvolution(op, state)
        full, _ = evo.run_ice_history(dates, ice_model, record=True)
        sparse, _ = evo.run_ice_history(dates, ice_model, record=stride)
        kept = sum(1 for snap in sparse if snap.internal_state is not None)
        assert kept < len(sparse)
        for adjoint in (evo.adjoint_ice_history, evo.continuous_adjoint_ice_history):
            reference = adjoint(dates, measurement, internal_states=full).itau[0]
            kernels = adjoint(dates, measurement, internal_states=sparse).itau[0]
            assert np.linalg.norm(kernels - reference) < tol * np.linalg.norm(
                reference
            ), (tau_factor, adjoint.__name__)

    # a migrating trajectory reconstructs the same way
    marine_model, marine_state = _marine_history_state(material, earth)
    evo = SeaLevelEvolution(
        ViscoelasticOperator(Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX),
        marine_state,
    )
    marine_dates = np.linspace(_MeltingCap.START, 0.0, 11)
    full, _ = evo.run_ice_history(
        marine_dates, marine_model, shorelines="migrating", record=True
    )
    sparse, _ = evo.run_ice_history(
        marine_dates, marine_model, shorelines="migrating", record=3
    )
    oceans = [snap.ocean_function for snap in full]
    reference = evo.adjoint_ice_history(
        marine_dates, measurement, ocean_functions=oceans, internal_states=full
    ).itau[0]
    kernels = evo.adjoint_ice_history(
        marine_dates, measurement, ocean_functions=oceans, internal_states=sparse
    ).itau[0]
    assert np.linalg.norm(kernels - reference) < 1e-10 * np.linalg.norm(reference)

    # a bare state list with gaps carries no loads to reconstruct from
    with pytest.raises(ValueError, match="checkpointed trajectory"):
        evo.adjoint_ice_history(
            marine_dates,
            measurement,
            ocean_functions=oceans,
            internal_states=[snap.internal_state for snap in sparse],
        )
