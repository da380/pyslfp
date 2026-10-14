"""The coupled time loop: elastic fingerprint recovery, mass
conservation, convergence order, and the rotational coupling."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh
from pyshtools import SHGrid

from pyslfp.core import EarthModel
from pyslfp.ice import BaseIceModel
from pyslfp.core import SeaLevelEquation
from pyslfp.core import EarthState
from pyslfp.love_numbers import Material, fluid_tidal_number, love_numbers
from pyslfp.viscoelastic import (
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
    calibrate_initial_sea_level,
)

RHO, VS, VP, RADIUS = 5510.0, 5000.0, 12000.0, 6.371e6
MU = RHO * VS * VS
LMAX = 4
TAU_SI = 1.0e10  # seconds; tau_nd below is in the model's time units


@pytest.fixture(scope="module")
def setting():
    model = LayeredIsotropicElastic([0.0, RADIUS], rho=[RHO], vp=[VP], vs=[VS])
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
    tau_nd = TAU_SI / earth.parameters.scales.factor(
        __import__("planetmodel").units.TIME
    )
    return material, earth, state, load, tau_nd


def _maxwell_operator(material):
    return ViscoelasticOperator(Rheology.maxwell(material, MU * TAU_SI), LMAX)


def test_elastic_fingerprint_is_recovered(setting):
    """With an infinite relaxation time everywhere the evolution's every
    snapshot is the elastic fingerprint of the same load."""
    material, earth, state, load, _ = setting
    shape = material.moduli.shape
    rheo = Rheology(material, [(np.full(shape, MU), np.zeros(shape))])
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
    """The background at 10 ka: the full cap, grounded by flotation."""
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
    # melting grounded ice raises the ocean mean
    mean = (
        earth.integrate(snapshots[-1].sea_level_change * state.ocean_function)
        / state.ocean_area
    )
    assert mean * earth.parameters.length_scale > 1.0
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
    ice_model, _ = _history_state(material, raw_earth)
    ice = ice_model.get_ice_thickness(
        _MeltingCap.START, LMAX, grid=raw_earth.grid_name, extend=raw_earth.extend
    )
    sea = raw_earth.constant_grid(3000.0 / raw_earth.parameters.length_scale)
    state = EarthState(ice, sea, raw_earth, exclude_caspian=False)
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
    ice_model, state = _history_state(material, earth)
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
    shape = material.moduli.shape
    rheo = Rheology(material, [(np.full(shape, MU), np.zeros(shape))])
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
