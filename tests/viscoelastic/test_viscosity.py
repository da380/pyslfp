"""
Tests for the log-viscosity space: its layers follow the rheology, its
values at the nodes are continuous within a layer and jump at interfaces, the
transpose of that evaluation is exact, a rheology round-trips through it, and
the derivative it builds from the inverse-relaxation-time kernels is the
derivative of an objective along the space, by finite differences.
"""

import numpy as np
import pytest
from pygeoinf.testing import check_coordinates, check_space
from pyshtools import SHGrid

from pyslfp.core import EarthState
from pyslfp.ice import AnalyticalIceModel
from pyslfp.viscoelastic import (
    LogViscositySpace,
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
    prem_maxwell,
    viscoelastic_earth,
)

from .body import LMAX, TAU


def _two_branches(material):
    """A Burgers-like body: two relaxing branches sharing the shear
    modulus, with relaxation times about a decade apart, in the material's
    units. The profiles are even in r, as a profile regular at the centre
    of the whole-ball test body must be."""
    _, mu = (np.asarray(a).real for a in material.moduli.kappa_mu())
    r = material.mesh.r
    return Rheology(
        material,
        [
            (0.6 * mu, (1.0 + 0.5 * r**2) / TAU),
            (0.4 * mu, (1.0 + r**2) / (10.0 * TAU)),
        ],
    )


@pytest.fixture(scope="module")
def prem():
    """PREM with a two-layer Maxwell mantle, and a second branch in the
    lower mantle alone."""
    model = prem_maxwell(upper_mantle=1.0e21, lower_mantle=3.0e21)
    _, maxwell = viscoelastic_earth(model, 8)
    material = maxwell.material
    mu_1 = np.array(maxwell.mu[0])
    mu_2 = np.zeros_like(mu_1)
    itau_2 = np.zeros_like(mu_1)
    lower = material.mesh.layer == 3
    mu_2[lower] = 0.3 * mu_1[lower]
    mu_1[lower] *= 0.7
    itau_2[lower] = 5.0 * maxwell.itau[0][lower]
    return Rheology(material, [(mu_1, maxwell.itau[0]), (mu_2, itau_2)])


def _random(space, seed):
    return space.from_components(np.random.default_rng(seed).standard_normal(space.dim))


# ==================================================================== #
#                             Structure                                #
# ==================================================================== #


def test_space_axioms(setting):
    material, _, _, _, _ = setting
    rheology = _two_branches(material)
    space = LogViscositySpace(rheology, 12)
    rng = np.random.default_rng(0)
    check_space(space, rng=rng, rebuild=lambda: LogViscositySpace(rheology, 12))
    check_coordinates(space, rng=rng)
    assert space.num_branches == 2
    assert space.labels == ("branch_0", "branch_1")
    assert space.branch_layers(0) == (0,) and space.branch_layers(1) == (0,)
    assert space.dim == 24


def test_the_layers_follow_the_rheology(prem):
    space = LogViscositySpace(prem, {2: 4, 3: 8, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2})
    assert space.branch_layers(0) == (2, 3, 4, 5, 6, 7, 8)
    assert space.branch_layers(1) == (3,)
    assert space.branch_space(0).labels[1] == "layer_3"
    assert space.dim == 24 + 8
    bounds = space.branch_space(1).layer_bounds[0]
    r = prem.material.mesh.r[prem.material.mesh.layer == 3]
    assert bounds == (r.min(), r.max())
    assert space.viscosity_unit > 0.0


def test_a_branch_relaxing_in_part_of_a_layer_is_refused(prem):
    mu, itau = np.array(prem.mu[0]), np.array(prem.itau[0])
    elements = np.flatnonzero(prem.material.mesh.layer == 3)
    itau[elements[: elements.size // 2]] = 0.0
    with pytest.raises(ValueError, match="part of layer 3"):
        LogViscositySpace(Rheology(prem.material, [(mu, itau)]), 4)
    with pytest.raises(ValueError, match="relaxes nowhere"):
        LogViscositySpace(Rheology(prem.material, [(mu, np.zeros_like(itau))]), 4)


# ==================================================================== #
#                           Nodal values                               #
# ==================================================================== #


def test_nodal_values_are_continuous_within_a_layer_and_jump_at_interfaces(prem):
    space = LogViscositySpace(prem, 6)
    mesh = prem.material.mesh
    nodal = space.nodal(_random(space, 1))
    values = nodal[0]
    for e in range(mesh.nspec - 1):
        same_layer = mesh.layer[e] == mesh.layer[e + 1]
        relaxing = mesh.layer[e] in space.branch_layers(0)
        if same_layer and relaxing:
            assert values[e, -1] == pytest.approx(values[e + 1, 0], rel=1e-10)
        elif relaxing and mesh.layer[e + 1] in space.branch_layers(0):
            assert abs(values[e, -1] - values[e + 1, 0]) > 1e-3
    # the second branch relaxes in layer 3 alone
    outside = mesh.layer != 3
    assert np.all(nodal[1][outside] == 0.0) and np.any(nodal[1][~outside] != 0.0)


def test_the_transpose_of_evaluation_is_exact(prem):
    space = LogViscositySpace(prem, 5)
    rng = np.random.default_rng(2)
    chi = _random(space, 3)
    nodal = [rng.standard_normal(prem.material.mesh.r.shape) for _ in range(2)]
    paired = sum(float(np.sum(d * v)) for d, v in zip(nodal, space.nodal(chi)))
    assert space.derivative(nodal)(chi) == pytest.approx(paired, rel=1e-12)
    gradient = space.gradient(nodal)
    assert space.inner_product(gradient, chi) == pytest.approx(paired, rel=1e-12)


def test_a_rheology_round_trips(setting):
    """The log viscosity of a smooth rheology projects into the space and
    names it back to the fit's accuracy, and the reference's own vector
    reproduces it."""
    material, _, _, _, _ = setting
    rheology = _two_branches(material)
    space = LogViscositySpace(rheology, 24)
    chi = space.log_viscosity()
    back = space.rheology(chi)
    # the fit is in the volume measure, so the centre of the ball, which
    # carries no volume, is where it is least accurate
    for k in range(2):
        np.testing.assert_allclose(back.mu[k], rheology.mu[k])
        np.testing.assert_allclose(back.itau[k], rheology.itau[k], rtol=1e-4)
    moved = space.add(chi, space.scale(0.1, _random(space, 4)))
    other = space.rheology(moved)
    assert not np.allclose(other.itau[0], rheology.itau[0], rtol=1e-3)
    again = space.rheology(space.log_viscosity(other))
    for k in range(2):
        np.testing.assert_allclose(again.itau[k], other.itau[k], rtol=1e-4)


def test_the_representer_settles_as_modes_are_added(setting):
    """A nodal derivative is a sum of point evaluations, bounded above
    order one half, so its representer converges as modes are added."""
    material, _, _, _, _ = setting
    rheology = _two_branches(material)
    r = material.mesh.r
    nodal = [np.exp(-((r - 0.6) ** 2) / 0.02), 0.5 * r]
    norms = []
    for modes in (8, 16, 32):
        space = LogViscositySpace(rheology, modes)
        norms.append(space.norm(space.gradient(nodal)))
    changes = np.abs(np.diff(norms))
    assert changes[1] < 0.6 * changes[0]


# ==================================================================== #
#                       The derivative in chi                          #
# ==================================================================== #


def test_the_kernel_along_the_space(setting):
    """The functional built from the inverse-relaxation-time kernels is the
    derivative of an objective along a direction of the space: finite
    differences of the objective through `rheology(chi + eps v)` converge to
    it at second order."""
    material, earth, _, _, _ = setting
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    state = EarthState.from_ice_model(ice_model, 10.0, earth, land=None)
    space = LogViscositySpace(_two_branches(material), 12)
    chi = space.log_viscosity()
    v = _random(space, 5)
    dates = np.linspace(10.0, 0.0, 6)
    rng = np.random.default_rng(6)
    g = earth.expand_coefficient(
        earth.expand_field(
            SHGrid.from_array(
                rng.normal(size=earth.zero_grid().data.shape), grid=earth.grid
            )
        )
    )

    def objective(rheology):
        evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
        snapshots, _ = evolution.run_ice_history(dates, ice_model, record=True)
        value = earth.integrate(
            SHGrid.from_array(
                g.data * snapshots[-1].response.field("relative_sea_level_change").data,
                grid=earth.grid,
            )
        )
        return value, evolution, snapshots

    base = space.rheology(chi)
    _, evolution, snapshots = objective(base)
    kernels = evolution.adjoint_ice_history(
        dates,
        {0.0: g},
        internal_states=[snap.internal_state for snap in snapshots],
    )
    predicted = space.from_itau_kernels(kernels.itau, base)(v)
    fd = {}
    for eps in (0.02, 0.01):
        plus = objective(space.rheology(space.add(chi, space.scale(eps, v))))[0]
        minus = objective(space.rheology(space.add(chi, space.scale(-eps, v))))[0]
        fd[eps] = (plus - minus) / (2.0 * eps)
    resid = {eps: abs(f - predicted) for eps, f in fd.items()}
    assert 3.0 < resid[0.02] / resid[0.01] < 5.0
    richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
    assert abs(richardson - predicted) < 1e-6 * abs(predicted)
