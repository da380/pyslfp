"""
Tests for the convention of the kernels with respect to a field in the Earth:
each is a density per unit volume sampled at the nodes, pairing with a
perturbation through the volume weights 4 pi r^2 w jac. The volume measure
integrates exactly what its quadrature should, a gradient's samples are
recovered with the centre extrapolated, the relaxation-time kernel is the
same function on radial meshes of different resolution and order while the
raw gradient entries are not, its volume pairing is the tangent-linear
derivative, and it is the L2 representative of the derivative under the
volume measure of pygeoinf's radial spaces.
"""

import numpy as np
import pytest
from planetmodel import RadialMesh
from pyshtools import SHCoeffs

from pyslfp.love_numbers import Material
from pyslfp.viscoelastic import (
    LogViscositySpace,
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
)

from .body import LMAX, TAU
from .test_evolution import _history_state
from .test_tangent_linear import _pair, _sea_level


def _volume_norm(material, kernel):
    return np.sqrt(material.volume_pairing(kernel, kernel))


def test_the_volume_measure(setting):
    """The weights integrate polynomials in r of the GLL rule's degree over
    the ball exactly, the pairing is their quadrature, and the samples of a
    gradient W f are f, at the centre too, where the weight vanishes and
    the sample comes from the element's polynomial through its other
    nodes."""
    material, _, _, _, _ = setting
    r = material.mesh.r
    weights = material.volume_weights
    assert weights.shape == r.shape
    assert weights[0, 0] == 0.0 and np.all(weights[r > 0.0] > 0.0)
    assert np.sum(weights) == pytest.approx(4.0 * np.pi / 3.0, rel=1e-13)
    # 4 pi int (1 + r) r^2 r^2 dr over the unit ball
    assert material.volume_pairing(1.0 + r, r**2) == pytest.approx(
        4.0 * np.pi * (1.0 / 5.0 + 1.0 / 6.0), rel=1e-13
    )
    # a cubic is the element's polynomial through four nodes, so the
    # centre's extrapolated sample is exact; its gradient entry is ignored
    f = 1.0 - 2.0 * r + 3.0 * r**3
    gradient = weights * f
    gradient[0, 0] = 1.0
    np.testing.assert_allclose(material.kernel_samples(gradient), f, rtol=1e-12)
    with pytest.raises(ValueError, match="shape"):
        material.volume_pairing(f[:-1], f[:-1])
    with pytest.raises(ValueError, match="shape"):
        material.kernel_samples(f.ravel())


def _measurement(earth):
    rng = np.random.default_rng(7)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    return earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )


def test_the_kernel_is_a_function_of_radius(setting):
    """The same problem on the test body's mesh (50 elements of 5 nodes),
    on 20 elements of 5 nodes and on 10 elements of 7: the discrete
    relaxation-time kernels agree as functions, interpolated onto the first
    mesh and compared in the volume-weighted L2 norm, to the radial
    discretisation error (observed 2.5e-7 and 6.4e-9), while the raw
    gradient entries W K differ at order one. On each mesh the volume
    pairing of the kernel with a perturbation of itau is the tangent-linear
    derivative, and the pairings with one smooth perturbation agree across
    the meshes."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    g = _measurement(earth)
    dates = np.linspace(10.0, 0.0, 6)

    def delta(r):
        return (0.3 + 0.7 * r**2) / TAU

    def kernel_on(material):
        evolution = SeaLevelEvolution(
            ViscoelasticOperator(Rheology.from_relaxation_time(material, TAU), LMAX),
            state,
        )
        snapshots, _ = evolution.run_ice_history(
            dates, ice_model, record=True, rtol=1e-12
        )
        kernel = evolution.adjoint_ice_history(
            dates, {0.0: g}, internal_states=snapshots, rtol=1e-12
        ).itau[0]
        direction = delta(material.mesh.r)
        tangent, _ = evolution.tangent_linear_ice_history(
            dates, snapshots, itau=[direction], rtol=1e-12
        )
        derivative = _pair(earth, g, _sea_level(tangent[-1]))
        paired = material.volume_pairing(kernel, direction)
        assert abs(paired - derivative) < 1e-9 * abs(derivative)
        return kernel, paired

    reference, reference_pairing = kernel_on(material)
    nodes = material.mesh.r
    weights = material.volume_weights
    model = material.model
    for mesh, tolerance in (
        (RadialMesh(model, ngll=5, drmax=0.05), 1e-6),
        (RadialMesh(model, ngll=7, drmax=0.1), 5e-8),
    ):
        other = Material(mesh, model)
        kernel, paired = kernel_on(other)
        element, basis = mesh.interpolation_weights(nodes)
        interpolated = np.einsum("...i,...i->...", basis, kernel[element])
        assert _volume_norm(material, interpolated - reference) < (
            tolerance * _volume_norm(material, reference)
        )
        assert abs(interpolated[0, 0] - reference[0, 0]) < 1e-4 * abs(reference[0, 0])
        # the gradient entries carry each mesh's own quadrature weights
        raw = np.einsum(
            "...i,...i->...", basis, (other.volume_weights * kernel)[element]
        )
        raw_reference = weights * reference
        assert np.linalg.norm(raw - raw_reference) > np.linalg.norm(raw_reference)
        assert abs(paired - reference_pairing) < 1e-9 * abs(reference_pairing)


def test_the_kernel_is_the_l2_representative(setting):
    """In the Lebesgue log-viscosity space, whose inner product is pygeoinf's
    volume integral 4 pi int u v r^2 dr, the Riesz representer of the
    functional chi -> int K chi dV formed with the volume weights is K
    itself, to the projection onto the modes: no factor of r^2 or 4 pi
    stands between the kernel and the space's measure. Without padding,
    32 modes hold an even profile to 5.5e-3."""
    material, _, _, _, _ = setting
    rheology = Rheology.from_relaxation_time(material, TAU)
    space = LogViscositySpace(rheology, 32, order=0.0, padding=0.0)
    kernel = np.cos(2.0 * material.mesh.r)
    representer = space.gradient([material.volume_weights * kernel])
    values = space.nodal(representer)[0]
    assert _volume_norm(material, values - kernel) < 1e-2 * _volume_norm(
        material, kernel
    )
