"""The graded mesh: spans the model, honours its boundaries, is as fine as
the uniform rule where each degree needs it, and gives the same Love
numbers."""

import numpy as np
import pytest
from planetmodel import PREM, LayeredIsotropicElastic, RadialMesh

from pyslfp.love_numbers import Material, graded_mesh, love_numbers


@pytest.fixture(scope="module")
def model():
    return PREM(ocean=False)


def test_shape_of_the_mesh(model):
    lmax = 4096
    mesh = graded_mesh(model, lmax)
    a = model.skeleton.boundaries[-1]
    assert mesh.left[0] == 0.0 and mesh.right[-1] == a
    for b in model.skeleton.boundaries:
        assert np.any(np.isclose(mesh.right, b)) or b == 0.0
    widths = mesh.right - mesh.left
    floor = 0.1 * a / (lmax + 1)
    # uniform at the floor near the surface, never below it
    assert np.isclose(widths[-1], floor)
    assert np.all(widths > 0.0)
    # the width never exceeds the rule at the element's top
    depth = a - mesh.right
    allowed = np.maximum(floor, 0.1 * a * np.log1p(-depth / a) / np.log(1e-8))
    assert np.all(widths <= allowed * (1.0 + 1e-9))
    # a few hundred elements, not ten (lmax + 1)
    assert 500 < mesh.nspec < 1500
    # each degree's sub-mesh is at least as fine as the uniform rule for it
    for l in (2, 50, 500, lmax):
        e0 = mesh.start_element(l)
        assert widths[e0:].max() <= 0.1 * a / (l + 1) * (1.0 + 1e-9)


def test_agrees_with_the_uniform_rule(model):
    lmax = 64
    graded = love_numbers(model, lmax)
    uniform = love_numbers(Material(RadialMesh(model, ngll=5, lmax=lmax), model), lmax)
    for name in ("h_u", "k_u", "h_phi", "k_phi", "l_u", "h_t", "k_t", "h_v"):
        g, u = getattr(graded, name), getattr(uniform, name)
        scale = np.abs(u).max()
        assert np.allclose(g, u, rtol=0.0, atol=2e-6 * scale), name
    assert graded.reciprocity_residual().max() < 1e-12


def test_refusals_and_a_shell():
    with pytest.raises(ValueError, match="eps"):
        graded_mesh(PREM(ocean=False), 4, eps=2.0)
    with pytest.raises(ValueError, match="lmax"):
        graded_mesh(PREM(ocean=False), -1)
    shell = LayeredIsotropicElastic(
        [0.5, 0.8, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0], vs=[1.0, 1.0]
    )
    mesh = graded_mesh(shell, 16)
    assert mesh.left[0] == 0.5 and np.any(np.isclose(mesh.right, 0.8))
