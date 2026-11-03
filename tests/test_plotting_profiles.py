"""
The radial profile plots: a nodal array becomes one radial field per layer,
exact at the nodes, the viscosity profile carries SI pascal seconds on a
logarithmic axis, and a log-viscosity vector is drawn over its branch's
layers.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyslfp.plotting import (
    plot_log_viscosity,
    plot_profile,
    plot_viscosity,
    radial_fields,
)
from pyslfp.viscoelastic import LogViscositySpace, prem_maxwell, viscoelastic_earth

LMAX = 4


@pytest.fixture(scope="module")
def rheology():
    _, rheology = viscoelastic_earth(
        prem_maxwell(upper_mantle=5e20, lower_mantle=3e21), LMAX
    )
    return rheology


def test_radial_fields_interpolate_the_nodes(rheology):
    material = rheology.material
    mesh = material.mesh
    nodal = np.sin(mesh.r)
    fields = radial_fields(material, nodal)
    assert len(fields) == np.unique(mesh.layer).size
    for i, field in zip(np.unique(mesh.layer), fields):
        elements = mesh.layer == i
        assert field.interval == (mesh.r[elements].min(), mesh.r[elements].max())
        r = mesh.r[elements].ravel()
        np.testing.assert_allclose(field(r), nodal[elements].ravel(), atol=1e-12)
    kept = radial_fields(material, nodal, layers=[int(np.unique(mesh.layer)[0])])
    assert kept[0] is not None and all(f is None for f in kept[1:])
    with pytest.raises(ValueError):
        radial_fields(material, nodal[:-1])


def test_plot_profile_draws_one_segment_per_layer(rheology):
    material = rheology.material
    mesh = material.mesh
    _, ax = plt.subplots()
    lines = plot_profile(ax, material, mesh.r, label="r", joiners=False)
    assert len(lines) == np.unique(mesh.layer).size
    assert lines[0].get_label() == "r"
    ys = np.concatenate([line.get_ydata() for line in lines])
    assert ys.min() == pytest.approx(mesh.r.min()) and ys.max() == pytest.approx(
        mesh.r.max()
    )
    plt.close("all")


def test_plot_viscosity_is_in_pascal_seconds_on_a_log_axis(rheology):
    _, ax = plt.subplots()
    lines = plot_viscosity(ax, rheology)
    assert ax.get_xscale() == "log"
    values = np.concatenate([line.get_xdata() for line in lines])
    assert values.min() == pytest.approx(5e20, rel=1e-6)
    assert values.max() == pytest.approx(3e21, rel=1e-6)
    plt.close("all")


def test_plot_log_viscosity_draws_the_branch_layers(rheology):
    space = LogViscositySpace(rheology, 6)
    chi = space.from_components(np.random.default_rng(0).standard_normal(space.dim))
    _, ax = plt.subplots()
    lines = plot_log_viscosity(ax, space, chi, joiners=False)
    assert len(lines) == len(space.branch_layers(0))
    mesh = space.mesh
    inside = np.isin(mesh.layer, space.branch_layers(0))
    radii = np.concatenate([line.get_ydata() for line in lines])
    assert radii.min() == pytest.approx(mesh.r[inside].min())
    assert radii.max() == pytest.approx(mesh.r[inside].max())
    plt.close("all")
