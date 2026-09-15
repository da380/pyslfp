"""Smoke tests of the Love number figures."""

import matplotlib
import numpy as np
import pytest
from planetmodel import PREM, frozen

from pyslfp.love_numbers import (
    LoveNumbers,
    love_numbers,
    plot_degree_solution,
    plot_love_numbers,
    solve_degree,
)

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def model():
    return PREM(ocean=False)


def test_plot_love_numbers(model):
    fig, axes = plot_love_numbers(love_numbers(model, 8))
    assert fig is not None and axes.shape == (2,)
    assert len(axes[0].lines) == 3 and len(axes[1].lines) == 3
    matplotlib.pyplot.close(fig)


def test_plot_table_without_tangential_columns(tmp_path, model):
    love = love_numbers(model, 8)
    path = tmp_path / "legacy.dat"
    np.savetxt(
        path,
        np.column_stack(
            [
                love.degree,
                love.h_u,
                love.k_u,
                love.h_phi,
                love.k_phi,
                love.h_t,
                love.k_t,
            ]
        ),
    )
    fig, axes = plot_love_numbers(
        LoveNumbers.from_file(
            path, G=love.G, radius=love.radius, surface_gravity=love.surface_gravity
        )
    )
    assert len(axes[0].lines) == 2 and len(axes[1].lines) == 2
    matplotlib.pyplot.close(fig)


def test_plot_complex_table_and_solution(model):
    cold = frozen(model, 2 * np.pi / 43200.0)
    fig, axes = plot_love_numbers(love_numbers(cold, 4))
    assert len(axes[0].lines) == 3
    matplotlib.pyplot.close(fig)
    fig, ax = plot_degree_solution(solve_degree(cold, 2, forcing="tide"), n_points=50)
    assert len(ax.lines) >= 3
    matplotlib.pyplot.close(fig)


def test_plot_degree_solution_on_given_axes(model):
    fig, ax = matplotlib.pyplot.subplots()
    out_fig, out_ax = plot_degree_solution(
        solve_degree(model, 2), ax=ax, n_points=50, normalise=False
    )
    assert out_fig is fig and out_ax is ax
    assert ax.get_ylim() == (0.0, 6368e3)
    matplotlib.pyplot.close(fig)
