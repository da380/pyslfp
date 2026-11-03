"""
Tests for the ice history operator: its adjoint identity in L2 and Sobolev
history spaces, and its adjoint as a Sobolev kernel, bounded as time modes are
added where the L2 representer grows, and converging as the date grid is
refined where the node duals at an observation date do not.
"""

import numpy as np
import pytest
from pygeoinf.interval import Lebesgue as TimeLebesgue
from pygeoinf.interval import Sobolev as TimeSobolev
from pygeoinf.testing import check_operator

from pyslfp.core import (
    EarthModel,
    EarthState,
    FieldHistorySpace,
    lebesgue_load_space,
    lebesgue_response_space,
    sobolev_load_space,
    sobolev_response_space,
)
from pyslfp.core.response import DISPLACEMENT
from pyslfp.ice import AnalyticalIceModel
from pyslfp.observations import GnssObservationModel
from pyslfp.viscoelastic import (
    IceHistoryOperator,
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
)

from .body import LMAX, TAU

START, END = 0.0, 10.0  # the run's span, in kiloyears
LENGTH = 2.0  # the time length scale, in kiloyears


def _domain(earth, time_kind, modes=16, field_kind="lebesgue"):
    # the history's time is the model's, from zero at the run's first date
    kiloyear = earth.parameters.kiloyear
    bounds = dict(lower=START * kiloyear, upper=END * kiloyear)
    if time_kind == "lebesgue":
        time_space = TimeLebesgue(modes, length_scale=LENGTH * kiloyear, **bounds)
    else:
        time_space = TimeSobolev(modes, 1.0, LENGTH * kiloyear, **bounds)
    if field_kind == "lebesgue":
        field_space = lebesgue_load_space(earth)
    else:
        b = earth.parameters.mean_sea_floor_radius
        field_space = sobolev_load_space(earth, 2.0, 0.3 * b)
    return FieldHistorySpace(time_space, field_space)


def _evolution(material, earth):
    """A Maxwell mantle with a relaxation time of about three kiloyears,
    resolved by the date grids here."""
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    state = EarthState.from_ice_model(ice_model, 10.0, earth, land=None)
    rheology = Rheology.from_relaxation_time(material, 10.0 * TAU)
    return SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev"])
def test_the_adjoint_identity(setting, kind):
    """<F u, g> = <u, F* g> for random histories and duals, the duals
    holding the horizontal displacement too: the operator's adjoint is the
    discrete adjoint carried onto the history space."""
    material, earth, _, _, _ = setting
    operator = IceHistoryOperator(
        _evolution(material, earth),
        np.linspace(10.0, 0.0, 5),
        _domain(earth, kind, field_kind=kind),
        [5.0, 0.0],
        space=lebesgue_response_space(earth, horizontal=True),
        rtol=1e-12,
    )
    check_operator(operator, rng=np.random.default_rng(0), trials=3, rtol=1e-8)


def _dual(earth, seed):
    space = lebesgue_response_space(earth)
    b = earth.parameters.mean_sea_floor_radius
    return space, space.heat_measure(0.5 * b, angular_velocity_std=1e3).sample(
        rng=np.random.default_rng(seed)
    )


def test_the_kernel_as_time_modes_are_added(setting):
    """With an observation inside the interval, the kernel in the Sobolev
    history space settles as time modes are added; in the L2 space it grows
    with them, as the observation's delta in time has no L2 representer."""
    material, earth, _, _, _ = setting
    evolution = _evolution(material, earth)
    space, dual = _dual(earth, 1)
    dates = np.linspace(10.0, 0.0, 33)
    norms = {"sobolev": [], "lebesgue": []}
    for kind, values in norms.items():
        for modes in (8, 16, 32):
            domain = _domain(earth, kind, modes)
            operator = IceHistoryOperator(
                evolution, dates, domain, [5.0], space=space, rtol=1e-12
            )
            values.append(domain.norm(operator.adjoint([dual])))
    sobolev, lebesgue = norms["sobolev"], norms["lebesgue"]
    assert abs(sobolev[2] - sobolev[1]) < 0.6 * abs(sobolev[1] - sobolev[0])
    assert lebesgue[2] > 1.6 * lebesgue[0]


def test_the_kernel_as_the_date_grid_is_refined(setting):
    """In a Sobolev history space the kernel converges as the date grid is
    refined, while the node dual at an observation date, per kiloyear, grows
    as the spacing shrinks, the observation's delta being lumped into it."""
    material, earth, _, _, _ = setting
    evolution = _evolution(material, earth)
    space, dual = _dual(earth, 2)
    domain = _domain(earth, "sobolev", 16)
    kernels, lumped = [], []
    for nodes in (5, 9, 17, 33):
        dates = np.linspace(10.0, 0.0, nodes)
        operator = IceHistoryOperator(
            evolution, dates, domain, [5.0], space=space, rtol=1e-12
        )
        kernels.append(operator.adjoint([dual]))
        duals = evolution.adjoint_ice_history(dates, {5.0: dual}, rtol=1e-12)
        node = int(np.flatnonzero(np.isclose(dates, 5.0))[0])
        spacing = dates[0] - dates[1]
        lumped.append(np.max(np.abs(duals.thickness[node].data)) / spacing)
    changes = [
        domain.norm(domain.subtract(finer, coarser))
        for coarser, finer in zip(kernels[:-1], kernels[1:])
    ]
    assert changes[2] < 0.5 * changes[1] < 0.25 * changes[0]
    assert lumped[3] > 1.6 * lumped[2] > 2.56 * lumped[1]


def test_what_is_refused(setting):
    material, earth, _, _, _ = setting
    evolution = _evolution(material, earth)
    domain = _domain(earth, "sobolev")
    dates = np.linspace(10.0, 0.0, 5)
    with pytest.raises(ValueError, match="not on the run's dates"):
        IceHistoryOperator(evolution, dates, domain, [4.0])
    with pytest.raises(ValueError, match="repeated"):
        IceHistoryOperator(evolution, dates, domain, [5.0, 5.0])
    with pytest.raises(ValueError):
        IceHistoryOperator(evolution, np.linspace(12.0, 0.0, 5), domain, [0.0])
    other = FieldHistorySpace(
        domain.time_space, lebesgue_load_space(EarthModel(LMAX + 2))
    )
    with pytest.raises(ValueError):
        IceHistoryOperator(evolution, dates, other, [0.0])


def test_a_gnss_rate_observable(setting):
    """GNSS velocities in the nuisance frame, the displacement rate over the
    last five kiloyears of the ice history operator, pass the adjoint check
    on the direct sum of the history space and the frame's rates."""
    material, earth, _, _, _ = setting
    b = earth.parameters.mean_sea_floor_radius
    space = sobolev_response_space(earth, 2.0, 0.3 * b, horizontal=True)
    operator = IceHistoryOperator(
        _evolution(material, earth),
        np.linspace(10.0, 0.0, 5),
        _domain(earth, "sobolev", field_kind="sobolev"),
        [5.0, 0.0],
        space=space,
        rtol=1e-12,
    )
    displacement = space.projection(DISPLACEMENT)
    past, present = (displacement @ operator.codomain.projection(i) for i in (0, 1))
    rate = (1.0 / 5.0) * (present - past) @ operator
    points = [(45.0, -60.0), (-33.8, 151.2), (51.5, 0.0), (10.0, 80.0)]
    model = GnssObservationModel(rate, points, frame="nuisance", noise=1.0)
    assert model.model_space.subspaces[0] == operator.domain
    check_operator(
        model.forward_operator, rng=np.random.default_rng(0), trials=2, rtol=1e-8
    )
