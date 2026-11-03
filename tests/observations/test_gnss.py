"""
Tests for GNSS: the Helmert design against the transformation it samples, the
frame-invariant reduction, and the observation model in each frame.
"""

import numpy as np
import pygeoinf as inf
import pytest
from pygeoinf.testing import check_operator

from pyslfp.elastic import FingerPrintOperator
from pyslfp.observations import GnssObservationModel, HelmertTransformation

POINTS = [
    (45.0, -60.0),
    (-33.8, 151.2),
    (51.5, 0.0),
    (10.0, 80.0),
    (-60.0, -20.0),
    (30.0, 120.0),
    (-10.0, -50.0),
    (70.0, 30.0),
]


@pytest.fixture(scope="module")
def fingerprint():
    """A fingerprint operator whose response holds the horizontal
    displacement, in a Sobolev space of order above 1 for point values."""
    return FingerPrintOperator.for_testing(
        16,
        load_parameters=(1.0, 0.1),
        response_parameters=(2.0, 0.1),
        rotational_feedbacks=False,
        horizontal=True,
    )


@pytest.fixture(scope="module")
def displacement_space(fingerprint):
    return fingerprint.codomain.member_space("displacement")


def _frame(point):
    """The unit radial, east and north vectors at a point."""
    lat, lon = np.radians(point)
    r = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    e = np.array([-np.sin(lon), np.cos(lon), 0.0])
    return r, e, np.cross(r, e)


# ==================================================================== #
#                       The Helmert design                             #
# ==================================================================== #


def test_the_design_samples_the_transformation():
    """T + s x + R x x at each station x = b r, in east, north and up."""
    b = 1.3
    helmert = HelmertTransformation(POINTS, b)
    assert helmert.parameter_names == (
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "scale",
    )
    h = np.random.default_rng(1).normal(size=7)
    translation, rotation, scale = h[:3], h[3:6], h[6]
    expected = []
    for point in POINTS:
        r, e, n = _frame(point)
        motion = translation + scale * b * r + np.cross(rotation, b * r)
        expected += [motion @ e, motion @ n, motion @ r]
    np.testing.assert_allclose(helmert(h), expected, atol=1e-14)


def test_a_translation_is_a_degree_one_field_with_v_equal_to_u(displacement_space):
    """A uniform translation T has U = T . r and horizontal part grad_1 of
    the same field."""
    field_space = displacement_space.field_space
    translation = np.array([0.3, -0.7, 0.4])
    grid = field_space.zero()
    lat, lon = np.meshgrid(
        np.radians(grid.lats()), np.radians(grid.lons()), indexing="ij"
    )
    grid.data = (
        translation[0] * np.cos(lat) * np.cos(lon)
        + translation[1] * np.cos(lat) * np.sin(lon)
        + translation[2] * np.sin(lat)
    )
    c = field_space.to_components(grid)
    u = displacement_space.from_components(
        np.concatenate([c, c[field_space.degrees >= 1]])
    )
    sampled = displacement_space.point_evaluation_operator(POINTS)(u)
    helmert = HelmertTransformation(
        POINTS, field_space.radius, parameters=("translation",)
    )
    np.testing.assert_allclose(sampled, helmert(translation), atol=1e-12)


def test_a_scale_is_a_degree_zero_vertical_field(displacement_space):
    field_space = displacement_space.field_space
    b, s = field_space.radius, 2.5e-3
    grid = field_space.zero()
    grid.data[:] = b * s
    c = field_space.to_components(grid)
    u = displacement_space.from_components(
        np.concatenate([c, np.zeros(np.count_nonzero(field_space.degrees >= 1))])
    )
    sampled = displacement_space.point_evaluation_operator(POINTS)(u)
    helmert = HelmertTransformation(POINTS, b, parameters=("scale",))
    np.testing.assert_allclose(sampled, helmert(np.array([s])), atol=1e-12)


def test_groups_and_directions_in_the_order_given():
    full = HelmertTransformation(POINTS, 1.0).matrix(form="components")
    part = HelmertTransformation(
        POINTS, 1.0, parameters=("scale", "translation"), directions=("up", "east")
    )
    assert part.parameter_names[0] == "scale"
    rows = full.reshape(len(POINTS), 3, 7)[:, [2, 0], :]
    np.testing.assert_allclose(
        part.matrix(form="components"),
        rows[:, :, [6, 0, 1, 2]].reshape(2 * len(POINTS), 4),
    )


def test_the_frame_reduction():
    """Z^T A = 0 and Z^T Z = I, with the adjoint exact."""
    helmert = HelmertTransformation(POINTS, 1.3)
    reduction = helmert.frame_reduction()
    assert reduction.codomain.dim == 3 * len(POINTS) - 7
    design = helmert.matrix(form="components")
    np.testing.assert_allclose(
        np.column_stack([reduction(a) for a in design.T]), 0.0, atol=1e-13
    )
    basis = np.column_stack(
        [reduction.adjoint(y) for y in np.eye(reduction.codomain.dim)]
    )
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-13)
    check_operator(reduction, rng=np.random.default_rng(0))


def test_what_the_design_refuses():
    with pytest.raises(ValueError, match="parameters"):
        HelmertTransformation(POINTS, 1.0, parameters=("translation", "shear"))
    with pytest.raises(ValueError, match="parameters"):
        HelmertTransformation(POINTS, 1.0, parameters=("scale", "scale"))
    with pytest.raises(ValueError, match="poles"):
        HelmertTransformation([(90.0, 0.0)], 1.0)
    # The vertical component alone sees no rotation.
    with pytest.raises(ValueError, match="changes none"):
        HelmertTransformation(POINTS, 1.0, directions=("up",)).frame_reduction()
    # One station gives three data for seven parameters.
    with pytest.raises(ValueError, match="3 data for 7"):
        HelmertTransformation(POINTS[:1], 1.0).frame_reduction()
    # Heights at stations of one latitude cannot tell the translation along
    # the axis from the scale.
    parallel = [(30.0, lon) for lon in (0.0, 45.0, 90.0, 180.0, 270.0)]
    with pytest.raises(ValueError, match="dependent"):
        HelmertTransformation(
            parallel, 1.0, parameters=("translation", "scale"), directions=("up",)
        ).frame_reduction()


# ==================================================================== #
#                       The observation model                          #
# ==================================================================== #


@pytest.mark.parametrize("frame", [None, "nuisance", "invariant"])
def test_the_forward_operator_in_each_frame(fingerprint, frame):
    model = GnssObservationModel.from_fingerprint(fingerprint, POINTS, frame=frame)
    n_data = 3 * len(POINTS) - (7 if frame == "invariant" else 0)
    assert model.data_space.dim == n_data
    check_operator(
        model.forward_operator, rng=np.random.default_rng(0), trials=2, rtol=1e-8
    )
    load = fingerprint.domain.heat_measure(0.3 * fingerprint.domain.radius).sample(
        rng=np.random.default_rng(1)
    )
    sampled = model.sampling_operator(model.observable(load))
    if frame == "nuisance":
        h = np.random.default_rng(2).normal(size=7)
        np.testing.assert_allclose(
            model.forward_operator((load, h)),
            sampled + model.helmert_transformation(h),
        )
        assert model.model_space.subspaces[0] == fingerprint.domain
    else:
        np.testing.assert_allclose(model.forward_operator(load), sampled)
        assert model.model_space == fingerprint.domain


def test_the_invariant_data_ignore_the_frame(fingerprint):
    """Adding any Helmert motion to the station data leaves Z^T d unchanged,
    and the noise is Z^T C Z."""
    deviations = np.linspace(0.1, 0.3, 3 * len(POINTS))
    model = GnssObservationModel.from_fingerprint(
        fingerprint, POINTS, frame="invariant", noise=deviations
    )
    rng = np.random.default_rng(3)
    data = rng.normal(size=3 * len(POINTS))
    moved = data + model.helmert_transformation(rng.normal(size=7))
    reduction = model.frame_reduction
    np.testing.assert_allclose(reduction(moved), reduction(data), atol=1e-13)
    basis = np.column_stack(
        [reduction.adjoint(y) for y in np.eye(reduction.codomain.dim)]
    )
    np.testing.assert_allclose(
        model.noise.covariance.matrix(form="components"),
        basis.T @ np.diag(deviations**2) @ basis,
        atol=1e-15,
    )


def test_the_invariant_frame_is_the_nuisance_frame_with_no_frame_prior(fingerprint):
    """As the prior on the frame widens, the nuisance frame's posterior mean
    of the load tends to the invariant frame's, as one over its variance."""
    prior = fingerprint.domain.heat_measure(0.3 * fingerprint.domain.radius)
    nuisance = GnssObservationModel.from_fingerprint(
        fingerprint, POINTS, frame="nuisance", noise=0.05
    )
    invariant = GnssObservationModel.from_fingerprint(
        fingerprint, POINTS, frame="invariant", noise=0.05
    )
    rng = np.random.default_rng(5)
    data = nuisance.forward_operator((prior.sample(rng=rng), rng.normal(size=7)))

    def mean(model, prior, data):
        problem = inf.LinearForwardProblem(model.forward_operator, error=model.noise)
        inversion = inf.LinearGaussianInversion(problem, prior)
        return inversion.with_solver(inf.CholeskySolver())(data).expectation

    target = fingerprint.domain.to_components(
        mean(invariant, prior, invariant.frame_reduction(data))
    )
    gaps = []
    for width in (10.0, 100.0, 1000.0):
        load, _ = mean(nuisance, nuisance.joint_prior(prior, width), data)
        components = fingerprint.domain.to_components(load)
        gaps.append(np.linalg.norm(components - target) / np.linalg.norm(target))
    assert gaps[2] < 1e-6
    assert 50.0 < gaps[0] / gaps[1] < 200.0


def test_the_joint_prior(fingerprint):
    model = GnssObservationModel.from_fingerprint(fingerprint, POINTS, frame="nuisance")
    prior = fingerprint.domain.heat_measure(0.3 * fingerprint.domain.radius)
    deviations = [1e-3] * 3 + [1e-4] * 3 + [1e-5]
    joint = model.joint_prior(prior, deviations)
    assert joint.domain == model.model_space
    frame = joint.affine_map(model.model_space.projection(1))
    np.testing.assert_allclose(
        frame.covariance.matrix(form="components"),
        np.diag(np.square(deviations)),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="frame prior"):
        model.joint_prior(
            prior,
            inf.GaussianMeasure.from_standard_deviation(inf.EuclideanSpace(3), 1.0),
        )
    plain = GnssObservationModel.from_fingerprint(fingerprint, POINTS)
    with pytest.raises(ValueError, match="nuisance"):
        plain.joint_prior(prior, 1.0)


def test_what_the_model_refuses(fingerprint):
    with pytest.raises(ValueError, match="surface displacement"):
        GnssObservationModel(
            fingerprint.codomain.field_operator("relative_sea_level_change")
            @ fingerprint,
            POINTS,
        )
    with pytest.raises(ValueError, match="frame"):
        GnssObservationModel.from_fingerprint(fingerprint, POINTS, frame="itrf")
    with pytest.raises(ValueError, match="station data"):
        GnssObservationModel.from_fingerprint(
            fingerprint,
            POINTS,
            frame="invariant",
            noise=inf.GaussianMeasure.from_standard_deviation(
                inf.EuclideanSpace(3 * len(POINTS) - 7), 1.0
            ),
        )
    vertical_only = FingerPrintOperator.for_testing(
        16,
        load_parameters=(1.0, 0.1),
        response_parameters=(2.0, 0.1),
        rotational_feedbacks=False,
    )
    with pytest.raises(ValueError, match="horizontal part"):
        GnssObservationModel.from_fingerprint(vertical_only, POINTS)
    model = GnssObservationModel.from_fingerprint(
        vertical_only,
        POINTS,
        directions=("up",),
        frame="invariant",
        parameters=("translation", "scale"),
    )
    assert model.data_space.dim == len(POINTS) - 4
    check_operator(model.forward_operator, rng=np.random.default_rng(0), trials=2)
