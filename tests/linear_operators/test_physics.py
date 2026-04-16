"""
Test suite for the FingerPrintOperator and core physical mappings.
"""

import pytest
import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators.physics import (
    FingerPrintOperator,
    centrifugal_potential_operator,
)

CHECK_RTOL = 1e-6
CHECK_ATOL = 1e-6

# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_lmax():
    """
    Sets the spherical harmonic truncation degree for the operator tests.
    Kept low (lmax=16) to ensure the adjoint checks (which run multiple
    forward/adjoint solves) execute quickly.
    """
    return 16


# ==================================================================== #
#                  1. Operator Initialization & Typing                 #
# ==================================================================== #


def test_lebesgue_operator_creation(operator_lmax):
    """
    Smoke test to ensure the FingerPrintOperator initializes correctly
    as a standard L2 (Lebesgue) operator when no Sobolev parameters are given.
    """
    op = FingerPrintOperator.for_testing(operator_lmax)

    assert isinstance(op, inf.LinearOperator)
    assert isinstance(op, FingerPrintOperator)
    # Check that the default domain is indeed a Lebesgue space
    assert "Lebesgue" in op.domain.__class__.__name__


def test_sobolev_operator_creation(operator_lmax):
    """
    Smoke test to ensure the FingerPrintOperator initializes correctly
    as a regularized Sobolev operator.
    """
    # Use order 2 for the load, meaning response should be order 3
    op = FingerPrintOperator.for_testing(
        operator_lmax, load_parameters=(2.0, 0.1), response_parameters=(3.0, 0.1)
    )

    assert isinstance(op, inf.LinearOperator)
    # Check that the domain is properly registered as a Sobolev space
    assert "Sobolev" in op.domain.__class__.__name__


def test_invalid_sobolev_orders(operator_lmax):
    """
    Due to elliptic regularity, the response fields gain exactly one
    Sobolev order over the loads. The operator must reject invalid pairs.
    """
    with pytest.raises(
        ValueError, match="cannot be greater than one plus the load order"
    ):
        # Load order 2 means response order cannot exceed 3. Asking for 4 should fail.
        FingerPrintOperator.for_testing(
            operator_lmax, load_parameters=(2.0, 0.1), response_parameters=(4.0, 0.1)
        )


# ==================================================================== #
#                  2. Mathematical Adjoint Identities                  #
# ==================================================================== #


@pytest.mark.parametrize(
    "sobolev, order, scale_factor, rotational_feedbacks",
    [
        # 1. Lebesgue (L2), no rotational feedbacks
        (False, None, None, False),
        # 2. Lebesgue (L2), with rotational feedbacks (Polar Wander)
        (False, None, None, True),
        # 3. Sobolev (Order 1), no rotational feedbacks
        (True, 1.0, 0.1, False),
        # 4. Sobolev (Order 2), with rotational feedbacks
        (True, 2.0, 0.2, True),
    ],
    ids=[
        "Lebesgue-NoRotation",
        "Lebesgue-Rotation",
        "Sobolev-O1-NoRotation",
        "Sobolev-O2-Rotation",
    ],
)
def test_fingerprint_operator_rigorous_checks(
    operator_lmax,
    sobolev: bool,
    order: float,
    scale_factor: float,
    rotational_feedbacks: bool,
):
    """
    Tests the mathematical properties of the FingerPrintOperator using
    pygeoinf's built-in .check() method.

    This autonomously verifies:
    1. Linearity: A(x + y) = Ax + Ay
    2. Adjoint Identity: <Ax, y> = <x, A*y>

    It uses spatially regular Gaussian measures to ensure the random
    test fields are physically smooth enough for the solver to converge.
    """
    rtol = 1e-9

    # 1. Construct the operator
    if sobolev:
        # For Sobolev spaces, the response order must be <= load_order + 1
        load_params = (order, scale_factor)
        resp_params = (order + 1.0, scale_factor)

        A = FingerPrintOperator.for_testing(
            operator_lmax,
            load_parameters=load_params,
            response_parameters=resp_params,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
    else:
        A = FingerPrintOperator.for_testing(
            operator_lmax, rotational_feedbacks=rotational_feedbacks, rtol=rtol
        )

    # 2. Fetch the physically-scaled testing measures directly from the operator
    domain_measure = A.load_measure_for_testing()
    codomain_measure = A.response_measure_for_testing()

    # 3. Run the comprehensive self-checks
    # We use n_checks=2 to keep the test suite fast while still proving consistency
    A.check(
        n_checks=2,
        check_rtol=CHECK_RTOL,
        check_atol=CHECK_ATOL,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


# ==================================================================== #
#                  3. Sub-Operator Adjoint Identities                  #
# ==================================================================== #


@pytest.fixture(scope="module")
def testing_state(operator_lmax):
    """Provides a shared EarthState for the sub-operator tests."""
    return EarthState.for_testing(operator_lmax)


@pytest.mark.parametrize("sobolev", [False, True])
def test_centrifugal_potential_operator(testing_state, sobolev):
    """Tests the operator mapping angular velocity to centrifugal potential."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        sobolev_params = (1.0, 0.1 * b)
    else:
        sobolev_params = None

    A = centrifugal_potential_operator(model, sobolev_parameters=sobolev_params)

    amc_std = model.parameters.rotation_frequency * b**4
    domain_measure = inf.GaussianMeasure.from_standard_deviation(A.domain, amc_std)
    codomain_measure = A.codomain.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=CHECK_RTOL,
        check_atol=CHECK_ATOL,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
