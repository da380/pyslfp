"""
Observation models and spectral estimators for GRACE gravimetry.

This module provides two distinct ways to link surface mass loads to
gravitational potential changes:
1. GraceObservationModel: The exact, iterative elastic Sea Level Equation.
2. WMBMethod: The fast, purely spectral approximation (Wahr, Molenaar, & Bryan, 1998),
   highly effective for building data-space preconditioners.
"""

from __future__ import annotations
from typing import Union

import numpy as np

from pygeoinf import (
    LinearOperator,
    HilbertSpaceDirectSum,
    EuclideanSpace,
    GaussianMeasure,
    DiagonalSparseMatrixLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev
from pygeoinf.symmetric_space.symmetric_space import InvariantGaussianMeasure

from pyslfp.core import EarthModel
from pyslfp.linear_operators.utils import check_response_space, check_load_space
from pyslfp.linear_operators.physics import FingerPrintOperator


# ================================================================ #
#                 Exact Physical Model (SLE)                       #
# ================================================================ #


def grace_observation_operator(
    response_space: HilbertSpaceDirectSum,
    obs_degree: int,
    /,
    *,
    minimum_degree: int = 2,
) -> LinearOperator:
    """
    Maps the physical response fields to a vector of spherical harmonic
    coefficients of the gravitational potential change.

    Args:
        response_space: The 4-component composite response space.
        obs_degree: The maximum spherical harmonic degree of the observations.
        minimum_degree: The minimum degree (usually 2, dropping degrees 0 and 1).

    Returns:
        A LinearOperator mapping from the response space to an N-dimensional
        EuclideanSpace.
    """
    check_response_space(response_space, point_values=False)

    # Subspace 2 is the Gravitational Potential Change
    gpc_space = response_space.subspace(2)

    # Map the spatial field to truncated SH coefficients
    spatial_to_coeffs = gpc_space.to_coefficient_operator(
        obs_degree, lmin=minimum_degree
    )

    # Combine with the subspace projection
    return spatial_to_coeffs @ response_space.subspace_projection(2)


class GraceObservationModel:
    """
    An observation model linking surface mass loads to GRACE spherical
    harmonic coefficients via the full, exact elastic Sea Level Equation.
    """

    def __init__(
        self,
        fingerprint_operator: FingerPrintOperator,
        obs_degree: int,
        /,
        *,
        minimum_degree: int = 2,
    ):
        """
        Args:
            fingerprint_operator: The physics operator mapping loads to responses.
            obs_degree: The maximum spherical harmonic degree of the observations.
            minimum_degree: The minimum degree to extract (defaults to 2).
        """
        self._fingerprint_operator = fingerprint_operator
        self._obs_degree = obs_degree
        self._minimum_degree = minimum_degree

        self._response_to_data_operator = grace_observation_operator(
            self.fingerprint_operator.codomain,
            obs_degree,
            minimum_degree=minimum_degree,
        )

        self._forward_operator = (
            self.response_to_data_operator @ self.fingerprint_operator
        )

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        return self._fingerprint_operator

    @property
    def response_to_data_operator(self) -> LinearOperator:
        return self._response_to_data_operator

    @property
    def forward_operator(self) -> LinearOperator:
        return self._forward_operator


# ================================================================ #
#                 Spectral Approximation (WMB)                     #
# ================================================================ #


class WMBMethod:
    """
    Implements the purely spectral Wahr, Molenaar, & Bryan (1998) method.

    This class bypasses the iterative Sea Level Equation, instead relying on
    load Love numbers ($k_l$) to directly map between surface mass loads and
    gravitational potential coefficients. It is highly optimized for building
    data-space preconditioners in Bayesian inversions.
    """

    def __init__(
        self, model: EarthModel, obs_degree: int, /, *, minimum_degree: int = 2
    ):
        self.model = model
        self.obs_degree = obs_degree
        self.minimum_degree = minimum_degree

        # Total number of spherical harmonic coefficients in the observation vector
        self.observation_dim = (obs_degree + 1) ** 2 - minimum_degree**2

    # ---------------------------------------------------------#
    #             Core Euclidean Coefficient Operators         #
    # ---------------------------------------------------------#

    def load_coefficient_to_potential_coefficient_operator(
        self,
    ) -> DiagonalSparseMatrixLinearOperator:
        """
        Maps load SH coefficients to potential SH coefficients by multiplying
        each degree by its elastic load Love number (k_l).
        """
        domain = EuclideanSpace(self.observation_dim)
        scaling_factors = np.zeros(self.observation_dim)

        for l in range(self.minimum_degree, self.obs_degree + 1):
            idx_start = l**2 - self.minimum_degree**2
            idx_end = (l + 1) ** 2 - self.minimum_degree**2
            scaling_factors[idx_start:idx_end] = self.model.love_numbers.k[l]

        return DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, scaling_factors
        )

    def potential_coefficient_to_load_coefficient_operator(
        self,
    ) -> DiagonalSparseMatrixLinearOperator:
        """Applies the inverse Love number scaling (1 / k_l)."""
        return self.load_coefficient_to_potential_coefficient_operator().inverse

    # ---------------------------------------------------------#
    #               Bridge Spatial/Coefficient Operators       #
    # ---------------------------------------------------------#

    def potential_coefficient_to_load_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ) -> LinearOperator:
        """
        Maps a vector of observed gravitational potential coefficients to an
        approximation of the causative continuous surface mass load.
        """
        check_load_space(load_space)

        coeffs_to_load = load_space.from_coefficient_operator(
            self.obs_degree, lmin=self.minimum_degree
        )
        scaling_operator = self.potential_coefficient_to_load_coefficient_operator()

        return coeffs_to_load @ scaling_operator

    # ---------------------------------------------------------#
    #                     Bayesian Preconditioning             #
    # ---------------------------------------------------------#

    def load_measure_to_observation_measure(
        self, load_measure: InvariantGaussianMeasure
    ) -> GaussianMeasure:
        """
        Pushes a prior Gaussian measure defined on the continuous load space
        forward into the truncated observation space by scaling with Love numbers.
        """
        if not isinstance(load_measure, InvariantGaussianMeasure):
            raise TypeError("load_measure must be an InvariantGaussianMeasure.")

        prior_variances = load_measure.spectral_variances
        prior_lmax = load_measure.domain.lmax
        max_mapped_degree = min(prior_lmax, self.obs_degree)

        observed_stds = np.zeros(self.observation_dim)

        for l in range(self.minimum_degree, max_mapped_degree + 1):
            in_start = l**2
            in_end = (l + 1) ** 2
            out_start = l**2 - self.minimum_degree**2
            out_end = (l + 1) ** 2 - self.minimum_degree**2

            prior_stds = np.sqrt(prior_variances[in_start:in_end])
            observed_stds[out_start:out_end] = prior_stds * abs(
                self.model.love_numbers.k[l]
            )

        observation_space = EuclideanSpace(self.observation_dim)
        return GaussianMeasure.from_standard_deviations(
            observation_space, observed_stds
        )

    def bayesian_normal_operator_preconditioner(
        self,
        prior_measure: InvariantGaussianMeasure,
        data_error_measure: GaussianMeasure,
        /,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> DiagonalSparseMatrixLinearOperator:
        """
        Constructs a highly efficient diagonal preconditioner for solving
        Bayesian normal equations by approximating the forward operator with WMB scaling.
        """
        mapped_prior = self.load_measure_to_observation_measure(prior_measure)

        aqa_diag = mapped_prior.covariance.extract_diagonal(
            parallel=parallel, n_jobs=n_jobs
        )
        r_diag = data_error_measure.covariance.extract_diagonal()

        normal_diag = aqa_diag + r_diag
        approx_normal_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            data_error_measure.domain, data_error_measure.domain, normal_diag
        )

        return approx_normal_op.inverse
