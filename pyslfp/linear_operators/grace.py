"""
Observation models and spectral estimators for GRACE gravimetry.

This module provides two distinct ways to link surface mass loads to
gravitational potential changes:

1. GraceObservationModel: The exact, iterative elastic Sea Level Equation.
2. WMBMethod: The fast, purely spectral approximation (Wahr, Molenaar, & Bryan, 1998),
   highly effective for building data-space preconditioners.

Both report the potential change either in pyslfp's own coefficient
convention, the default, or as the dimensionless Stokes coefficients that
GRACE Level-2 products publish; see ``stokes_coefficient_operator``.
"""

from __future__ import annotations
from typing import Tuple, Union

import numpy as np

from pygeoinf import (
    LinearOperator,
    DirectSum,
    EuclideanSpace,
    GaussianMeasure,
    DiagonalLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from planetmodel import units

from pyslfp.core import EarthModel
from pyslfp.linear_operators.utils import check_response_space, check_load_space
from pyslfp.linear_operators.physics import FingerPrintOperator


# ================================================================ #
#                 Coefficient Convention                           #
# ================================================================ #
#
# The GRACE operators report a field u by the coefficients c_lm of
#
#     u = sum_lm c_lm Y_lm,
#
# with Y_lm the real spherical harmonics orthonormal on the unit sphere
# (pyshtools' "ortho" normalisation, int Y_lm^2 dOmega = 1) and without the
# Condon-Shortley phase: pyslfp's own convention. Coefficients are ordered
# by degree and, within a degree, m = 0, 1, ..., l, -1, ..., -l, a negative
# order standing for the sine term of order |m|.
#
# A space on a sphere of radius b uses components relative to Y_lm / b,
# which is orthonormal there, so its components are b c_lm. The factor 1/b
# below makes the reported coefficients independent of the radius, and so of
# the units the space is expressed in.


def _unit_sphere_coefficient_operator(
    space: Union[Lebesgue, Sobolev], max_degree: int, min_degree: int, /
) -> LinearOperator:
    """Maps a field to its coefficients in the convention above."""
    return (1.0 / space.radius) * space.coefficient_operator(
        lmax=max_degree, lmin=min_degree
    )


def _unit_sphere_synthesis_operator(
    space: Union[Lebesgue, Sobolev], max_degree: int, min_degree: int, /
) -> LinearOperator:
    """Maps coefficients in the convention above to the field they define."""
    return space.radius * space.from_coefficient_operator(
        lmax=max_degree, lmin=min_degree
    )


# ================================================================ #
#                 Stokes Coefficients                              #
# ================================================================ #

# The constants to which the CSR and JPL GRACE and GRACE-FO Level-2
# releases refer their Stokes coefficients, in SI. GFZ uses the same GM
# with a reference radius of 6378136.46 m; a product's header gives its own.
GRACE_GRAVITATIONAL_PARAMETER: float = 3.986004415e14  # m^3 s^-2
GRACE_REFERENCE_RADIUS: float = 6378136.3  # m

_CONVENTIONS = ("orthonormal", "stokes")


def stokes_coefficient_operator(
    model: EarthModel,
    max_obs_degree: int,
    /,
    *,
    min_obs_degree: int = 2,
    gravitational_parameter: float = GRACE_GRAVITATIONAL_PARAMETER,
    reference_radius: float = GRACE_REFERENCE_RADIUS,
) -> DiagonalLinearOperator:
    """
    Converts coefficients of the potential change, in this module's
    convention, into the Stokes coefficients of GRACE Level-2 products.

    Those products write the geopotential outside the Earth as

        V(r) = (GM/a) sum_lm (a/r)^(l+1) (dC_lm cos m lambda + dS_lm sin m lambda) P_lm,

    with a and GM the product's reference radius and gravitational
    parameter, and P_lm fully normalised in the geodetic sense: the
    harmonics have mean square one over the sphere (pyshtools' "4pi"
    normalisation), without the Condon-Shortley phase. V is positive, while
    pyslfp's potential phi is negative near added mass, so V = -phi. Given
    the coefficients phi_lm of phi on the sphere of radius b, orthonormal on
    the unit sphere, matching the two at r = b gives

        dC_lm = -(a / GM) (b / a)^(l+1) phi_lm / sqrt(4 pi),

    and the same for dS_lm, which sits at the negative orders. The factor
    (b / a)^(l+1) continues the potential up from b to the reference
    radius. Here b is the model's mean sea floor radius, the radius of
    pyslfp's field spaces and of its Love numbers. The result is
    dimensionless, so it does not depend on pyslfp's units.

    Args:
        model: The Earth model, supplying b and the units.
        max_obs_degree: The maximum degree of the observations.
        min_obs_degree: The minimum degree of the observations.
        gravitational_parameter: The product's GM, in m^3 s^-2.
        reference_radius: The product's reference radius a, in metres.

    Returns:
        The diagonal operator on the observation vector, one factor per
        degree.
    """
    scales = model.parameters.scales
    gm = gravitational_parameter / scales.factor(units.Dimensions(length=3, time=-2))
    a = reference_radius / scales.factor(units.LENGTH)
    b = model.parameters.mean_sea_floor_radius

    degrees = np.concatenate(
        [np.full(2 * l + 1, l) for l in range(min_obs_degree, max_obs_degree + 1)]
    )
    factors = -(a / gm) * (b / a) ** (degrees + 1) / np.sqrt(4.0 * np.pi)
    return DiagonalLinearOperator(EuclideanSpace(degrees.size), factors)


def _check_convention(convention: str, /) -> None:
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_CONVENTIONS}, not {convention!r}."
        )


# ================================================================ #
#                 Exact Physical Model (SLE)                       #
# ================================================================ #


def grace_observation_operator(
    response_space: DirectSum,
    max_obs_degree: int,
    /,
    *,
    min_obs_degree: int = 2,
) -> LinearOperator:
    """
    Maps the physical response fields to a vector of spherical harmonic
    coefficients of the gravitational potential change.

    The coefficients are those of the potential change on the sphere of the
    response space, orthonormal on the unit sphere and in pyslfp's units;
    the convention is set out at the head of this module.

    Args:
        response_space: The 4-component composite response space.
       max_obs_degree: The maximum spherical harmonic degree of the observations.
        min_obs_degree: The minimum degree (usually 2, dropping degrees 0 and 1).

    Returns:
        A LinearOperator mapping from the response space to an N-dimensional
        EuclideanSpace.
    """
    check_response_space(response_space, point_values=False)
    gpc_space = response_space.subspace(2)
    spatial_to_coeffs = _unit_sphere_coefficient_operator(
        gpc_space, max_obs_degree, min_obs_degree
    )
    return spatial_to_coeffs @ response_space.projection(2)


class GraceObservationModel:
    """
    An observation model linking surface mass loads to GRACE spherical
    harmonic coefficients via the full, exact elastic Sea Level Equation.
    """

    def __init__(
        self,
        fingerprint_operator: FingerPrintOperator,
        max_obs_degree: int,
        /,
        *,
        min_obs_degree: int = 2,
        convention: str = "orthonormal",
        gravitational_parameter: float = GRACE_GRAVITATIONAL_PARAMETER,
        reference_radius: float = GRACE_REFERENCE_RADIUS,
    ):
        """
        Args:
            fingerprint_operator: The physics operator mapping loads to responses.
            max_obs_degree: The maximum spherical harmonic degree of the observations.
            min_obs_degree: The minimum degree to extract (defaults to 2).
            convention: "orthonormal" for the potential change's coefficients
                in pyslfp's convention, "stokes" for the Stokes coefficients
                of GRACE Level-2 products; see ``stokes_coefficient_operator``.
            gravitational_parameter: The product's GM in m^3 s^-2, for
                "stokes".
            reference_radius: The product's reference radius in metres, for
                "stokes".
        """
        _check_convention(convention)
        self._fingerprint_operator = fingerprint_operator
        self._max_obs_degree = max_obs_degree
        self._min_obs_degree = min_obs_degree
        self._convention = convention

        self._response_to_data_operator = grace_observation_operator(
            self.fingerprint_operator.codomain,
            max_obs_degree,
            min_obs_degree=min_obs_degree,
        )
        if convention == "stokes":
            self._response_to_data_operator = (
                stokes_coefficient_operator(
                    fingerprint_operator.model,
                    max_obs_degree,
                    min_obs_degree=min_obs_degree,
                    gravitational_parameter=gravitational_parameter,
                    reference_radius=reference_radius,
                )
                @ self._response_to_data_operator
            )

        self._forward_operator = (
            self._response_to_data_operator @ self.fingerprint_operator
        )

    @property
    def min_obs_degree(self) -> int:
        """
        Returns the minimum observation degree
        """
        return self._min_obs_degree

    @property
    def max_obs_degree(self) -> int:
        """
        Returns the maximum observation degree
        """
        return self._max_obs_degree

    @property
    def convention(self) -> str:
        """The coefficient convention of the observations."""
        return self._convention

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        """The underlying fingerprint operator."""
        return self._fingerprint_operator

    @property
    def response_to_data_operator(self) -> LinearOperator:
        """The operator from the response space to the observations."""
        return self._response_to_data_operator

    @property
    def forward_operator(self) -> LinearOperator:
        """The composed operator from model space to the observations."""
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

    With ``convention="stokes"`` the potential coefficients are the Stokes
    coefficients of GRACE Level-2 products throughout: the conversion is
    folded into the per-degree scaling, so every operator and measure below
    reports them.
    """

    def __init__(
        self,
        model: EarthModel,
        max_obs_degree: int,
        /,
        *,
        min_obs_degree: int = 2,
        convention: str = "orthonormal",
        gravitational_parameter: float = GRACE_GRAVITATIONAL_PARAMETER,
        reference_radius: float = GRACE_REFERENCE_RADIUS,
    ):
        """
        Args:
            model: The Earth model, supplying the Love numbers.
            max_obs_degree: The maximum spherical harmonic degree of the observations.
            min_obs_degree: The minimum degree of the observations.
            convention: "orthonormal" or "stokes", as for
                ``GraceObservationModel``.
            gravitational_parameter: The product's GM in m^3 s^-2, for
                "stokes".
            reference_radius: The product's reference radius in metres, for
                "stokes".
        """
        _check_convention(convention)
        self._model = model
        self._max_obs_degree = max_obs_degree
        self._min_obs_degree = min_obs_degree
        self._convention = convention

        # Total number of spherical harmonic coefficients in the observation vector
        self._observation_dim = (max_obs_degree + 1) ** 2 - min_obs_degree**2

        self._output_scaling = (
            stokes_coefficient_operator(
                model,
                max_obs_degree,
                min_obs_degree=min_obs_degree,
                gravitational_parameter=gravitational_parameter,
                reference_radius=reference_radius,
            ).eigenvalues
            if convention == "stokes"
            else np.ones(self._observation_dim)
        )

    @property
    def convention(self) -> str:
        """The coefficient convention of the observations."""
        return self._convention

    @property
    def min_obs_degree(self) -> int:
        """
        Returns the minimum observation degree
        """
        return self._min_obs_degree

    @property
    def max_obs_degree(self) -> int:
        """
        Returns the maximum observation degree
        """
        return self._max_obs_degree

    @property
    def observation_dim(self) -> int:
        """
        Returns the dimension of the data space.
        """
        return self._observation_dim

    def _observed_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """The degree and signed order of each observed coefficient, in the
        order of the observation vector."""
        degrees, orders = [], []
        for l in range(self.min_obs_degree, self.max_obs_degree + 1):
            signed = list(range(0, l + 1)) + list(range(-1, -l - 1, -1))
            degrees += [l] * len(signed)
            orders += signed
        return np.array(degrees, dtype=int), np.array(orders, dtype=int)

    # ---------------------------------------------------------#
    #             Core Euclidean Coefficient Operators         #
    # ---------------------------------------------------------#

    def load_coefficient_to_potential_coefficient_operator(
        self,
    ) -> DiagonalLinearOperator:
        """
        Maps load SH coefficients to potential SH coefficients by multiplying
        each degree by its elastic load Love number (k_l), and, for the
        "stokes" convention, by the conversion to Stokes coefficients.
        """
        domain = EuclideanSpace(self._observation_dim)
        scaling_factors = np.zeros(self._observation_dim)

        for l in range(self.min_obs_degree, self.max_obs_degree + 1):
            idx_start = l**2 - self.min_obs_degree**2
            idx_end = (l + 1) ** 2 - self.min_obs_degree**2
            scaling_factors[idx_start:idx_end] = self._model.love_numbers.k[l]

        return DiagonalLinearOperator(domain, scaling_factors * self._output_scaling)

    def potential_coefficient_to_load_coefficient_operator(
        self,
    ) -> DiagonalLinearOperator:
        """Applies the inverse Love number scaling (1 / k_l)."""
        domain = EuclideanSpace(self._observation_dim)
        values = self.load_coefficient_to_potential_coefficient_operator().eigenvalues
        return DiagonalLinearOperator(domain, np.reciprocal(values))

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

        coeffs_to_load = _unit_sphere_synthesis_operator(
            load_space, self.max_obs_degree, self.min_obs_degree
        )
        scaling_operator = self.potential_coefficient_to_load_coefficient_operator()

        return coeffs_to_load @ scaling_operator

    def load_to_potential_coefficient_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ) -> LinearOperator:
        """
        Maps a load to the corresponding potential coefficients based on the WMB theory.
        """

        check_load_space(load_space)

        load_to_coeffs = _unit_sphere_coefficient_operator(
            load_space, self.max_obs_degree, self.min_obs_degree
        )
        scaling_operator = self.load_coefficient_to_potential_coefficient_operator()

        return scaling_operator @ load_to_coeffs

    # ---------------------------------------------------------#
    #                     Bayesian Preconditioning             #
    # ---------------------------------------------------------#

    def load_measure_to_observation_measure(
        self, load_measure: GaussianMeasure
    ) -> GaussianMeasure:
        """
        Pushes an invariant Gaussian measure on the load space forward into
        the observation space, through the WMB map
        ``load_to_potential_coefficient_operator``.

        An invariant measure has the spherical harmonics as the eigenvectors
        of its covariance, C Y_k = sigma_k Y_k, the eigenvalues sigma_k being
        relative to the load space's inner product. In that inner product
        the harmonics are orthogonal with ||Y_k||^2 = g_k, where g_k = b^2
        on a Lebesgue space of radius b and b^2 (1 + L^2 l(l+1)/b^2)^s on a
        Sobolev space of order s and length scale L. The coefficient
        c_k = (u, Y_k) / g_k of a draw u therefore has variance

            E[c_k^2] = (C Y_k, Y_k) / g_k^2 = sigma_k / g_k,

        not sigma_k. The WMB map multiplies each coefficient by k_l, so the
        push-forward is the measure with independent observations of
        variance k_l^2 sigma_k / g_k.

        Above the load space's truncation the WMB operator gives zero, and
        so does the expectation. The field the measure describes does not
        stop at the truncation, though, so those observed degrees keep a
        variance: the coefficient variance of the space's highest degree,
        times k_l^2. This keeps the measure non-degenerate when it serves as
        a data error measure.

        Up to the truncation the result is the exact push-forward, formed
        from the spectrum alone, without applying the covariance.

        Raises:
            TypeError: if the measure is not invariant, that is, if its
                covariance is not diagonal in the spherical harmonics of a
                Lebesgue or Sobolev load space.
        """
        load_space = load_measure.domain
        check_load_space(load_space)
        covariance = load_measure.covariance
        if not (
            isinstance(covariance, DiagonalLinearOperator)
            and covariance.domain == load_space
        ):
            raise TypeError(
                "load_measure must be an invariant measure: its covariance "
                "must be diagonal in the spherical harmonics of the load "
                "space, as for one built by invariant_measure, sobolev_measure "
                "or heat_measure."
            )

        # The space's components refer to Y_k / b, which carry the metric
        # weight g_k / b^2; dividing it out, and b^2 for the unit-sphere
        # convention, leaves sigma_k / g_k.
        coefficient_variances = (
            covariance.eigenvalues / load_space.metric_values / load_space.radius**2
        )

        degrees, orders = self._observed_labels()
        resolved = degrees <= load_space.lmax
        variances = np.empty(self._observation_dim)
        variances[resolved] = coefficient_variances[
            load_space.component_of(degrees[resolved], orders[resolved])
        ]
        variances[~resolved] = coefficient_variances[
            load_space.component_of(load_space.lmax, 0)
        ]

        love_scaling = self.load_coefficient_to_potential_coefficient_operator()
        observed_variances = love_scaling.eigenvalues**2 * variances

        observation_space = EuclideanSpace(self._observation_dim)
        expectation = (
            None
            if load_measure.has_zero_expectation
            else self.load_to_potential_coefficient_operator(load_space)(
                load_measure.expectation
            )
        )
        precision = (
            DiagonalLinearOperator(observation_space, np.reciprocal(observed_variances))
            if np.all(observed_variances > 0.0)
            else None
        )
        return GaussianMeasure(
            observation_space,
            expectation=expectation,
            covariance_factor=DiagonalLinearOperator(
                observation_space, np.sqrt(observed_variances)
            ),
            precision=precision,
        )

    def bayesian_normal_operator_preconditioner(
        self,
        prior_measure: GaussianMeasure,
        data_error_measure: GaussianMeasure,
        /,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> DiagonalLinearOperator:
        """
        Constructs a diagonal preconditioner for the Bayesian normal
        equations, approximating the forward operator by the WMB scaling.

        The prior must be invariant; see
        ``load_measure_to_observation_measure``. A data error measure with a
        diagonal covariance gives up its diagonal at no cost; any other has
        it probed column by column, which ``parallel`` spreads over
        ``n_jobs`` workers.
        """
        mapped_prior = self.load_measure_to_observation_measure(prior_measure)

        aqa_diag = mapped_prior.covariance.diagonals()[0]
        r_diag = data_error_measure.covariance.diagonals(
            n_jobs=n_jobs if parallel else None
        )[0]

        normal_diag = aqa_diag + r_diag
        return DiagonalLinearOperator(
            data_error_measure.domain, np.reciprocal(normal_diag)
        )
