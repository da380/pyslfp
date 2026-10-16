"""The per-step weight algebra of the viscoelastic time-stepping schemes.

Everything is a function of h = dt / tau, evaluated per node and per
branch.  The exponential trapezoid update of an internal variable m
driven by a strain scalar d is

    m^{n+1} = e m^n + alpha d^n + beta d^{n+1},

with e = exp(-h), phi1 = (1 - e) / h, alpha = phi1 - e and
beta = 1 - phi1; phi1 is series-expanded below h = 1e-3, where the
direct quotient loses digits.  The backward Euler update is

    m^{n+1} = (m^n + h d^{n+1}) / (1 + h),

and the exponential Euler update is the trapezoid's with the strain
frozen at the start, m^{n+1} = e m^n + (1 - e) d^n.  Eliminating
m^{n+1} from the step's equilibrium leaves an effective shear modulus
mu_inf + sum_k w_k mu_k whose weight w_k is 1 / (1 + h) for backward
Euler and 1 - beta = phi1 for the trapezoid.

An elastic region is h = 0 (tau infinite, stored as itau = 0): e = 1,
alpha = beta = 0 and both effective weights are 1, so the variable
stays put and the effective modulus is the unrelaxed one.  The weights
lie in (0, 1] at the nodes where they are evaluated; anything that
ever interpolates them between nodes must clamp to that interval.

This is a port of AdGIA's `detail::ExponentialTrapezoidWeights` and
the scheme eliminations of `viscoelastic.cpp`; the two must stay
algebraically identical for the cross-benchmarks to isolate spatial
discretisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "exponential_trapezoid_weights",
    "backward_euler_weight",
    "exponential_trapezoid_weight_derivatives",
    "backward_euler_weight_derivative",
]

_SERIES_SWITCH = 1e-3


def _phi1(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(1 - exp(-h)) / h, by the quotient away from zero and by the
    alternating series below the switch, exact at h = 0."""
    small = np.abs(h) < _SERIES_SWITCH
    hs = np.where(small, h, 1.0)  # keep the series arguments benign
    series = 1.0 - hs / 2.0 + hs**2 / 6.0 - hs**3 / 24.0 + hs**4 / 120.0
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = (1.0 - e) / np.where(small, 1.0, h)
    return np.where(small, series, quotient)


def exponential_trapezoid_weights(
    h: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(e, alpha, beta) of the exponential trapezoid update for
    h = dt / tau, elementwise: e = exp(-h), alpha = phi1 - e,
    beta = 1 - phi1.  The effective-modulus weight is 1 - beta = phi1.
    Negative h gives the weights of the EXACT inverse step: the update
    at -dt with the start and end strains swapped, which is how a
    checkpointing driver reconstructs the forward trajectory during a
    reverse sweep; dissipation then amplifies roundoff by exp(abs h)."""
    h = np.asarray(h, dtype=float)
    e = np.exp(-h)
    phi1 = _phi1(h, e)
    return e, phi1 - e, 1.0 - phi1


def backward_euler_weight(h: ArrayLike) -> np.ndarray:
    """The backward Euler weight 1 / (1 + h) for h = dt / tau >= 0,
    elementwise: both the effective-modulus weight and the factor on
    the start state in the memory force."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("h = dt / tau must be non-negative")
    return 1.0 / (1.0 + h)


def _dphi1(h: np.ndarray, e: np.ndarray, phi1: np.ndarray) -> np.ndarray:
    """d phi1 / dh = (e - phi1) / h, by the quotient above the switch
    and by the alternating series below it, exact at h = 0."""
    small = np.abs(h) < _SERIES_SWITCH
    hs = np.where(small, h, 1.0)
    series = -0.5 + hs / 3.0 - hs**2 / 8.0 + hs**3 / 30.0 - hs**4 / 144.0
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = (e - phi1) / np.where(small, 1.0, h)
    return np.where(small, series, quotient)


def exponential_trapezoid_weight_derivatives(
    h: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(de, dalpha, dbeta), the h-derivatives of the exponential
    trapezoid weights, elementwise for h = dt / tau >= 0: de = -e,
    dalpha = dphi1 - de and dbeta = -dphi1, so the effective-modulus
    weight's derivative is dphi1 = -dbeta.  A viscosity sensitivity
    needs these at fixed dt, where dh = dt d(1/tau)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("h = dt / tau must be non-negative")
    e = np.exp(-h)
    phi1 = _phi1(h, e)
    dphi1 = _dphi1(h, e, phi1)
    return -e, dphi1 + e, -dphi1


def backward_euler_weight_derivative(h: ArrayLike) -> np.ndarray:
    """d/dh of the backward Euler weight: -1 / (1 + h)^2."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0.0):
        raise ValueError("h = dt / tau must be non-negative")
    return -1.0 / (1.0 + h) ** 2
