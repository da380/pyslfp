"""Objective functionals defined through subsidiary fits.

A decay time is not a field value: it is the parameter of a curve
fitted to a sea-level or displacement time-series, so its sensitivity
to a model comes from differentiating the fit's stationarity
conditions (the implicit function theorem), which turns the fitted
parameter into time-dependent weights on the series -- and behind
those weights every kernel driver applies unchanged.

The datum only exists where the physics delivers it: rebound curves
look exponential close to the centre of a former ice sheet and only
AFTER the melting has ceased, so the site functional belongs in the
near field and the fit window strictly after the load stops changing
(Lambeck 2006 discusses the appropriateness of such measurements).
Outside that regime the fit lands on the near-degenerate ridge of
long decay times and is refused.

The fit here follows Al-Attar & Tromp (2014, Appendix C): the series
is modelled as a e^(-s t) + b, and the model times join the parameters
with Gaussian penalties towards the nominal times, accounting for the
timing uncertainty of the data (Mitrovica & Peltier 1993).  That
treatment is one construction among several -- the uncertainties are
explicit arguments so the choice stays visible, and a zero time
uncertainty pins the times and recovers the plain three-parameter fit.
Applied to synthetic data, the estimator (not the abstract decay time)
is the datum, so a comparison with observations should fit both with
the same uncertainties.  The stationarity problem is solved
by a variable-projection seed (linear in amplitude and offset at each
trial rate) polished by trust-region Gauss-Newton, the stabilised
counterpart of a damped Newton iteration on the normal equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pygeoinf import EuclideanSpace, Functional, LinearFunctional, LinearOperator
from pyshtools import SHGrid
from scipy.optimize import least_squares

__all__ = [
    "DecayTimeFit",
    "DecayTimeFunctional",
    "fit_decay_time",
    "decay_time_measurement",
]


@dataclass(frozen=True)
class DecayTimeFit:
    """A fitted decay curve a e^(-s t) + b and the datum's exact
    sensitivity to its data: `weights[j]` is the derivative of
    `decay_time` with respect to the series value at `times[j]`, from
    the implicit differentiation of the fit's stationarity, so

        delta decay_time = sum_j weights[j] delta values[j]

    to first order.  `hessian[j, k]` is the second derivative of
    `decay_time` with respect to the series values at `times[j]` and
    `times[k]`, from differentiating the stationarity once more, so the
    second-order term is half of delta values . hessian . delta values.
    Times, the decay time and the weights share the caller's time unit;
    `model_times` are the fitted times when the timing uncertainty freed
    them, the nominal ones otherwise."""

    decay_time: float
    amplitude: float
    offset: float
    model_times: np.ndarray
    weights: np.ndarray
    misfit: float
    hessian: np.ndarray | None = None


def _curve_derivatives(
    a: float, s: float, tm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first, second and third derivatives of f = a e^(-s t) + b at
    each sample, in the sample's own variables (a, b, s, t): arrays of
    shape (n, 4), (n, 4, 4) and (n, 4, 4, 4), symmetric in the
    derivative indices."""
    n = tm.size
    E = np.exp(-s * tm)
    A, S, T = 0, 2, 3  # the offset b (index 1) enters linearly
    g = np.zeros((n, 4))
    g[:, A] = E
    g[:, 1] = 1.0
    g[:, S] = -a * tm * E
    g[:, T] = -a * s * E

    h = np.zeros((n, 4, 4))

    def put2(i, j, value):
        h[:, i, j] = value
        h[:, j, i] = value

    put2(A, S, -tm * E)
    put2(A, T, -s * E)
    put2(S, S, a * tm**2 * E)
    put2(S, T, a * E * (s * tm - 1.0))
    put2(T, T, a * s**2 * E)

    d3 = np.zeros((n, 4, 4, 4))

    def put3(i, j, k, value):
        for p, q, r in {
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        }:
            d3[:, p, q, r] = value

    put3(A, S, S, tm**2 * E)
    put3(A, S, T, E * (s * tm - 1.0))
    put3(A, T, T, s**2 * E)
    put3(S, S, S, -a * tm**3 * E)
    put3(S, S, T, a * tm * E * (2.0 - s * tm))
    put3(S, T, T, a * s * E * (2.0 - s * tm))
    put3(T, T, T, -a * s**3 * E)
    return g, h, d3


def _rate_hessian(
    a: float,
    s: float,
    tm: np.ndarray,
    r: np.ndarray,
    wd: np.ndarray,
    H: np.ndarray,
    dq_dd: np.ndarray,
    free_times: bool,
) -> np.ndarray:
    """The second derivative of the fitted rate s with respect to the
    data.  Differentiating the stationarity F(q, d) = 0 twice gives

        H q''[u, v] = -(F_qq[q'u, q'v] + F_qd[q'u, v] + F_qd[q'v, u]),

    the misfit being quadratic in the data, and the rate's row of
    H^{-1} turns the right-hand side into a contraction of the misfit's
    third derivative with (z, q'u, q'v), z = H^{-1} e_s.  The time
    penalties are quadratic in the parameters and drop out."""
    n = tm.size
    g, h, d3 = _curve_derivatives(a, s, tm)
    nloc = 4 if free_times else 3
    g, h, d3 = g[:, :nloc], h[:, :nloc, :nloc], d3[:, :nloc, :nloc, :nloc]

    z = np.linalg.solve(H, np.eye(H.shape[0])[2])

    # each sample's own variables: (a, b, s) shared, and its model time
    def local(v):
        out = np.empty((n, nloc) + v.shape[1:])
        out[:, :3] = v[:3][None]
        if free_times:
            out[:, 3] = v[3:]
        return out

    zl = local(z)  # (n, nloc)
    ql = local(dq_dd)  # (n, nloc, n): sample, variable, datum

    gz = wd * np.einsum("ip,ip->i", g, zl)
    gq = wd[:, None] * np.einsum("ip,ipj->ij", g, ql)
    hq = wd[:, None, None] * np.einsum("ipq,iqj->ipj", h, ql)
    zhq = np.einsum("ip,ipj->ij", zl, hq)
    qhq = np.einsum("ipj,ipk->ijk", ql, hq)
    third = np.einsum("i,ijk->jk", gz, qhq)
    cross = np.einsum("ij,ik->jk", gq, zhq)
    third += cross + cross.T
    third += np.einsum("i,ipqr,ip,iqj,irk->jk", r * wd, d3, zl, ql, ql, optimize=True)
    # F_qd[q'u, e_k] = -wd_k (whitened second derivative of sample k) q'u
    mixed = -wd[:, None] * zhq  # [k, j]
    return -(third + mixed + mixed.T)


def _initial_guess(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """A variable-projection start: for each trial rate the amplitude
    and offset are the linear least-squares pair, and the best trial
    seeds the full fit."""
    span = t[-1] - t[0]
    best = None
    for s in np.geomspace(0.1 / span, 30.0 / span, 60):
        E = np.exp(-s * (t - t[0]))
        A = np.column_stack((E, np.ones_like(t)))
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = A @ coeff - y
        rss = float(r @ r)
        if best is None or rss < best[0]:
            best = (rss, coeff[0] * np.exp(s * t[0]), coeff[1], s)
    _, a, b, s = best
    return a, b, s


def fit_decay_time(
    times: np.ndarray,
    values: np.ndarray,
    *,
    time_uncertainty: float | np.ndarray,
    value_uncertainty: float | np.ndarray,
) -> DecayTimeFit:
    """Fits values ~ a e^(-s times) + b and returns the decay time 1/s
    with its sensitivity to the values.

    Both uncertainties are required: only their ratio shapes the fit
    and the weights, and a value uncertainty far from the series'
    scale drowns one misfit term in the other -- there is no default
    that survives a change of units.  With a non-zero
    `time_uncertainty` the model times are free
    parameters penalised towards the nominal ones, the treatment of
    timing errors of Al-Attar & Tromp (2014, Appendix C; Mitrovica &
    Peltier 1993); zero pins them.  Only the ratio of the two
    uncertainties matters to the fit and to the weights.  `times` must
    be increasing and in the unit the decay time is wanted in; the
    fitted inverse rate must come out positive, else the window holds
    no decay and the fit is refused.

    Args:
        times: The nominal sample times, increasing.
        time_uncertainty: One-sigma timing error, a scalar or one per
            sample; zero fixes the times.
        value_uncertainty: One-sigma series error, a scalar or one per
            sample.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.size < 4 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be increasing, with at least four entries")
    if y.shape != t.shape:
        raise ValueError("values must match times")
    sig_d = np.broadcast_to(np.asarray(value_uncertainty, dtype=float), t.shape)
    sig_t = np.broadcast_to(np.asarray(time_uncertainty, dtype=float), t.shape)
    if np.any(sig_d <= 0.0) or np.any(sig_t < 0.0):
        raise ValueError(
            "value uncertainties must be positive and time "
            "uncertainties non-negative"
        )
    free_times = bool(np.any(sig_t > 0.0))
    if free_times and np.any(sig_t == 0.0):
        raise ValueError("time uncertainties must be all zero or all positive")
    n = t.size
    wd = 1.0 / sig_d

    a0, b0, s0 = _initial_guess(t, y)

    if free_times:
        wt = 1.0 / sig_t

        def residual(q):
            a, b, s = q[:3]
            tm = q[3:]
            return np.concatenate((wd * (a * np.exp(-s * tm) + b - y), wt * (tm - t)))

        def jacobian(q):
            a, b, s = q[:3]
            tm = q[3:]
            E = np.exp(-s * tm)
            J = np.zeros((2 * n, 3 + n))
            J[:n, 0] = wd * E
            J[:n, 1] = wd
            J[:n, 2] = -wd * a * tm * E
            J[np.arange(n), 3 + np.arange(n)] = -wd * a * s * E
            J[n + np.arange(n), 3 + np.arange(n)] = wt
            return J

        q0 = np.concatenate(([a0, b0, s0], t))
    else:

        def residual(q):
            a, b, s = q
            return wd * (a * np.exp(-s * t) + b - y)

        def jacobian(q):
            a, b, s = q
            E = np.exp(-s * t)
            return np.column_stack((wd * E, wd, -wd * a * t * E))

        q0 = np.array([a0, b0, s0])

    result = least_squares(
        residual, q0, jac=jacobian, xtol=1e-15, ftol=1e-15, gtol=1e-14
    )
    a, b, s = result.x[:3]
    tm = result.x[3:] if free_times else t.copy()
    if s <= 0.0:
        raise ValueError(
            "the window does not constrain a decay: the fitted rate " "is not positive"
        )
    if s <= 0.0 or s * (t[-1] - t[0]) < 1e-2:
        # the s -> 0 ridge: a e^(-s t) + b degenerates to a line with
        # enormous compensating amplitude and offset, and the window
        # constrains no decay
        raise ValueError(
            "the window does not constrain a decay: the fitted decay "
            "time is not positive and well below a hundred window spans"
        )

    # the exact sensitivity: stationarity gives
    # delta q = -H^{-1} (grad_q grad_d chi) delta d, with H the FULL
    # Hessian of the misfit (second-order residual terms included)
    nq = 3 + n if free_times else 3

    def stationarity(q):
        """The whitened residual, the whitened model derivatives df/dq
        per sample, and the misfit's gradient and full Hessian."""
        a, b, s = q[:3]
        tm = q[3:] if free_times else t
        E = np.exp(-s * tm)
        r = wd * (a * E + b - y)
        df = np.zeros((n, nq))
        df[:, 0] = wd * E
        df[:, 1] = wd
        df[:, 2] = -wd * a * tm * E
        if free_times:
            df[np.arange(n), 3 + np.arange(n)] = -wd * a * s * E
        H = df.T @ df
        # second derivatives of f, contracted with the residual
        H[0, 2] -= float(np.sum(r * wd * tm * E))
        H[2, 0] = H[0, 2]
        H[2, 2] += float(np.sum(r * wd * a * tm**2 * E))
        if free_times:
            idx = 3 + np.arange(n)
            H[0, idx] -= r * wd * s * E
            H[idx, 0] = H[0, idx]
            H[2, idx] += r * wd * a * E * (s * tm - 1.0)
            H[idx, 2] = H[2, idx]
            H[idx, idx] += r * wd * a * s**2 * E + (1.0 / sig_t) ** 2
        gradient = df.T @ r
        if free_times:
            gradient[3:] += (tm - t) / sig_t**2
        return r, df, gradient, H

    q = np.array(result.x, dtype=float)
    r, df, gradient, H = stationarity(q)
    newton = np.linalg.solve(H, gradient)
    if abs(newton[2]) > 1e-8 * s:
        raise ValueError(
            "the fit did not reach stationarity; check that the two "
            "uncertainties sit near the scales of the series and times"
        )
    # Newton steps from there put the fit on the stationary point to
    # rounding: the sensitivities differentiate the stationarity, and
    # are exact only where it holds
    for _ in range(3):
        q -= newton
        r, df, gradient, H = stationarity(q)
        newton = np.linalg.solve(H, gradient)
    a, b, s = q[:3]
    tm = q[3:].copy() if free_times else t.copy()
    misfit = 0.5 * float(r @ r)
    if free_times:
        misfit += 0.5 * float(np.sum(((tm - t) / sig_t) ** 2))
    # grad_q grad_d chi: each data value enters its residual alone
    B = -df.T * wd[None, :]
    dq_dd = -np.linalg.solve(H, B)
    ds_dd = dq_dd[2]
    weights = -ds_dd / s**2
    # tau = 1 / s, so tau'' = -s'' / s^2 + 2 s' s'^T / s^3
    d2s = _rate_hessian(a, s, tm, r, wd, H, dq_dd, free_times)
    hessian = -d2s / s**2 + 2.0 * np.outer(ds_dd, ds_dd) / s**3
    hessian = 0.5 * (hessian + hessian.T)

    return DecayTimeFit(
        decay_time=1.0 / s,
        amplitude=a,
        offset=b,
        model_times=tm,
        weights=weights,
        misfit=misfit,
        hessian=hessian,
    )


class DecayTimeFunctional(Functional):
    """The decay time of a sampled series as a functional of its values.

    The domain is the Euclidean space of the series' values at the fit
    times; the value is the decay time of :func:`fit_decay_time`, the
    gradient its implicit-function weights and the Hessian the second
    derivative of the same implicit function.  Composed with an operator
    that samples a response at the fit times, it is the decay time as a
    nonlinear objective, with the composition's gradient and Hessian by
    the chain rule.

    Each point costs one fit, kept for the value, the gradient and the
    Hessian at that point.

    Args:
        times: The nominal sample times, increasing, in the unit the
            decay time is wanted in.
        time_uncertainty: One-sigma timing error, a scalar or one per
            sample; zero fixes the times.
        value_uncertainty: One-sigma series error, a scalar or one per
            sample.
    """

    def __init__(
        self,
        times: np.ndarray,
        /,
        *,
        time_uncertainty: float | np.ndarray,
        value_uncertainty: float | np.ndarray,
    ) -> None:
        self._times = np.array(times, dtype=float)
        if self._times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        self._time_uncertainty = time_uncertainty
        self._value_uncertainty = value_uncertainty
        self._point: np.ndarray | None = None
        self._fit: DecayTimeFit | None = None
        super().__init__(EuclideanSpace(self._times.size))

    @property
    def times(self) -> np.ndarray:
        """The nominal sample times."""
        return self._times.copy()

    def fit(self, values: np.ndarray, /) -> DecayTimeFit:
        """The fit at a series, with the fitted curve's parameters."""
        values = np.asarray(values, dtype=float)
        if self._point is None or not np.array_equal(values, self._point):
            self._fit = fit_decay_time(
                self._times,
                values,
                time_uncertainty=self._time_uncertainty,
                value_uncertainty=self._value_uncertainty,
            )
            self._point = values.copy()
        return self._fit

    @property
    def has_derivative(self) -> bool:
        """Always true: the fit's weights are the gradient."""
        return True

    def _value(self, x: np.ndarray) -> float:
        return float(self.fit(x).decay_time)

    def _derivative(self, x: np.ndarray) -> LinearFunctional:
        return LinearFunctional.from_representer(self.domain, self.fit(x).weights)

    def _hessian(self, x: np.ndarray) -> LinearOperator:
        return LinearOperator.from_matrix(
            self.domain, self.domain, self.fit(x).hessian, form="components"
        )


def decay_time_measurement(
    dates: np.ndarray,
    series: np.ndarray,
    spatial_dual: SHGrid,
    *,
    time_uncertainty: float | np.ndarray,
    value_uncertainty: float | np.ndarray,
) -> tuple[DecayTimeFit, dict[float, SHGrid]]:
    """The decay time of a site's sea-level series as a kernel-ready
    objective.

    `series[j]` is the functional int spatial_dual xi(date_j) dS read
    off a forward run's snapshots over the fit window, and `dates` are
    its kiloyears before present, decreasing; the fit runs in elapsed
    time, so the decay time comes out in kiloyears.  Returns the fit
    and the measurement dictionary {date_j: weights[j] spatial_dual}
    whose kernels, through any adjoint driver, are the decay time's.
    """
    dates = np.asarray(dates, dtype=float)
    if dates.ndim != 1 or np.any(np.diff(dates) >= 0.0):
        raise ValueError("dates must be decreasing (kiloyears before present)")
    fit = fit_decay_time(
        dates[0] - dates,
        series,
        time_uncertainty=time_uncertainty,
        value_uncertainty=value_uncertainty,
    )
    measurement = {
        float(date): SHGrid.from_array(w * spatial_dual.data, grid=spatial_dual.grid)
        for date, w in zip(dates, fit.weights)
    }
    return fit, measurement
