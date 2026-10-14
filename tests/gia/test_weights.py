"""The scheme weight algebra: bounds, the series switch, and the
prescribed-strain exactness properties the steppers rely on."""

import numpy as np

from pyslfp.gia import backward_euler_weight, exponential_trapezoid_weights


def test_limits_and_bounds():
    e, alpha, beta = exponential_trapezoid_weights(0.0)
    assert e == 1.0 and alpha == 0.0 and beta == 0.0
    assert backward_euler_weight(0.0) == 1.0

    h = np.logspace(-8, 3, 200)
    e, alpha, beta = exponential_trapezoid_weights(h)
    w = backward_euler_weight(h)
    for a in (1.0 - beta, w):
        assert np.all(a > 0.0) and np.all(a <= 1.0)
    # e underflows to exactly zero for very stiff steps, which is benign
    assert np.all(e >= 0.0) and np.all(e <= 1.0)
    assert np.all(alpha >= 0.0) and np.all(alpha < 1.0)
    assert np.all(beta >= 0.0) and np.all(beta < 1.0)
    # the update is an average: e + alpha + beta = 1
    assert np.allclose(e + alpha + beta, 1.0, rtol=1e-14)


def test_series_switch_is_seamless():
    h = 1e-3 * (1.0 + 1e-6 * np.array([-1.0, 1.0]))
    e, alpha, beta = exponential_trapezoid_weights(h)
    assert abs(alpha[0] / alpha[1] - 1.0) < 1e-5
    assert abs(beta[0] / beta[1] - 1.0) < 1e-5
    # and against an independent accurate evaluation: expm1 keeps full
    # precision through the small-h cancellation.  phi1 = 1 - beta is
    # accurate to round-off everywhere; beta itself loses a few digits
    # just above the switch, where the quotient's cancellation error is
    # amplified by phi1 / beta -- the port keeps AdGIA's grade.
    hs = np.logspace(-8, 1, 50)
    _, _, beta_s = exponential_trapezoid_weights(hs)
    phi1 = -np.expm1(-hs) / hs
    assert np.allclose(1.0 - beta_s, phi1, rtol=1e-13)
    assert np.allclose(beta_s, 1.0 - phi1, rtol=1e-9)


def test_exponential_euler_exact_for_constant_strain():
    """m' = (d - m)/tau with d held: the one-step update equals the
    exact propagator whatever the step."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        tau, d, m0, dt = rng.uniform(0.1, 10.0, size=4)
        e, _, _ = exponential_trapezoid_weights(dt / tau)
        stepped = e * m0 + (1.0 - e) * d
        exact = d + (m0 - d) * np.exp(-dt / tau)
        assert abs(stepped - exact) < 1e-14 * max(1.0, abs(exact))


def test_trapezoid_exact_for_linear_strain():
    """d(t) = a + b t: the exact solution from m(0) = m0 is
    m(t) = a + b t - b tau + (m0 - a + b tau) e^(-t/tau), and one
    trapezoid step reproduces it at any step size."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        tau, a, b, m0, t0, dt = rng.uniform(0.1, 5.0, size=6)

        def exact(t):
            return a + b * t - b * tau + (m0 - a + b * tau) * np.exp(-t / tau)

        e, alpha, beta = exponential_trapezoid_weights(dt / tau)
        stepped = e * exact(t0) + alpha * (a + b * t0) + beta * (a + b * (t0 + dt))
        want = exact(t0 + dt)
        assert abs(stepped - want) < 1e-12 * max(1.0, abs(want))


def test_backward_euler_formula():
    h = np.array([0.0, 0.5, 2.0, 1e4])
    m, d = 3.0, -2.0
    w = backward_euler_weight(h)
    assert np.allclose(w * m + (1.0 - w) * d, (m + h * d) / (1.0 + h), rtol=1e-15)
