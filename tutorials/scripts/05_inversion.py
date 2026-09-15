# %% [markdown]
# # 5. A Bayesian inversion of tide gauge data
#
# With the fingerprint as a linear operator, an inverse problem is a
# matter of stating a prior on the ice thickness change, a noise model on
# the data, and letting `pygeoinf` do the rest. This script inverts
# synthetic tide gauge data from the GLOSS network for the change in
# thickness of the grounded ice sheets, preconditioning the solve with a
# coarser copy of the same problem.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
from cartopy import crs as ccrs

import pyslfp as sl
from pyslfp.linear_operators import (
    FingerPrintOperator,
    TideGaugeObservationModel,
    ice_projection_operator,
    ocean_average_operator,
    read_gloss_tide_gauge_data,
)

LMAX = 128  # the model inverted
SURROGATE_LMAX = 48  # the preconditioner
SPACE_ORDER, SPACE_SCALE_KM = 2.0, 200.0  # the Sobolev load and response spaces
PRIOR_ORDER, PRIOR_SCALE_KM, PRIOR_STD_M = 3.0, 300.0, 0.5  # the prior
NOISE_STD_MM = 1.0  # the noise on each gauge

names, points = read_gloss_tide_gauge_data()
print(f"{len(points)} GLOSS stations")

# %% [markdown]
# ## The forward problem
#
# The construction is needed at two truncation degrees, so it is a
# function. The prior is a stationary Sobolev field restricted to the
# grounded ice; the forward operator maps ice thickness change to load
# and through the tide gauge model.


# %%
def build(lmax):
    state = sl.EarthState.from_defaults(lmax=lmax)
    params = state.model.parameters
    space_scale = SPACE_SCALE_KM * 1.0e3 / params.length_scale
    fingerprint = FingerPrintOperator(
        state,
        load_parameters=(SPACE_ORDER, space_scale),
        response_parameters=(SPACE_ORDER, space_scale),
    )
    ice_space = fingerprint.domain
    gauges = TideGaugeObservationModel(fingerprint, points, names=names)
    forward = gauges.forward_operator * params.ice_density
    prior = ice_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        PRIOR_ORDER,
        PRIOR_SCALE_KM * 1.0e3 / params.length_scale,
        std=PRIOR_STD_M / params.length_scale,
    ).affine_mapping(
        operator=ice_projection_operator(state, ice_space, exclude_ice_shelves=True)
    )
    noise = inf.GaussianMeasure.from_standard_deviation(
        inf.EuclideanSpace(len(points)), NOISE_STD_MM * 1.0e-3 / params.length_scale
    )
    problem = inf.LinearForwardProblem(forward, data_error_measure=noise)
    return (
        state,
        params,
        fingerprint,
        ice_space,
        prior,
        problem,
        inf.LinearBayesianInversion(problem, prior),
    )


state, params, fingerprint, ice_space, prior, problem, inversion = build(LMAX)
metre = params.length_scale
mm = 1000.0 * metre
print(f"model space {ice_space.dim} dimensions; data space {problem.data_space.dim}")

# %% [markdown]
# ## A synthetic truth and its data

# %%
np.random.seed(4)
truth, data = problem.synthetic_model_and_data(prior)
print(f"truth from {truth.data.min() * metre:.2f} to {truth.data.max() * metre:.2f} m")
print(
    f"data standard deviation {data.std() * mm:.2f} mm against {NOISE_STD_MM} mm of noise"
)

fig, ax = sl.create_map_figure(figsize=(11, 5))
sl.plot_points(
    points,
    data=data * mm,
    ax=ax,
    s=25,
    symmetric=True,
    colorbar=True,
    colorbar_kwargs={"label": "Observed sea level change (mm)"},
)

# %% [markdown]
# ## The posterior
#
# The normal equations are solved by conjugate gradients, preconditioned
# by the factorised normal operator of the coarser problem. Solving twice
# shows what the preconditioner buys.

# %%
_, _, _, _, _, _, surrogate = build(SURROGATE_LMAX)
start = time.time()
preconditioner = inf.CholeskySolver(galerkin=True)(surrogate.normal_operator)
print(f"surrogate factorised in {time.time() - start:.1f} s")

for label, operator in (("without", None), ("with", preconditioner)):
    solver = inf.CGSolver(rtol=1.0e-5)
    start = time.time()
    posterior = inversion.model_posterior_measure(data, solver, preconditioner=operator)
    posterior.expectation
    print(
        f"{label} preconditioner: {solver.iterations} iterations in "
        f"{time.time() - start:.1f} s"
    )

posterior_mean = posterior.expectation
print(
    f"chi-squared at the truth {problem.chi_squared(truth, data):.1f}, "
    f"at the posterior mean {problem.chi_squared(posterior_mean, data):.1f}, "
    f"data dimension {problem.data_space.dim}"
)

VIEWS = {
    "Antarctica": (ccrs.SouthPolarStereo(), [-180, 180, -90, -63]),
    "Greenland": (ccrs.Orthographic(-42, 72), [-75, -8, 58, 85]),
}
limit = max(abs(truth.data).max(), abs(posterior_mean.data).max()) * metre
for name, (projection, extent) in VIEWS.items():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5.5),
        subplot_kw={"projection": projection},
        layout="constrained",
    )
    for ax, (field, label) in zip(
        axes, [(truth, "Truth (m)"), (posterior_mean, "Posterior mean (m)")]
    ):
        sl.plot(
            field * metre,
            ax=ax,
            map_extent=extent,
            vmin=-limit,
            vmax=limit,
            colorbar_kwargs={"label": label},
        )
    fig.suptitle(name)

# %% [markdown]
# ## Sea level contributions
#
# Any linear functional of the model has a posterior of its own. Here the
# barystatic contribution of each ice sheet, and the total through the
# sea level equation itself, with the reduction in variance the data buy.

# %%
west, east, peninsula, greenland = state.ice_basin_groupings(scheme="macro_regions")
masks, _ = state.grouped_ice_projections(groupings=[west + peninsula, east, greenland])
SHEETS = ["West Antarctica", "East Antarctica", "Greenland"]
contribution = ice_space.l2_products_operator(masks) * (
    -params.ice_density / (params.water_density * state.ocean_area) * mm
)
response_space = fingerprint.codomain
total = (
    (
        ocean_average_operator(state, response_space.subspace(0))
        @ response_space.subspace_projection(0)
        @ fingerprint
    )
    * params.ice_density
    * mm
)


def summarise(operator, labels):
    post = posterior.affine_mapping(operator=operator).with_dense_covariance()
    pri = prior.affine_mapping(operator=operator).with_dense_covariance()
    true = operator(truth)
    post_cov = post.covariance.matrix(dense=True)
    pri_cov = pri.covariance.matrix(dense=True)
    for i, label in enumerate(labels):
        reduction = 100.0 * (1.0 - post_cov[i, i] / pri_cov[i, i])
        print(
            f"{label:<18}{true[i]:>9.2f}{post.expectation[i]:>12.2f}"
            f"{np.sqrt(post_cov[i, i]):>9.2f}{np.sqrt(pri_cov[i, i]):>13.2f}{reduction:>10.1f}%"
        )
    return post, pri


print(
    f"{'Contribution (mm)':<18}{'Truth':>9}{'Posterior':>12}{'Sigma':>9}{'Prior sigma':>13}{'Reduction':>11}"
)
sheet_posterior, sheet_prior = summarise(contribution, SHEETS)
summarise(total, ["All grounded ice"])

inf.plot_corner_distributions(
    sheet_posterior,
    prior_measure=sheet_prior,
    true_values=contribution(truth),
    labels=[f"{name} (mm)" for name in SHEETS],
    title="Barystatic sea level contributions",
)

plt.show()
