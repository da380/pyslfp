# %% [markdown]
# # 12. Ice altimetry and GRACE over the ice sheets
#
# Over an ice sheet, gravimetry sees mass and altimetry sees height, and
# the two differ by the firn. The surface of the grounded ice rises by
#
# $$\delta s = h + a + u,$$
#
# with $h$ the ice-equivalent thickness change (the change in mass per unit
# area over the density of ice), $a$ the change in firn air content, which
# has no mass, and $u$ the vertical displacement of the solid surface in
# the response to the load. This is the pattern of the ocean, where
# altimetry sees the sea surface, GRACE the mass, and a density field
# carries the height that has no mass.
#
# This script sets up a small synthetic experiment: a true $(h, a)$ made
# of ice dynamics, snowfall and firn compaction, synthetic GRACE
# coefficients and ice altimetry, and two Bayesian inversions, of GRACE
# alone and of the two together. Every amplitude below is a placeholder,
# chosen to make a readable figure and not to represent any data set.

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
from cartopy import crs as ccrs

import pyslfp as sl
from pyslfp.core import sobolev_load_space
from pyslfp.elastic import FingerPrintOperator
from pyslfp.models import ice_and_firn_air
from pyslfp.observations import (
    GraceObservationModel,
    IceAltimetryObservationModel,
    grounded_ice_altimetry_points,
)

LMAX = 64
SPACE_SCALE_KM = 200.0  # the length scale of the Sobolev spaces
LOAD_ORDER, FIELD_ORDER = 0.25, 2.0  # loads carry indicators; fields are smooth

# The data. Placeholders.
ALTIMETRY_SPACING_DEG = 1.5  # about the grid's own spacing at this degree
ALTIMETRY_NOISE_M = 0.05  # per point
GRACE_MAX_DEGREE = 30
GRACE_NOISE_STOKES = 1.0e-12  # per Stokes coefficient, the same at every degree

# The prior, as three independent processes: (pointwise std in metres,
# correlation length in km). Placeholders.
PRIOR_ORDER = 2.0
DYNAMICS = (0.4, 150.0)  # ice-equivalent thickness, no firn change
SNOWFALL = (0.15, 400.0)  # ice-equivalent thickness of the snowfall anomaly
COMPACTION = (0.2, 250.0)  # firn air content, no mass
SNOW_DENSITY, ICE_DENSITY = 400.0, 917.0  # kg / m^3, for the air a snowfall holds

POSTERIOR_DRAWS = 200  # for the pointwise standard deviations, good to 5 %

start = time.time()
state = sl.EarthState.from_defaults(lmax=LMAX)
params = state.model.parameters
metre = 1.0 / params.length_scale
km = 1000.0 * metre

# %% [markdown]
# ## The model space and the two observations
#
# The load space has order 1/4, since a load carries the indicator of the
# grounded ice, and the response one order more, where fields have point
# values. The two fields live in a smoother space, on which the prior is
# put. `ice_and_firn_air` joins the ice sheets, which load the Earth, to
# their firn air content, which does not; its `ice_surface_height` is the
# observable of ice altimetry, and the potential change of its response
# that of GRACE.
#
# Ice altimetry is modelled on grounded ice alone: a floating shelf stands
# on the sea surface. `grounded_ice_altimetry_points` returns evenly spaced
# points on the grounded ice of the state, and the observation model
# refuses any other.

# %%
space_scale = SPACE_SCALE_KM * km
fingerprint = FingerPrintOperator(
    state,
    load_parameters=(LOAD_ORDER, space_scale),
    response_parameters=(LOAD_ORDER + 1.0, space_scale),
)
field_space = sobolev_load_space(state.model, FIELD_ORDER, space_scale)
setup = ice_and_firn_air(
    state,
    fingerprint.domain,
    space=field_space,
    observable_space=fingerprint.field_space,
)
ICE, FIRN = setup.names

points = grounded_ice_altimetry_points(state, spacing=ALTIMETRY_SPACING_DEG)
altimetry = IceAltimetryObservationModel(
    setup.ice_surface_height(fingerprint),
    state,
    points,
    noise=ALTIMETRY_NOISE_M * metre,
)
grace = GraceObservationModel(
    fingerprint.codomain.projection("potential_change") @ setup.response(fingerprint),
    state.model,
    GRACE_MAX_DEGREE,
    convention="stokes",
    noise=GRACE_NOISE_STOKES,
)
print(
    f"model space {setup.space.dim} dimensions; {len(points)} altimetry points "
    f"and {grace.data_space.dim} GRACE coefficients"
)

# %% [markdown]
# ## A correlated prior
#
# Three processes move the two fields. Ice dynamics changes $h$ alone and
# firn compaction $a$ alone, while a snowfall anomaly of ice-equivalent
# thickness $q$ adds $q$ to $h$ and $(\rho_i / \rho_s - 1)\,q$ to $a$, the
# air that snow of density $\rho_s$ holds. Independent priors on the three
# therefore give a prior on $(h, a)$ that is correlated, degree by degree,
# and `correlated_measure` builds it from the covariance matrix of the
# pair at each harmonic.

# %%
air_per_ice = ICE_DENSITY / SNOW_DENSITY - 1.0


def spectrum(std_m, scale_km):
    """The spectral variances of a Sobolev measure on the fields' space."""
    measure = field_space.sobolev_measure(
        PRIOR_ORDER, scale_km * km, pointwise_std=std_m * metre
    )
    return measure.covariance.eigenvalues


dynamics, snowfall, compaction = (
    spectrum(*process) for process in (DYNAMICS, SNOWFALL, COMPACTION)
)
covariances = np.empty((field_space.dim, 2, 2))
covariances[:, 0, 0] = dynamics + snowfall
covariances[:, 0, 1] = covariances[:, 1, 0] = air_per_ice * snowfall
covariances[:, 1, 1] = air_per_ice**2 * snowfall + compaction
prior = field_space.correlated_measure(covariances, labels=setup.names)
prior_std = {
    ICE: np.hypot(DYNAMICS[0], SNOWFALL[0]),
    FIRN: np.hypot(air_per_ice * SNOWFALL[0], COMPACTION[0]),
}
print(
    f"prior pointwise std: h {prior_std[ICE]:.2f} m, a {prior_std[FIRN]:.2f} m, "
    f"correlation {air_per_ice * SNOWFALL[0] ** 2 / (prior_std[ICE] * prior_std[FIRN]):.2f}"
)

# %% [markdown]
# ## A truth with distinct patterns, and its data
#
# The truth is built from the same three processes, each in its own place:
# dynamic thinning in the Amundsen Sea sector and in West Greenland, a
# snowfall anomaly over Dronning Maud Land, and firn compaction in the
# interiors of East Antarctica and of Greenland. So there is mass change
# without firn change, firn change without mass change, and one place
# where the two go together.


# %%
def bump(latitude, longitude, amplitude_m, width_km):
    """A Gaussian bump on the sphere, as a field of the fields' space."""
    lats, lons = np.meshgrid(
        np.radians(state.lats()), np.radians(state.lons()), indexing="ij"
    )
    lat0, lon0 = np.radians(latitude), np.radians(longitude)
    cosine = np.sin(lats) * np.sin(lat0) + np.cos(lats) * np.cos(lat0) * np.cos(
        lons - lon0
    )
    distance = params.mean_sea_floor_radius * np.arccos(np.clip(cosine, -1.0, 1.0))
    grid = state.model.zero_grid()
    grid.data[:] = (
        amplitude_m * metre * np.exp(-0.5 * (distance / (width_km * km)) ** 2)
    )
    return field_space.from_components(field_space.to_components(grid))


true_dynamics = bump(-76.0, -105.0, -1.0, 300.0) + bump(69.5, -48.0, -1.0, 200.0)
true_snowfall = bump(-75.0, 25.0, 0.25, 600.0)
true_compaction = bump(-75.0, 120.0, -0.3, 500.0) + bump(75.0, -40.0, -0.4, 250.0)
truth = (
    true_dynamics + true_snowfall,
    air_per_ice * true_snowfall + true_compaction,
)

rng = np.random.default_rng(12)
grace_data = grace.forward_operator(truth) + grace.noise.sample(rng=rng)
altimetry_data = altimetry.forward_operator(truth) + altimetry.noise.sample(rng=rng)

displacement = (
    fingerprint.field_space.point_evaluation_operator(points)
    @ fingerprint.codomain.field_operator("displacement")
    @ setup.response(fingerprint)
)(truth)
print(
    f"surface height change at the points from {altimetry_data.min() / metre:.2f} "
    f"to {altimetry_data.max() / metre:.2f} m, of which the solid surface moves "
    f"from {displacement.min() / metre * 100:.1f} to "
    f"{displacement.max() / metre * 100:.1f} cm"
)

# %% [markdown]
# ## The two inversions
#
# The data are few and the fingerprint is fast at this degree, so each
# forward operator is assembled as a matrix, a row for each datum by one
# adjoint solve, and the normal equations are factorised. The joint
# problem stacks the two matrices. At a higher degree one would keep the
# operators matrix-free and precondition, as in the fifth tutorial.

# %%
operators = {
    "GRACE": grace.forward_operator.assembled(),
    "altimetry": altimetry.forward_operator.assembled(),
}
print(f"forward operators assembled after {time.time() - start:.0f} s")

joint_space = inf.EuclideanSpace(grace.data_space.dim + altimetry.data_space.dim)
joint_operator = inf.LinearOperator.from_matrix(
    setup.space,
    joint_space,
    np.vstack([op.matrix(form="components") for op in operators.values()]),
    form="components",
)
joint_noise = inf.GaussianMeasure.from_standard_deviations(
    joint_space,
    np.concatenate(
        [
            np.full(grace.data_space.dim, GRACE_NOISE_STOKES),
            np.full(altimetry.data_space.dim, ALTIMETRY_NOISE_M * metre),
        ]
    ),
)
cases = {
    "GRACE alone": (operators["GRACE"], grace.noise, grace_data),
    "GRACE and altimetry": (
        joint_operator,
        joint_noise,
        np.concatenate([grace_data, altimetry_data]),
    ),
}

grounded = state.ice_sheet_projection(exclude_ice_shelves=True)
on_ice = np.isfinite(grounded.data)
area = np.cos(np.radians(state.lats()))[:, None] * np.ones_like(grounded.data)


def rms(values):
    """The root mean square over the grounded ice, weighted by area, in
    metres."""
    return np.sqrt(np.average(values[on_ice] ** 2, weights=area[on_ice])) / metre


posterior_mean, posterior_std = {}, {}
for case, (operator, noise, data) in cases.items():
    problem = inf.LinearForwardProblem(operator, error=noise)
    inversion = inf.LinearGaussianInversion(problem, prior, solver=inf.CholeskySolver())
    posterior = inversion(data)
    posterior_mean[case] = posterior.expectation
    print(
        f"{case}: chi-squared at the posterior mean "
        f"{problem.chi_squared(posterior.expectation, data):.0f} for "
        f"{problem.data_space.dim} data"
    )
    # The pointwise standard deviations, estimated from draws.
    draws = posterior.samples(POSTERIOR_DRAWS, rng=rng)
    posterior_std[case] = tuple(
        np.sqrt(
            np.mean(
                [(draw[i].data - posterior.expectation[i].data) ** 2 for draw in draws],
                axis=0,
            )
        )
        for i in range(2)
    )
print(f"inversions done after {time.time() - start:.0f} s")

# %% [markdown]
# ## What each data set resolves
#
# GRACE alone recovers the mass change at its own resolution and can say
# of the firn only what the prior's correlation implies: it puts a firn
# change under every mass change, and none where the firn changed alone.
# With altimetry the two fields separate. The errors over the grounded ice
# and the posterior standard deviations say by how much.

# %%
print(f"\n{'Over the grounded ice (m)':<28}{'h':>8}{'a':>8}")
print(
    f"{'prior std':<28}{prior_std[ICE]:>8.3f}{prior_std[FIRN]:>8.3f}\n"
    f"{'rms of the truth':<28}" + "".join(f"{rms(field.data):>8.3f}" for field in truth)
)
for case in cases:
    errors = [rms(m.data - t.data) for m, t in zip(posterior_mean[case], truth)]
    stds = [rms(std) for std in posterior_std[case]]
    print(
        f"{case + ': rms error':<28}{errors[0]:>8.3f}{errors[1]:>8.3f}\n"
        f"{case + ': rms std':<28}{stds[0]:>8.3f}{stds[1]:>8.3f}"
    )

VIEWS = {
    "Antarctica": (ccrs.SouthPolarStereo(), [-180, 180, -90, -63], (17, 7.5)),
    "Greenland": (ccrs.Orthographic(-42, 72), [-75, -8, 58, 85], (17, 6.5)),
}
ROWS = {
    ICE: "Ice-equivalent thickness change $h$",
    FIRN: "Firn air content change $a$",
}
output = Path("12_ice_altimetry_figures")
output.mkdir(exist_ok=True)


def masked(values):
    """A field in metres, shown on the grounded ice alone."""
    grid = state.model.zero_grid()
    grid.data[:] = values / metre
    return grid * grounded


for view, (projection, extent, figsize) in VIEWS.items():
    fig, axes = sl.subplots(2, 5, projection, figsize=figsize)
    for i, name in enumerate(ROWS):
        means = [truth[i]] + [posterior_mean[case][i] for case in cases]
        titles = ["Truth"] + [f"{case}: mean" for case in cases]
        limit = max(np.abs(mean.data[on_ice]).max() for mean in means) / metre
        for ax, mean, title in zip(axes[i, :3], means, titles):
            _, signed = sl.plot(
                masked(mean.data),
                ax=ax,
                map_extent=extent,
                gridlines=False,
                vmin=-limit,
                vmax=limit,
                colorbar=False,
            )
            ax.set_title(title)
        for ax, case in zip(axes[i, 3:], cases):
            std = posterior_std[case][i]
            _, spread = sl.plot(
                masked(std),
                ax=ax,
                map_extent=extent,
                gridlines=False,
                cmap="Purples",
                vmin=0.0,
                vmax=prior_std[name],
                colorbar=False,
            )
            ax.set_title(f"{case}: std\n(rms {rms(std):.2f} m)")
        fig.colorbar(signed, ax=axes[i, :3], shrink=0.8, label=f"{ROWS[name]} (m)")
        fig.colorbar(
            spread,
            ax=axes[i, 3:],
            shrink=0.8,
            label=f"Posterior std (m); prior {prior_std[name]:.2f} m",
        )
    fig.suptitle(f"{view}: ice altimetry and GRACE, a synthetic experiment")
    path = output / f"ice_altimetry_{view.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path.resolve()}")

print(f"finished after {time.time() - start:.0f} s")
plt.show()
