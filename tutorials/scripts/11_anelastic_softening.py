# %% [markdown]
# # 11. Anelastic softening and sea level fingerprints
#
# An elastic fingerprint uses Love numbers computed from the moduli of a
# seismological model, which for PREM are those at a period of one second.
# The mantle is anelastic, and its moduli fall with frequency, so a load
# that grows over months, years or decades meets a softer Earth than the
# seismic one. This script estimates the size of that effect on
# fingerprints, and on the mass that GRACE would infer from them, using
# the complex Love numbers of a model frozen at a frequency (tutorial 3).
#
# The estimate is rough by construction. For each period T the model is
# frozen at the angular frequency 2 pi / T, its complex Love numbers are
# computed, and their real parts are used as the Love numbers of an
# "elastic" fingerprint with fixed shorelines. This is the in-phase
# response to a load oscillating at that period; a real load is not
# oscillatory, and the out-of-phase part is dropped, its size being
# reported instead.
#
# The rheology is PREM's shear quality factor Q with a frequency
# dependence. At periods shorter than a corner period T_m the band is
# PREM's own, a constant Q with logarithmic dispersion about one second.
# At longer periods Q falls as a power of frequency, Q = Q_PREM
# (T_m / T)^alpha, with the dispersion that causality requires. Written
# for the shear compliance J = 1 / mu, with x = T / T_m >= 1,
#
#   J / J_0 = 1 + (2 / pi Q) ln(T_m / T_0)
#               + (1 / Q) cot(alpha pi / 2) (x^alpha - 1) - (i / Q) x^alpha,
#
# whose limit as alpha tends to zero is the logarithmic law continued to
# all periods. The corner is put at the period of the gravest seismic
# normal mode, 0S2, as is usual in studies of the anelasticity of the body
# tides, for which alpha between about 0.2 and 0.3 has been inferred. The
# bulk modulus follows the logarithmic law with PREM's Q_kappa, which is
# negligible.
#
# The run takes about a minute at degree 256.

# %%
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM
from planetmodel.character import DENSITY
from planetmodel.fields import ComposedField
from planetmodel.materials import kappa_mu, moduli, named
from planetmodel.rheology import frozen
from planetmodel.units import FREQUENCY
from planetmodel.vocabulary import Constant

import pyslfp as sl
from pyslfp.love_numbers.table import AXIAL_NAMES, NAMES

LMAX = 256
YEAR = 3.15576e7  # seconds
REFERENCE_PERIOD = 1.0  # seconds; PREM's
CORNER_PERIOD = 3233.0  # seconds; the period of 0S2
ALPHAS = (0.0, 0.15, 0.3)
PERIODS = {"1 month": YEAR / 12.0, "1 year": YEAR, "10 years": 10.0 * YEAR}
GRACE_DEGREES = (2, 60)

# %% [markdown]
# ## The softened model
#
# `planetmodel.frozen` freezes a model under the logarithmic law. The
# function below does the same for the law above, storing the complex
# moduli A, C, F, L, N of each layer, shifted through the Voigt averages
# as `planetmodel.rheology` does, so that the Love number solver reads it
# as it would any frozen model.


# %%
def modulus_factor(quality: np.ndarray, period: float, alpha: float, /) -> np.ndarray:
    """The complex factor by which a modulus of quality factor `quality`
    is multiplied at `period`; one where the quality factor is not
    positive, which is planetmodel's mark for no loss."""
    quality = np.asarray(quality, dtype=float)
    loss = np.zeros(quality.shape)
    np.divide(1.0, quality, out=loss, where=quality > 0.0)
    if period <= CORNER_PERIOD:
        dispersion = (2.0 / np.pi) * np.log(period / REFERENCE_PERIOD)
        absorption = 1.0
    else:
        x = period / CORNER_PERIOD
        dispersion = (2.0 / np.pi) * np.log(CORNER_PERIOD / REFERENCE_PERIOD)
        if alpha > 0.0:
            dispersion += (x**alpha - 1.0) / np.tan(0.5 * np.pi * alpha)
        else:
            dispersion += (2.0 / np.pi) * np.log(x)
        absorption = x**alpha
    return 1.0 / (1.0 + loss * (dispersion - 1j * absorption))


def softened(model, period: float, alpha: float, /):
    """The model frozen at `period` (in seconds) under the power-law
    absorption band of exponent `alpha` in shear, and the logarithmic band
    in bulk."""

    def shift(modulus, quality, exponent):
        return ComposedField(
            lambda m, q: np.asarray(m) * (modulus_factor(q, period, exponent) - 1.0),
            [modulus, quality],
            character=DENSITY,
        )

    layers = []
    for layer in model.layers:
        fields = dict(layer.fields)
        base = moduli(layer)
        kappa, mu = kappa_mu(layer)
        dmu = shift(mu, layer["qmu"], alpha) if "qmu" in layer else 0.0 * kappa
        dkappa = (
            shift(kappa, layer["qkappa"], 0.0) if "qkappa" in layer else 0.0 * kappa
        )
        shifted = {
            "A": base["A"] + dkappa + (4.0 / 3.0) * dmu,
            "C": base["C"] + dkappa + (4.0 / 3.0) * dmu,
            "F": base["F"] + dkappa - (2.0 / 3.0) * dmu,
            "L": base["L"] + dmu,
            "N": base["N"] + dmu,
        }
        fields.update({name: named(f, name) for name, f in shifted.items()})
        layers.append(fields)
    constants = dict(model.constants)
    omega = 2.0 * np.pi / period
    constants["omega"] = Constant(
        omega, FREQUENCY, meaning="angular frequency the model is frozen at"
    )
    return model.replaced(layers=layers, constants=constants)


def in_phase(love: sl.LoveNumbers, /) -> sl.LoveNumbers:
    """The real parts of a complex table, as a real table."""
    arrays = {name: getattr(love, name).real.copy() for name in NAMES}
    axial = {name: float(np.real(getattr(love, name))) for name in AXIAL_NAMES}
    return replace(love, omega=None, **arrays, **axial)


# %% [markdown]
# ## Love numbers
#
# The reference is PREM without its ocean, elastic, with its moduli at one
# second. Against it, each exponent and period gives a complex table. Two
# checks are printed: the law above at alpha = 0 against planetmodel's own
# logarithmic freezing, from which it differs at second order in 1 / Q by
# being written for the compliance, and the degree-2 tidal Love number,
# which is the quantity the body tides constrain.

# %%
prem = PREM(ocean=False)
elastic = sl.LoveNumbers.from_model(prem, LMAX)
elastic_c = elastic.conventional()


def potential_factor(love: sl.LoveNumbers, /) -> np.ndarray:
    """(1 + k'_l) of a table over that of the elastic one, by degree: the
    factor on a mass coefficient recovered from the potential with the
    elastic numbers. One at degrees 0 and 1, which GRACE does not give,
    and where 1 + k'_1 vanishes in the centre-of-mass frame."""
    factor = np.ones(LMAX + 1)
    factor[2:] = (1.0 + love.conventional()["k"].real[2:]) / (1.0 + elastic_c["k"][2:])
    return factor


tables: dict[tuple[float, str], sl.LoveNumbers] = {}
for alpha in ALPHAS:
    for label, period in PERIODS.items():
        tables[alpha, label] = sl.LoveNumbers.from_model(
            softened(prem, period, alpha), LMAX
        )

decade = PERIODS["10 years"]
own = sl.LoveNumbers.from_model(frozen(prem, 2.0 * np.pi / decade), LMAX)
ratio = tables[0.0, "10 years"].conventional()["h"][1:] / own.conventional()["h"][1:]
print(
    "alpha = 0 at 10 years against planetmodel.frozen, largest relative "
    f"difference in h': {np.max(np.abs(ratio - 1.0)):.2e}"
)

lines = ["", "Love numbers, in-phase change from elastic PREM in per cent"]
lines.append(
    f"{'alpha':>5} {'period':>9} {'k_2^T':>7} "
    + " ".join(f"{'h_' + str(l):>7}" for l in (2, 10, 50, 200))
    + " "
    + " ".join(f"{'1+k_' + str(l):>7}" for l in (2, 10, 50, 200))
    + f" {'max |Im/Re| h':>14}"
)
lines.append(f"{'':>5} {'elastic':>9} {elastic.tidal()['k'][2]:7.4f}")
for (alpha, label), table in tables.items():
    c = table.conventional()
    dh = 100.0 * (c["h"].real[1:] / elastic_c["h"][1:] - 1.0)
    dk = 100.0 * (potential_factor(table)[1:] - 1.0)
    lag = np.max(np.abs(c["h"].imag[1:] / c["h"].real[1:]))
    lines.append(
        f"{alpha:5.2f} {label:>9} {table.tidal()['k'][2].real:7.4f} "
        + " ".join(f"{dh[l - 1]:7.2f}" for l in (2, 10, 50, 200))
        + " "
        + " ".join(f"{dk[l - 1]:7.2f}" for l in (2, 10, 50, 200))
        + f" {lag:14.3f}"
    )
print("\n".join(lines))

# %% [markdown]
# ## Fingerprints
#
# Each table gives an Earth model, and with the present-day ICE-7G state a
# fingerprint solver. Three loads are used: ten per cent of the Greenland
# ice sheet, ten per cent of the West Antarctic ice sheet, and a disc over
# the Amazon basin standing for a change of land water. Each response is
# scaled by the barystatic sea level change of its load, so that the
# fields are in metres per metre of global mean sea level.
#
# The GRACE diagnostic takes the softened Earth as the truth. GRACE
# measures the potential, and the surface mass is recovered from its
# coefficient of degree l on dividing by 1 + k'_l. Using the elastic
# number where the softened one applies multiplies the recovered mass
# coefficient by (1 + k'_l softened) / (1 + k'_l elastic). That factor is
# applied to the full load, direct and ocean, within GRACE's degrees, and
# the result integrated over the support of the direct load. The potential
# of the rotational feedback at degree 2 is left out of this.

# %%
reference_model = sl.EarthModel(LMAX, love_numbers=elastic)
reference_state = sl.EarthState.from_defaults(lmax=LMAX)


def solver_for(love: sl.LoveNumbers, /) -> sl.LinearSeaLevelEquation:
    """The fingerprint solver on the present-day state with these Love
    numbers."""
    model = sl.EarthModel(LMAX, love_numbers=love)
    state = sl.EarthState(
        reference_state.ice_thickness,
        reference_state.sea_level,
        model,
        land=reference_state.land,
    )
    return sl.LinearSeaLevelEquation(state)


elastic_solver = solver_for(elastic)
state = elastic_solver.state
parameters = state.model.parameters

loads = {
    "Greenland": state.greenland_load(fraction=0.1),
    "West Antarctica": state.west_antarctic_load(fraction=0.1),
    "Amazon disc": state.disk_load(8.0, -5.0, -62.0, 1.0 / parameters.length_scale),
}


def barystatic(load) -> float:
    """The global mean sea level change of a direct load."""
    return -state.model.integrate(load) / (parameters.water_density * state.ocean_area)


def band_limited(field, factor=None):
    """The field within GRACE's degrees, its coefficients multiplied by
    `factor` by degree where given."""
    coefficients = state.model.expand_field(field)
    weights = np.zeros(LMAX + 1)
    weights[GRACE_DEGREES[0] : GRACE_DEGREES[1] + 1] = 1.0
    if factor is not None:
        weights = weights * factor
    coefficients.coeffs *= weights[None, :, None]
    return state.model.expand_coefficient(coefficients)


def rms(field, mask) -> float:
    """The root mean square of a field over a mask."""
    return np.sqrt(
        state.model.integrate(field * field * mask) / state.model.integrate(mask)
    )


ocean = state.ocean_function
everywhere = state.model.constant_grid(1.0)

results: dict[tuple, tuple] = {}
for name, load in loads.items():
    scale = 1.0 / barystatic(load)
    sea_level, displacement, _, _ = elastic_solver.solve_sea_level_equation(load)
    results[name, "elastic"] = (sea_level * scale, displacement * scale)

lines = [
    "",
    "Fingerprints, change from the elastic one",
    "  sea level: rms over the oceans, in per cent of the barystatic change",
    "             and of the rms of the elastic fingerprint",
    "  displacement: global rms, in per cent of the elastic rms",
    "  GRACE: bias in the mass recovered over the load with elastic numbers",
    "",
    f"{'load':>16} {'alpha':>5} {'period':>9} {'sea level':>10} {'':>8} "
    f"{'displacement':>12} {'GRACE mass':>10}",
    f"{'':>16} {'':>5} {'':>9} {'% of GMSL':>10} {'% of rms':>8} "
    f"{'% of rms':>12} {'%':>10}",
]
for (alpha, label), table in tables.items():
    solver = solver_for(in_phase(table))
    factor = potential_factor(table)
    for name, load in loads.items():
        scale = 1.0 / barystatic(load)
        sea_level, displacement, _, _ = solver.solve_sea_level_equation(load)
        sea_level, displacement = sea_level * scale, displacement * scale
        results[name, alpha, label] = (sea_level, displacement)
        elastic_sea_level, elastic_displacement = results[name, "elastic"]

        support = state.model.zero_grid()
        support.data[:] = (load.data != 0.0).astype(float)
        full_load = load + state.direct_load_from_sea_level_change(sea_level / scale)
        true_mass = state.model.integrate(band_limited(full_load) * support)
        seen_mass = state.model.integrate(band_limited(full_load, factor) * support)

        sea_level_change = rms(sea_level - elastic_sea_level, ocean)
        displacement_change = rms(displacement - elastic_displacement, everywhere)
        lines.append(
            f"{name:>16} {alpha:5.2f} {label:>9} "
            f"{100.0 * sea_level_change:10.3f} "
            f"{100.0 * sea_level_change / rms(elastic_sea_level, ocean):8.2f} "
            f"{100.0 * displacement_change / rms(elastic_displacement, everywhere):12.2f} "
            f"{100.0 * (seen_mass / true_mass - 1.0):10.3f}"
        )
print("\n".join(lines))

# %% [markdown]
# ## Reading the numbers
#
# Under PREM's own band, alpha = 0, the mass recovered from GRACE with
# elastic Love numbers is low by a little under half of one per cent, and
# the figure hardly depends on the load, because 1 + k' changes by a
# nearly constant factor across GRACE's degrees. Most of that is present
# at one month: the dispersion is logarithmic in period, and one second
# to one month is three times as many decades as one month to ten years.
# The reference period of the seismological model therefore matters more
# than the time scale of the load.
#
# The displacement is more sensitive than the potential or the sea level.
# The change of h' is largest near degree 50, whose solutions sample the
# low-velocity zone, where Q is least, so the softening is felt in the
# uplift beneath and beside the load, and little in the far field.
#
# With alpha = 0.3 the changes at ten years are several times larger, but
# the out-of-phase part of h' is then above a tenth of the in-phase part,
# and the in-phase fingerprint is no longer a fair summary: a load history
# in such a body calls for the time-domain solver of `pyslfp.viscoelastic`
# (tutorial 6).

# %% [markdown]
# ## Figures

# %%
degrees = elastic.degree[1:]
styles = {"1 month": ":", "1 year": "--", "10 years": "-"}
colours = dict(zip(ALPHAS, ("#1b6ca8", "#d1495b", "#3a7d44")))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
for (alpha, label), table in tables.items():
    c = table.conventional()
    kwargs = dict(color=colours[alpha], ls=styles[label])
    axes[0].semilogx(
        degrees, 100.0 * (c["h"].real[1:] / elastic_c["h"][1:] - 1.0), **kwargs
    )
    axes[1].semilogx(
        degrees[1:],
        100.0 * (potential_factor(table)[2:] - 1.0),
        **kwargs,
    )
    axes[2].semilogx(
        degrees,
        100.0 * np.abs(c["h"].imag[1:] / c["h"].real[1:]),
        label=rf"$\alpha$ = {alpha:g}, {label}",
        **kwargs,
    )
axes[0].set_title("$h'_l$: in-phase change from elastic (%)")
axes[1].set_title("$1 + k'_l$: in-phase change from elastic (%)")
axes[2].set_title("$h'_l$: out-of-phase part, |Im / Re| (%)")
axes[1].axvspan(*GRACE_DEGREES, color="0.9", zorder=0)
for ax in axes:
    ax.set_xlabel("degree")
    ax.grid(alpha=0.3)
axes[2].legend(fontsize=8, ncol=1)

columns = [("elastic",), (0.0, "10 years"), (ALPHAS[-1], "10 years")]
titles = [
    "elastic",
    r"$\alpha$ = 0, 10 years, minus elastic",
    rf"$\alpha$ = {ALPHAS[-1]:g}, 10 years, minus elastic",
]
for name in loads:
    fig, axes = sl.subplots(2, 3, figsize=(17, 7), layout="constrained")
    for j, (key, title) in enumerate(zip(columns, titles)):
        sea_level, displacement = results[(name, *key)]
        if j > 0:
            sea_level = sea_level - results[name, "elastic"][0]
            displacement = displacement - results[name, "elastic"][1]
        # Far-field limits: the near field is an order of magnitude larger.
        limit = 1.5 if j == 0 else 0.05 if j == 1 else 0.2
        sl.plot(
            sea_level * state.ocean_projection(),
            ax=axes[0, j],
            vmin=-limit,
            vmax=limit,
            colorbar_kwargs={"label": "sea level (m per m GMSL)"},
        )
        sl.plot(
            displacement,
            ax=axes[1, j],
            vmin=-limit,
            vmax=limit,
            colorbar_kwargs={"label": "vertical displacement (m per m GMSL)"},
        )
        axes[0, j].set_title(title)
    fig.suptitle(name)

plt.show()
