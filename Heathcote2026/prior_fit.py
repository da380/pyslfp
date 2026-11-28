"""
The scales of the priors, fitted through the masks (Heathcote et al., 2026).

Each field of the model is a smooth invariant Gaussian field, with
coefficient variances v_l = a (1 + s^2 l (l + 1) / r^2)^(-q), and the load
operator confines it to its support. What a source (a model's or an
analysis's year-on-year changes) shows of a field is therefore the field
times the indicator of its support, and it is to the degree variances of
that product that the scale s is fitted here, with the exponent q fixed:

- ice thickness: the grounded ice of the two ice sheets, with the
  modulation towards the edge (`common.ice_modulation`'s profile 1 + a p);
- land water: the land outside the two ice sheets;
- manometric sea level: the ocean, less the field's mean over the ocean;
- steric height: the ocean.

The expected degree variances of the masked field are the average over
draws of the invariant field made with the same random numbers at every
trial scale, so that the misfit is a smooth function of the scale. The fit
is by least squares on the logarithms of the degree variances, with the
amplitude free, as `common.fit_spectrum` does for an unmasked field; the
amplitude of the prior is then the root mean square of the source over the
support, and the fitted spectrum is shown with that amplitude.

For the ice the nearness function p is smoothed at the prior's own scale,
so the growth a is refitted to the source's root mean square at each point
whenever the scale changes, and the two are iterated.

Sources: the year-on-year changes of the ESA Earth System Model, 1995 to
2006 (`esm_loads`), for the ice, the land water and the ocean's bottom
pressure (and, for the case of the ocean removed from gravity data, its
error series AOe07); the EN4 analyses over the same years (`en4_steric`)
for the steric height.

Outputs, in output_plots_prior_fit/:

- prior_fit.txt: for each field the fitted scales (fits over degrees 2 to
  60 and 2 to 180), the misfits over degrees 2 to 60 and 61 to 180, and
  the amplitudes;
- prior_fit.png: the source's root mean square coefficient by degree with
  the fitted masked spectra;
- prior_fit.npz: the spectra plotted.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from pyslfp.core.spaces import underlying_space

import common
import en4_steric
import esm_loads

OUTPUT_DIR = "output_plots_prior_fit"
ICE, LAND, MANOMETRIC, STERIC, RESIDUAL = (
    "ice thickness",
    "land water",
    "manometric sea level",
    "steric height",
    "ocean residual (AOe07)",
)
# The exponent of each field, and the scale (km) and exponent in use before
# the refit, for comparison.
EXPONENTS = {ICE: 2.5, LAND: 2.5, MANOMETRIC: 2.5, STERIC: 3.0, RESIDUAL: 2.5}
BEFORE = {
    ICE: (180.0, 1.5),
    LAND: (340.0, 1.5),
    MANOMETRIC: (330.0, 2.5),
    STERIC: (280.0, 3.0),
    RESIDUAL: (200.0, 2.5),
}
BANDS = ((2, 60), (61, 180))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior scales through the masks.")
    parser.add_argument("--lmax", type=int, default=255)
    parser.add_argument("--source-degree", type=int, default=180)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--years", type=int, nargs=2, default=[1995, 2006])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=[ICE, LAND, MANOMETRIC, STERIC, RESIDUAL],
        choices=[ICE, LAND, MANOMETRIC, STERIC, RESIDUAL],
    )
    parser.add_argument(
        "--free-exponent",
        action="store_true",
        help="Also fit the exponent with the scale, through the mask, over "
        "degrees 2 to the source's: a test of the fixed exponents.",
    )
    parser.add_argument("--ice-margin-growth", type=float, default=10.0)
    parser.add_argument(
        "--taper-degree",
        type=float,
        default=180.0,
        help="The priors of the ice, the land water, the manometric sea level "
        "and the residual follow (1 + s^2 k)^-q, with q and s both fitted over "
        "the source's degrees, and beyond this degree decay faster "
        "(`common.tapered_spectrum`). Zero: no taper, q fixed at EXPONENTS.",
    )
    parser.add_argument("--taper-order", type=float, default=2.5)
    parser.add_argument(
        "--plot-before",
        action="store_true",
        help="Also plot the spectra in use before the fit (BEFORE), a working "
        "comparison.",
    )
    parser.add_argument(
        "--ice-margin-scale-km",
        type=float,
        default=180.0,
        help="The scale of the nearness function of the ice modulation; zero "
        "ties it to the trial scale of the prior.",
    )
    parser.add_argument("--scale-bounds-km", type=float, nargs=2, default=[50.0, 6000.0])
    return parser.parse_args()


class Fitter:
    """The draws, the transforms and the masks shared by the fits."""

    def __init__(self, args):
        self.args = args
        physics = common.budget_physics(args.lmax)
        self.state, self.model = physics.state, physics.state.model
        self.space = physics.load_space
        self.flat = underlying_space(self.space)
        self.radius = self.flat.radius
        self.scale_mm = physics.scale_mm
        p = self.model.parameters
        self.km = 1000.0 / p.length_scale
        lmax = args.lmax
        self.degree_of = np.repeat(np.arange(lmax + 1), 2 * np.arange(lmax + 1) + 1)
        self.k = self.degree_of * (self.degree_of + 1.0) / self.radius**2
        self.edges = np.arange(lmax + 1) ** 2
        rng = np.random.default_rng(args.seed)
        self.xi = rng.standard_normal((args.draws, self.flat.dim))
        self.sea = self.state.ocean_function
        self.one = self.model.constant_grid(1.0)
        self.supports = physics.supports
        self.physics = physics
        self._nearness = {}
        # The nearness function at the trial scale, for the curve "before".
        self.tied_margin = False
        self.taper = (
            (args.taper_degree, args.taper_order) if args.taper_degree > 0.0 else None
        )

    # ---- spectra -------------------------------------------------------
    def variances(self, fields) -> np.ndarray:
        """Mean square coefficient by degree, as `common.degree_variances`,
        to the source's degree."""
        squares = np.mean(
            [(self.flat.to_components(f) / self.radius) ** 2 for f in fields], axis=0
        )
        sums = np.add.reduceat(squares, self.edges)
        top = self.args.source_degree + 1
        return (sums / (2.0 * np.arange(self.args.lmax + 1) + 1.0))[:top]

    def spectrum(self, scale_km: float, exponent: float, taper=None) -> np.ndarray:
        """Coefficient variances by component of the invariant field with
        unit pointwise variance; `taper` as `common.tapered_spectrum`."""
        v = common.tapered_spectrum(
            self.k, scale_km * self.km, exponent, self.radius, taper
        )
        return v * (4.0 * np.pi / v.sum())

    def draws(self, scale_km: float, exponent: float, taper=None):
        amplitude = self.radius * np.sqrt(self.spectrum(scale_km, exponent, taper))
        return [self.flat.from_components(amplitude * x) for x in self.xi]

    def rms(self, mask, fields) -> float:
        area = self.model.integrate(mask)
        return float(
            np.sqrt(np.mean([self.model.integrate(mask * f * f) for f in fields]) / area)
        )

    def less_ocean_mean(self, field):
        mean = self.model.integrate(self.sea * field) / self.model.integrate(self.sea)
        return self.sea * (field - mean * self.one)

    # ---- sources -------------------------------------------------------
    def sources(self) -> dict:
        args, model = self.args, self.model
        years = list(range(args.years[0], args.years[1] + 1))
        loads = {
            name: esm_loads.load_changes(model, name, years=years)[1]
            for name in ("H", "I", "O", "AOe07")
        }
        out = {ICE: [], LAND: [], MANOMETRIC: [], RESIDUAL: []}
        for water, ice, ocean, error in zip(*(loads[n] for n in ("H", "I", "O", "AOe07"))):
            parts = common.field_loads(self.physics, water, ice, ocean)
            out[ICE].append(parts[common.ICE] * self.scale_mm)
            out[LAND].append(parts[common.LAND] * self.scale_mm)
            out[MANOMETRIC].append(parts[common.OCEAN] * self.scale_mm)
            out[RESIDUAL].append(self.less_ocean_mean(error) * self.scale_mm)
        template = model.zero_grid()
        lats, lons = np.meshgrid(template.lats(), template.lons(), indexing="ij")
        steric = []
        for change in en4_steric.changes_at(lats, lons, years)[1]:
            field = template.copy()
            field.data[:] = 1000.0 * change
            steric.append(self.sea * field)
        out[STERIC] = steric
        return out

    # ---- the ice modulation -------------------------------------------
    def ice_sheets(self) -> dict:
        return {
            sheet: self.state.ice_sheet_projection(
                value=0.0, sheets=sheet, exclude_ice_shelves=True
            )
            for sheet in ("greenland", "antarctic")
        }

    def nearness(self, scale_km: float) -> np.ndarray:
        if self.args.ice_margin_scale_km > 0.0 and not self.tied_margin:
            scale_km = self.args.ice_margin_scale_km
        key = round(float(scale_km), 3)
        if key not in self._nearness:
            smoother = self.flat.heat_measure(scale_km * self.km).covariance
            grounded = self.state.ice_sheet_projection(
                value=0.0, exclude_ice_shelves=True
            )
            self._nearness = {
                key: np.clip(smoother(self.one - grounded).data, 0.0, 1.0)
            }
        return self._nearness[key]

    def ice_weight(self, scale_km: float, growth: float, stds: dict):
        """The modulation times the indicator of the grounded ice: the std
        of the masked ice field at each point, mm."""
        profile = 1.0 + growth * self.nearness(scale_km)
        weight = self.model.zero_grid()
        for sheet, grounded in self.ice_sheets().items():
            inside = grounded.data > 0.5
            shape = grounded.copy()
            shape.data[:] = np.where(inside, profile, 0.0)
            rms = np.sqrt(
                self.model.integrate(shape * shape) / self.model.integrate(grounded)
            )
            weight.data[inside] = stds[sheet] / rms * profile[inside]
        return weight

    def fit_growth(self, scale_km: float, pointwise, stds: dict) -> float:
        """The growth a for which the profile is nearest, in the least
        squares sense over the grounded ice of both sheets, to the source's
        root mean square at each point."""

        def misfit(growth):
            difference = self.ice_weight(scale_km, growth, stds) - pointwise
            return self.model.integrate(difference * difference)

        return float(minimize_scalar(misfit, bounds=(0.0, 200.0), method="bounded").x)

    # ---- the masked model ---------------------------------------------
    def masked(self, name, scale_km, exponent, weight=None, taper=None):
        """Draws of a field as the source would show it, with unit pointwise
        std of the invariant field (the ice: std as the weight has it)."""
        fields = self.draws(scale_km, exponent, taper)
        if name == ICE:
            return [weight * f for f in fields]
        if name == LAND:
            return [self.supports[common.LAND] * f for f in fields]
        if name in (MANOMETRIC, RESIDUAL):
            return [self.less_ocean_mean(f) for f in fields]
        return [self.sea * f for f in fields]

    def fit_scale(self, name, exponent, target, band, weight_of=None) -> float:
        low, high = band
        log_target = np.log(target[low : high + 1])

        def misfit(log_scale):
            scale_km = float(np.exp(log_scale))
            weight = None if weight_of is None else weight_of(scale_km)
            v = self.variances(self.masked(name, scale_km, exponent, weight))
            return float(np.var(np.log(v[low : high + 1]) - log_target))

        bounds = np.log(self.args.scale_bounds_km)
        fit = minimize_scalar(
            misfit, bounds=tuple(bounds), method="bounded", options={"xatol": 2.0e-3}
        )
        return float(np.exp(fit.x))

    def fit_scale_and_exponent(
        self, name, target, band, starts, weight_of=None, taper=None
    ):
        """The scale and the exponent together; with `taper` the exponent
        may fall below one, the taper supplying the decay the space needs.
        The best of the fits from `starts`, each (scale in km, exponent)."""
        low, high = band
        log_target = np.log(target[low : high + 1])
        floor, ceiling = (1.05, 8.0) if taper is None else (0.2, taper[1])

        def misfit(x):
            scale_km, exponent = float(np.exp(x[0])), float(x[1])
            lower, upper = self.args.scale_bounds_km
            if not (lower <= scale_km <= upper and floor <= exponent <= ceiling):
                return 1.0e6
            weight = None if weight_of is None else weight_of(scale_km)
            v = self.variances(self.masked(name, scale_km, exponent, weight, taper))
            return float(np.var(np.log(v[low : high + 1]) - log_target))

        best = None
        for start in starts:
            fit = minimize(
                misfit,
                [np.log(start[0]), start[1]],
                method="Nelder-Mead",
                options={"xatol": 5.0e-3, "fatol": 1.0e-5, "maxiter": 150},
            )
            if best is None or fit.fun < best.fun:
                best = fit
        return float(np.exp(best.x[0])), float(best.x[1])

    def variance_beyond(self, scale_km, exponent, taper, degree: int) -> float:
        """The share of the invariant field's pointwise variance above a
        degree, the spectrum carried far beyond the model's own."""
        degrees = np.arange(0, 6001, dtype=float)
        k = degrees * (degrees + 1.0) / self.radius**2
        power = (2.0 * degrees + 1.0) * common.tapered_spectrum(
            k, scale_km * self.km, exponent, self.radius, taper
        )
        return float(power[degree + 1 :].sum() / power.sum())


def band_statistics(model, target) -> list:
    """For each band: the mean ratio of the model's root mean square
    coefficient to the source's (geometric), and the scatter about it as a
    factor."""
    out = []
    for low, high in BANDS:
        d = 0.5 * (np.log(model[low : high + 1]) - np.log(target[low : high + 1]))
        out.append((float(np.exp(d.mean())), float(np.exp(d.std()))))
    return out


def main():
    args = parse_arguments()
    common.start_run(OUTPUT_DIR, args)
    started = time.time()
    fitter = Fitter(args)
    top = args.source_degree
    lines, spectra = [], {}

    # A check of the conventions: unmasked draws have the spectrum asked for.
    v = fitter.variances(fitter.draws(500.0, 2.5))
    expected = fitter.spectrum(500.0, 2.5)[fitter.edges][: top + 1]
    ratio = v[2:61] / expected[2:61]
    lines.append(
        f"Check: degree variances of {args.draws} unmasked draws over those asked "
        f"for, degrees 2 to 60: mean {ratio.mean():.3f}, range {ratio.min():.2f} to "
        f"{ratio.max():.2f}."
    )
    print(lines[-1], flush=True)

    sources = fitter.sources()
    sheets = fitter.ice_sheets()
    lines.append(
        f"Sources: {len(sources[ICE])} year-on-year changes, {args.years[0]} to "
        f"{args.years[1]}; fields at degree {args.lmax}, spectra to degree {top}."
    )

    for name in args.fields:
        print(f"\n{name}...", flush=True)
        fields = sources[name]
        target = fitter.variances(fields)
        exponent = EXPONENTS[name]
        before_scale, before_exponent = BEFORE[name]
        weight_of = weight_before = None
        note = ""
        if name == ICE:
            support = sheets["greenland"] + sheets["antarctic"]
            stds = {s: fitter.rms(m, fields) for s, m in sheets.items()}
            pointwise = fields[0].copy()
            pointwise.data[:] = np.sqrt(np.mean([f.data**2 for f in fields], axis=0))
            pointwise = support * pointwise
            std = fitter.rms(support, fields)
            growths = {}

            def weight_at(scale_km, growth=None):
                if growth is None:
                    key = round(float(scale_km), 1)
                    if key not in growths:
                        growths[key] = fitter.fit_growth(scale_km, pointwise, stds)
                    growth = growths[key]
                return fitter.ice_weight(scale_km, growth, stds)

            weight_of = weight_at
            fitter.tied_margin = True
            weight_before = weight_at(before_scale, args.ice_margin_growth)
            fitter.tied_margin = False
            note = (
                f"  rms over the grounded ice: Greenland {stds['greenland']:.0f}, "
                f"Antarctica {stds['antarctic']:.0f} mm EWT\n"
            )
        else:
            support = fitter.supports[common.LAND] if name == LAND else fitter.sea
            std = fitter.rms(support, fields)

        def model_spectrum(scale_km, q, weight=None, taper=None):
            if name == ICE and weight is None:
                weight = weight_of(scale_km)
            v = fitter.variances(fitter.masked(name, scale_km, q, weight, taper))
            return v if name == ICE else v * std**2

        if fitter.taper is not None and name != STERIC:
            # The source's own decay as far as it goes, then the taper.
            scale, exponent = fitter.fit_scale_and_exponent(
                name,
                target,
                (2, top),
                [(300.0, 1.2), (1000.0, 0.8)],
                weight_of,
                fitter.taper,
            )
            fitted = model_spectrum(scale, exponent, taper=fitter.taper)
            before = model_spectrum(before_scale, before_exponent, weight_before)
            unmasked = (
                std**2
                * fitter.spectrum(scale, exponent, fitter.taper)[fitter.edges][: top + 1]
            )
            spectra[name] = dict(
                source=target,
                fit_2_60=fitted,
                fit_2_top=fitted,
                before=before,
                unmasked=unmasked,
                scales=np.array([scale, scale]),
            )
            lines.append(
                f"\n{name}: exponent {exponent:.2f}, scale {scale:.0f} km, taper at "
                f"degree {fitter.taper[0]:.0f} to exponent {fitter.taper[1]} "
                f"(rms over the support {std:.2f} mm)"
            )
            if note:
                lines.append(note.rstrip("\n"))
            if name == ICE:
                growth = fitter.fit_growth(scale, pointwise, stds)
                lines.append(
                    f"  growth {growth:.1f}, nearness at "
                    f"{args.ice_margin_scale_km:.0f} km"
                )
            lines.append(
                "  model's rms coefficient over the source's (mean ratio, scatter "
                f"as a factor), degrees {BANDS[0][0]}-{BANDS[0][1]} | "
                f"{BANDS[1][0]}-{BANDS[1][1]}:"
            )
            for label, v in (("fit", fitted), ("before", before)):
                (m1, s1), (m2, s2) = band_statistics(v, target)
                lines.append(
                    f"    {label:10s} {m1:5.2f} x/{s1:4.2f} | {m2:5.2f} x/{s2:4.2f}"
                )
            lines.append(
                "  share of the prior's pointwise variance above degree 180: "
                f"{fitter.variance_beyond(scale, exponent, fitter.taper, 180):.3f}, "
                f"above 255: {fitter.variance_beyond(scale, exponent, fitter.taper, 255):.3f}"
            )
            print("\n".join(lines[-7:]), flush=True)
            continue

        results = {}
        for band in ((2, 60), (2, top)):
            scale = fitter.fit_scale(name, exponent, target, band, weight_of)
            results[band] = (scale, model_spectrum(scale, exponent))
            print(f"  fit over degrees {band[0]} to {band[1]}: {scale:.0f} km", flush=True)
        before = model_spectrum(before_scale, before_exponent, weight_before)
        scale, fitted = results[(2, 60)]
        unmasked = std**2 * fitter.spectrum(scale, exponent)[fitter.edges][: top + 1]
        spectra[name] = dict(
            source=target,
            fit_2_60=fitted,
            fit_2_top=results[(2, top)][1],
            before=before,
            unmasked=unmasked,
        )

        lines.append(f"\n{name} (exponent {exponent}; rms over the support {std:.2f} mm)")
        if note:
            lines.append(note.rstrip("\n"))
        rows = [
            (f"fit, degrees 2 to 60: {scale:.0f} km", fitted),
            (
                f"fit, degrees 2 to {top}: {results[(2, top)][0]:.0f} km",
                results[(2, top)][1],
            ),
            (f"before: {before_scale:.0f} km, exponent {before_exponent}", before),
        ]
        if args.free_exponent:
            free = fitter.fit_scale_and_exponent(
                name, target, (2, top), [(scale, exponent)], weight_of
            )
            rows.append(
                (
                    f"exponent free, degrees 2 to {top}: {free[0]:.0f} km, "
                    f"exponent {free[1]:.2f}",
                    model_spectrum(*free),
                )
            )
            spectra[name]["free"] = rows[-1][1]
        lines.append(
            "  model's rms coefficient over the source's (mean ratio, scatter as a "
            f"factor), degrees {BANDS[0][0]}-{BANDS[0][1]} | {BANDS[1][0]}-{BANDS[1][1]}:"
        )
        for label, v in rows:
            (m1, s1), (m2, s2) = band_statistics(v, target)
            lines.append(f"    {label:58s} {m1:5.2f} x/{s1:4.2f} | {m2:5.2f} x/{s2:4.2f}")
        if name == ICE:
            for key in (round(scale, 1), round(results[(2, top)][0], 1)):
                if key in growths:
                    lines.append(f"  growth at {key:.0f} km: {growths[key]:.1f}")
            spectra[name]["growth"] = fitter.fit_growth(scale, pointwise, stds)
            lines.append(f"  growth at the fitted scale: {spectra[name]['growth']:.1f}")
        spectra[name]["scales"] = np.array([scale, results[(2, top)][0]])
        print("\n".join(lines[-8:]), flush=True)

    lines.append(f"\nTime: {time.time() - started:.0f} s.")
    with open(f"{OUTPUT_DIR}/prior_fit.txt", "w") as file:
        file.write("\n".join(lines) + "\n")
    np.savez(
        f"{OUTPUT_DIR}/prior_fit.npz",
        **{f"{n}|{k}": v for n, d in spectra.items() for k, v in d.items()},
    )

    degrees = np.arange(top + 1)
    fig, axes = plt.subplots(
        1, len(spectra), figsize=(5.2 * len(spectra), 5.0), layout="constrained"
    )
    for ax, (name, d) in zip(np.atleast_1d(axes), spectra.items()):
        ax.loglog(degrees[1:], np.sqrt(d["source"][1:]), color="black", label="Source")
        # With the taper the other fields have one fit; the steric height,
        # with its exponent fixed, is shown by its fit to degree 60 alone.
        single = args.taper_degree > 0.0 or bool(
            np.allclose(d["fit_2_60"], d["fit_2_top"])
        )
        ax.loglog(
            degrees[1:],
            np.sqrt(d["fit_2_60"][1:]),
            color="tab:blue",
            label="Prior, as its source shows it"
            if single
            else f"Masked field, {d['scales'][0]:.0f} km (fit to degree 60)",
        )
        if not single:
            ax.loglog(
                degrees[1:],
                np.sqrt(d["fit_2_top"][1:]),
                color="tab:blue",
                linestyle="--",
                label=f"Masked field, {d['scales'][1]:.0f} km (fit to degree {top})",
            )
        ax.loglog(
            degrees[1:],
            np.sqrt(d["unmasked"][1:]),
            color="tab:blue",
            linestyle=":",
            label="The same field without the mask",
        )
        if args.plot_before:
            ax.loglog(
                degrees[1:],
                np.sqrt(d["before"][1:]),
                color="tab:red",
                linewidth=1,
                label=f"Before: {BEFORE[name][0]:.0f} km, exponent {BEFORE[name][1]}",
            )
        if "free" in d:
            ax.loglog(
                degrees[1:], np.sqrt(d["free"][1:]), color="tab:green", linewidth=1,
                label="Exponent free",
            )
        floor = np.sqrt(d["source"][1:].min()) / 3.0
        ax.set_ylim(floor, None)
        ax.set_title(f"{name[0].upper()}{name[1:]}")
        ax.set_xlabel("Degree")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("Root mean square coefficient (mm)")
    fig.savefig(f"{OUTPUT_DIR}/prior_fit.png", dpi=200)
    print(f"\nWritten to {OUTPUT_DIR}/ in {time.time() - started:.0f} s.")


if __name__ == "__main__":
    main()
