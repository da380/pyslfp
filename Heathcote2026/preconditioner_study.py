"""
A study of preconditioners for the ocean experiments (Heathcote et al.,
2026).

The inversions solve the data-space normal equations, (A Q A* + R) x = d,
by conjugate gradients. The posterior expectation costs one solve; every
quantity of interest, every row of an estimator kernel and every posterior
sample costs another at a tighter tolerance, so the number of iterations
per solve sets the cost of the experiments. This script builds the problem
of `joint_inversion.py` (the altimetry data, GRACE's, or both), draws one
set of synthetic data, and solves the same system under a list of
preconditioners, recording for each the cost of building it and the
iterations to three tolerances.

The preconditioners, by name (--variants):

- none, noise: no preconditioner, and the inverse of the local noise.
- base: the one the inversions use (`common.surrogate_preconditioner`),
  the Woodbury inverse of a surrogate problem at --surrogate-degree, mixed
  with the inverse of the noise with weight alpha = 0.1.
- alpha-<a>: the same with another mixing weight.
- degree-<n>: the same with the surrogate at another degree.
- all-loads: the surrogate keeps the loads of every field, not of the mass
  sources alone.
- full-sle: the surrogate solves the sea level equation, not one pass.
- unresolved, unresolved-alpha-0: the signal the surrogate cannot produce,
  that of degrees above its own, is added to its noise as a variance
  (`common.unresolved_altimetry_variance`, `common.unresolved_grace_variance`),
  with the usual mixing and with none.
- low-rank: the base corrected for the parts of the noise the surrogate
  leaves out (`common.low_rank_correction`): the common modes of the
  altimetry, the error of the GIA correction, and the leading --low-rank
  modes of the large-scale correlated field.
- both, both-alpha-0: the last two together.
- matched, matched-<any of the above>: the surrogate's prior has the
  variances of the true prior at the degrees it keeps
  (`common.product_prior` with a reference), in place of the same pointwise
  std at its own truncation.
- wmb: for GRACE's coefficients, the diagonal the GRACE experiments use:
  the prior's variance in each coefficient as WMB would have it, plus the
  noise, with no surrogate; the altimetry points keep the inverse of their
  local noise plus the variance of eta.
- cheap: the exact inverse of the normal matrix of cheap physics, with the
  true prior, assembled in the data space and factorised. The cheap physics
  has no sea level equation: GRACE's coefficients are those of the loads
  through the load Love numbers, as WMB has them, and the altimetry sees
  eta alone. The two sets of data are then uncoupled. It costs one
  application of the cheap operator and the prior per datum, and a dense
  factorisation.
- cheap-low-rank: that, corrected for the low-rank parts of the noise.
- cheap-fingerprint: that, corrected also for what the mass sources add to
  the altimetry through the sea level equation, from a surrogate at
  --surrogate-degree with the true prior's variances.
- local, local-low-rank, local-fingerprint: as the three before, but
  nothing dense is formed on the altimetry data, whose number is not
  bounded. The covariance of eta between the points, tapered to vanish
  beyond --cutoff-km (the taper of Gaspari and Cohn, which keeps it
  positive definite), is a sparse matrix, factorised once. GRACE's block
  is still dense: it has at most a few thousand coefficients.
- exact, exact-low-rank: as local and local-low-rank, but GRACE's block is
  that of the true operator, sea level equation included, assembled at the
  cost of one application of the operator and its adjoint per coefficient.
  For GRACE's data alone the preconditioner is then the inverse of the
  normal matrix but for the low-rank parts of the noise; with the altimetry
  it leaves out only the coupling of the two sets of data.
- wmb-block, reduced-<n>, wmb-sle-<n>, the last two with any of the
  suffixes -taper, -smooth and -fields: as exact-low-rank, with GRACE's
  block built more cheaply (`Problem.grace_signal`). wmb-block: WMB's
  physics, the load Love numbers and no sea level equation, at the
  problem's degree and with the true prior. reduced-<n>: the true physics,
  sea level equation included, on a model of degree n, given the problem's
  loads truncated at that degree; the coefficients above n have the noise
  alone. wmb-sle-<n>: WMB's physics at the problem's degree plus what the
  sea level equation adds to it on the model of degree n. wmb-sl-<n>: the
  model of degree n gives the sea level change alone, and its ocean load is
  formed on the problem's own ocean function, so that it reaches every
  degree of the data. -taper: the
  loads given to the model of degree n fall smoothly to zero towards n.
  -smooth: that model's ocean function is the problem's, band-limited, and
  not the sharp one of its own coarse grid. -fields: the fields are
  truncated before they are masked, and not the loads after. The low-rank
  parts of the noise are corrected for as in exact-low-rank.
- exact-block: exact-low-rank by the same route as those, so that
  --block-cache keeps it.
- model-<n>: as those, with GRACE's block built wholly on a model of degree
  n: its own grid, masks, loads and sea level equation, and the prior's
  independent fields truncated at n. No operation is at the problem's
  degree. At n the problem's degree it is the exact block.
- <any local, exact or block variant>+vecchia-<k>: the altimetry's part is
  not the tapered covariance but the sparse inverse of `vecchia.py`, each
  point regressed on its k nearest neighbours among those before it in the
  maximin order: no cut-off, no taper, positive definite by construction.
- two-level, two-level-low-rank: the scalable form of both ideas. Each
  field's prior spectrum is split smoothly at about half of
  --surrogate-degree into a long-wavelength part and the rest. The
  long-wavelength part goes to the surrogate, with its sea level equation
  and the coupling of the two data sets; the rest is treated as a further
  term of the noise in the surrogate problem, sparse on the altimetry
  points and dense on GRACE's coefficients, as in `local`.
- hybrid, hybrid-low-rank: the two-level form for the altimetry and the
  local one for GRACE, uncoupled. The surrogate predicts the altimetry
  alone, from the mass sources and the long-wavelength part of eta;
  GRACE's block is the dense one of `local`, with the whole spectrum.
- spectral-<r>: `pygeoinf.SpectralPreconditioner` at rank r on the operator
  itself, which uses no surrogate and costs applications of the full
  operator to build.
- sketch-<k> or sketch-<k>-<block>: GRACE's block from k applications of
  the physics of a block above (the exact one if none is named) in place
  of one a datum: the randomised Nystrom approximation in the metric of
  the noise (`Problem.sketched_signal`). The cost of the build is k rows
  of that block.

Output, in output_plots_preconditioner_study/: preconditioner_metrics.txt,
a table per case.
"""

import argparse
import copy
import hashlib
import os
import time
from typing import Optional

import numpy as np
from scipy import linalg, sparse

import pygeoinf as inf
from pyslfp import EarthState, set_num_threads
from pyslfp.core import sobolev_load_space
from pyslfp.core.spaces import underlying_space
from pyslfp.elastic import FingerPrintOperator
from pyslfp.models import sea_level_change_to_load_operator

import common
import recycling
from vecchia import VecchiaInverse, unit_vectors

OUTPUT_DIR = "output_plots_preconditioner_study"
METRICS_FILE = os.path.join(OUTPUT_DIR, "preconditioner_metrics.txt")
TOLERANCES = (1.0e-2, 1.0e-4, 1.0e-6)
# The most altimetry points on which a variant may form a dense matrix.
DENSE_ALTIMETRY_LIMIT = 5000
CASES = ("altimetry", "grace", "joint")
DEFAULT_VARIANTS = [
    "noise",
    "base",
    "alpha-0",
    "alpha-0.03",
    "alpha-0.3",
    "degree-half",
    "all-loads",
    "full-sle",
    "unresolved",
    "unresolved-alpha-0",
    "low-rank",
    "both",
    "both-alpha-0",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteration counts of the ocean inversions under a list "
        "of preconditioners."
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument(
        "--low-rank",
        type=int,
        default=40,
        help="Modes kept of the large-scale correlated altimetry noise.",
    )
    parser.add_argument(
        "--seed-directions",
        type=int,
        default=0,
        help="After each variant's solve, the data's solve again with the "
        "preconditioner corrected by this many random directions "
        "(`recycling.RecycledSolver.seed`): a randomised low-rank correction "
        "of the preconditioned operator.",
    )
    parser.add_argument(
        "--seed-power",
        type=int,
        default=1,
        help="With --seed-directions: powers of the preconditioned operator "
        "applied to the random vectors.",
    )
    parser.add_argument(
        "--block-cache",
        default=None,
        metavar="DIRECTORY",
        help="Keep the signal part of each GRACE block in this directory, "
        "under a name made from every option that is not listed as leaving "
        "it unchanged, and read it back if it is there. A change to the "
        "code is not seen: empty the directory after one.",
    )
    parser.add_argument(
        "--recycle",
        type=int,
        default=0,
        help="After each variant's solve, this many further solves with "
        "other right-hand sides (`Problem.right_hand_sides`), each reusing "
        "the search directions of those before it (`recycling.py`), and "
        "the same solves without reuse, for comparison.",
    )
    parser.add_argument(
        "--legacy-grace-noise",
        action="store_true",
        help="Give GRACE the noise of the submitted paper's runs, in which "
        "each coefficient's variance was the eigenvalue of the noise "
        "field's covariance operator, not that eigenvalue over the metric "
        "of the field's space: on H^2 the variance is then too large by "
        "(1 + s^2 l (l + 1))^2, with s the scale of the space.",
    )
    parser.add_argument(
        "--grace-white-crossover",
        type=int,
        default=None,
        help="Give GRACE noise that is white as a load, the same variance "
        "in every coefficient of the load before the load Love numbers, at "
        "the level of the signal's own variance at this degree, estimated "
        "from samples of the prior. The signal's spectrum falls with "
        "degree, so the signal-to-noise ratio does too and is one here.",
    )
    parser.add_argument(
        "--preconditioner-floor-mm",
        type=float,
        default=0.0,
        help="Add to GRACE's noise, in the preconditioners that assemble its "
        "block and nowhere else, an error that is white as a load with this "
        "pointwise std in mm EWT.",
    )
    parser.add_argument(
        "--cutoff-km",
        type=float,
        default=1500.0,
        help="The distance beyond which the altimetry block of the local "
        "and two-level preconditioners takes two points as uncorrelated.",
    )
    common.add_ocean_arguments(parser, surrogate=True)
    common.add_altimetry_noise_arguments(parser, prefix="alt-")
    common.add_joint_grace_noise_arguments(parser)
    common.add_gia_arguments(parser)
    common.add_parallel_arguments(parser)
    return parser.parse_args()


gaspari_cohn, SparseSolve, definite = (
    common.gaspari_cohn,
    common.SparseSolve,
    common.definite,
)


def low_pass(space, degree: Optional[int]) -> np.ndarray:
    """The weight, per component of a sphere space, of the long-wavelength
    part of a spectrum split at about half of `degree`: a Gaussian in the
    degree, negligible at `degree` itself. Zero everywhere if `degree` is
    None, so that nothing is long-wavelength."""
    if degree is None:
        return np.zeros(space.dim)
    return np.exp(-((space.degrees / (0.5 * degree)) ** 2))


def taper(space, degree: int) -> np.ndarray:
    """The weight, per component of a sphere space, of a taper that is one
    to half of `degree` and falls as a squared cosine to zero just above
    `degree`."""
    start = 0.5 * degree
    phase = np.clip((space.degrees - start) / (degree + 1.0 - start), 0.0, 1.0)
    return np.cos(0.5 * np.pi * phase) ** 2


def assemble(operator, n_jobs=None) -> np.ndarray:
    """The matrix of a self-adjoint operator on a Euclidean data space, one
    application per datum. Each worker is handed one share of the columns,
    so that the operator is carried to it once, and workers are used only
    if the whole assembly is slow (as `common.exact_preconditioner`)."""
    space = operator.domain

    def columns(indices):
        return np.array(
            [
                space.to_components(operator(space.basis_vector(int(index))))
                for index in indices
            ]
        )

    start = time.perf_counter()
    columns([0])
    serial_seconds = (time.perf_counter() - start) * space.dim
    workers = 1 if n_jobs is None or serial_seconds < 60.0 else n_jobs
    shares = np.array_split(np.arange(space.dim), workers)
    matrix = np.vstack(inf.parallel.parallel_map(columns, shares, n_jobs=workers))
    return 0.5 * (matrix + matrix.T)


def images(operator, probes: np.ndarray, n_jobs=None) -> np.ndarray:
    """The images of the columns of `probes` under an operator on a
    Euclidean data space, as columns: one application each, shared between
    workers as in `assemble`."""
    space = operator.domain

    def columns(indices):
        return np.array(
            [
                space.to_components(operator(space.from_components(probes[:, index])))
                for index in indices
            ]
        )

    count = probes.shape[1]
    start = time.perf_counter()
    columns([0])
    serial_seconds = (time.perf_counter() - start) * count
    workers = 1 if n_jobs is None or serial_seconds < 60.0 else n_jobs
    shares = np.array_split(np.arange(count), workers)
    return np.vstack(inf.parallel.parallel_map(columns, shares, n_jobs=workers)).T


def white_load_noise(physics, args, prior, forward, degree: int, rng, samples=30):
    """
    GRACE noise that is white as a load: every coefficient of the load has
    the same variance, taken to GRACE's coefficients by the load Love
    numbers, at the level at which it equals the signal's variance at
    `degree`. The signal's variance by degree is estimated from `samples`
    draws of the prior through `forward`.
    """
    space = physics.load_space
    degrees = space.degrees
    observed = degrees[(degrees >= 2) & (degrees <= args.obs_degree)]
    signal = np.array([forward(prior.sample(rng=rng)) for _ in range(samples)])
    signal_variance = np.mean(signal[:, observed == degree] ** 2)
    # A unit variance in every load coefficient, as an invariant measure:
    # the eigenvalues carry the metric of the space.
    unit = inf.GaussianMeasure(
        space, covariance=inf.DiagonalLinearOperator(space, space.metric_values)
    )
    shape = (
        common.wmb_method(physics.state, args)
        .load_measure_to_observation_measure(unit)
        .covariance.diagonals()[0]
    )
    variance = shape * signal_variance / np.mean(shape[observed == degree])
    ratio = np.sqrt(
        [
            np.mean(signal[:, observed == l] ** 2) / np.mean(variance[observed == l])
            for l in (2, 8, 16, 32, min(60, args.obs_degree), args.obs_degree)
        ]
    )
    print(
        "  GRACE noise white as a load; signal-to-noise at degrees 2, 8, 16, 32, "
        f"{min(60, args.obs_degree)}, {args.obs_degree}: "
        + ", ".join(f"{r:.2f}" for r in ratio),
        flush=True,
    )
    domain = forward.codomain
    return inf.GaussianMeasure.from_standard_deviations(domain, np.sqrt(variance))


class _Inverse:
    """A `vecchia.VecchiaInverse` with the two methods of
    `common.SparseSolve`: `apply` the covariance, `solve` its inverse."""

    def __init__(self, approximate):
        self.apply, self.solve = approximate.forward, approximate.inverse


class Problem:
    """One case of the study: its forward operator, noise and data, and what
    a preconditioner may use of them."""

    def __init__(self, case, physics, prior, calib, args, rng):
        self.case, self.physics, self.args, self.calib = case, physics, args, calib
        self.prior, self._cheap, self._short, self._reduced = prior, None, {}, {}
        stds = calib["derived_stds"]
        parts = {}
        alt_noise, self.alt_local = common.altimetry_noise(
            physics,
            args.prior_order,
            parts=parts,
            **common.altimetry_noise_options(args, "alt_"),
        )
        self.grace_noise, grace_info = common.joint_grace_noise(
            physics, stds["ice_std"], args.prior_order, args
        )
        # GRACE's noise covariance as a matrix, if it is not diagonal.
        self.grace_matrix = grace_info.get("covariance")
        if args.legacy_grace_noise:
            space = physics.field_space
            observed = (space.degrees >= 2) & (space.degrees <= args.obs_degree)
            self.grace_noise = common.with_added_variance(
                self.grace_noise,
                self.grace_noise.covariance.diagonals()[0]
                * (space.metric_values[observed] - 1.0),
            )
        altimetry, grace = physics.altimetry, physics.grace(args)
        if args.grace_white_crossover is not None:
            self.grace_noise = white_load_noise(
                physics, args, prior, grace, args.grace_white_crossover, rng
            )
            self.grace_matrix = None
        use_alt, use_grace = case != "grace", case != "altimetry"
        if case == "joint":
            self.forward = physics.joint(args)
            noise = inf.GaussianMeasure.from_product(
                [alt_noise, self.grace_noise], domain=self.forward.codomain
            )
        else:
            self.forward = altimetry if use_alt else grace
            noise = alt_noise if use_alt else self.grace_noise
        gia = common.ocean_gia_error(physics, args, altimetry=use_alt, grace=use_grace)
        if gia is not None:
            noise = noise + gia
        self.noise = noise
        self.space = self.forward.codomain

        # The low-rank factors of the noise beyond its local part, each into
        # this case's data space.
        self.factors = []
        if use_alt:
            for name, measure in parts.items():
                if name == "correlated":
                    measure = measure.low_rank_approximation(
                        rank=min(args.low_rank, measure.domain.dim - 1), rng=rng
                    )
                factor = measure.covariance_factor
                if case == "joint":
                    factor = self.space.inclusion(0) @ factor
                self.factors.append(factor)
        if gia is not None:
            self.factors.append(gia.covariance_factor)

        problem = inf.LinearForwardProblem(self.forward, error=noise)
        self.forward_problem = problem
        _, self.data = problem.synthetic_model_and_data(prior, rng=rng)
        self.normal = inf.LinearGaussianInversion(problem, prior).normal_operator

    def right_hand_sides(self, count: int, rng) -> list:
        """Right-hand sides of the kinds a run solves for after the data's:
        alternately a new draw of the data, as a posterior sample needs, and
        A Q B* e for a quantity of interest B, as a row of its posterior
        covariance needs. The quantities are the barystatic and steric
        global mean sea level changes and the three parts of the regional
        mean sea level change. Returns (label, vector) pairs."""
        setup = self.physics.setup
        targets = [
            ("barystatic GMSL", setup.barystatic_gmsl(), 0),
            ("steric GMSL", setup.steric_gmsl, 0),
        ]
        regional = self.physics.regional_decomposition(common.OCEAN_REGION)
        targets += [(f"regional part {i + 1}", regional, i) for i in range(3)]
        sides = []
        for index in range(count):
            if index % 2 == 0:
                _, data = self.forward_problem.synthetic_model_and_data(
                    self.prior, rng=rng
                )
                sides.append(("data draw", data))
            else:
                label, target, row = targets[(index // 2) % len(targets)]
                unit = target.codomain.basis_vector(row)
                sides.append(
                    (label, self.forward(self.prior.covariance(target.adjoint(unit))))
                )
        return sides

    def add_grace_noise(self, matrix: np.ndarray) -> None:
        """Adds GRACE's noise covariance to a matrix on its data, in place."""
        if self.grace_matrix is not None:
            matrix += self.grace_matrix
        else:
            matrix[np.diag_indices_from(matrix)] += (
                self.grace_noise.covariance.diagonals()[0]
            )
        if self.args.preconditioner_floor_mm > 0.0:
            matrix[np.diag_indices_from(matrix)] += common.white_load_variance(
                self.physics.state,
                self.physics.load_space,
                self.args,
                self.args.preconditioner_floor_mm,
            )

    def grace_noise_with(self, variance: np.ndarray):
        """GRACE's noise with a variance added to each coefficient."""
        if self.grace_matrix is None:
            return common.with_added_variance(self.grace_noise, variance)
        return inf.GaussianMeasure.from_covariance_matrix(
            self.grace_noise.domain,
            self.grace_matrix + np.diag(variance),
            form="galerkin",
        )

    def cheap_inverse(self, n_jobs):
        """The inverse of the normal matrix of cheap physics, built once."""
        if self._cheap is None:
            physics, args = self.physics, self.args
            setup = physics.setup
            # A dense matrix on every datum: for small tests alone. The
            # altimetry points are not bounded in number, and nothing dense
            # is to be formed on them at scale (`short_noise` is sparse).
            if self.case != "grace" and len(physics.points) > DENSE_ALTIMETRY_LIMIT:
                raise ValueError(
                    f"The cheap variants form a dense matrix on the altimetry "
                    f"data; refused above {DENSE_ALTIMETRY_LIMIT} points. Use "
                    "the local variants."
                )
            parts = []
            if self.case != "grace":
                parts.append(
                    physics.fingerprint.field_space.point_evaluation_operator(
                        physics.points
                    )
                    @ setup.observables[common.GEOCENTRIC_SEA_LEVEL]
                )
            if self.case != "altimetry":
                wmb = common.wmb_method(physics.state, args)
                parts.append(
                    wmb.load_to_potential_coefficient_operator(physics.load_space)
                    @ setup.load_operator
                )
            forward = parts[0] if len(parts) == 1 else inf.ColumnLinearOperator(parts)
            signal = forward @ self.prior.covariance @ forward.adjoint
            matrix = signal.matrix(form="components", by="rows", n_jobs=n_jobs)
            matrix = 0.5 * (matrix + matrix.T)
            # The altimetry data come first, with a diagonal local noise.
            count = 0 if self.case == "grace" else self.alt_local.domain.dim
            if count:
                matrix[np.arange(count), np.arange(count)] += (
                    self.alt_local.covariance.diagonals()[0]
                )
            if self.case != "altimetry":
                self.add_grace_noise(matrix[count:, count:])
            self._cheap = inf.LinearOperator.from_matrix(
                self.space, self.space, np.linalg.inv(matrix), form="components"
            )
        return self._cheap

    def reduced_model(self, degree: int, smooth: bool):
        """A model of degree `degree` for the sea level equation's part of
        GRACE's block: its fingerprint operator on loads, the truncation of
        the problem's loads to it, the coefficients it predicts and WMB's
        map on it. With `smooth`, its ocean function is not the sharp one
        of its own coarse grid but the problem's, truncated at `degree`
        with a taper: a band-limited function that takes values between
        nought and one along the coasts."""
        key = (degree, smooth)
        if key not in self._reduced:
            physics, args = self.physics, self.args
            state = EarthState.from_defaults(lmax=degree)
            scale = args.load_scale_km * 1000.0 / state.model.parameters.length_scale
            if smooth:
                flat = underlying_space(physics.load_space)
                coarse = underlying_space(
                    sobolev_load_space(state.model, args.load_order, scale)
                )
                # Set before anything reads the state's own, which is made
                # on first use and kept.
                state.__dict__["ocean_function"] = coarse.spectral_operator(
                    taper(coarse, degree)
                )(flat.degree_transfer_operator(coarse)(physics.state.ocean_function))
            fingerprint = FingerPrintOperator(
                state,
                load_parameters=(args.load_order, scale),
                response_parameters=(args.load_order + 1.0, scale),
            )
            reduced_args = copy.copy(args)
            reduced_args.obs_degree = min(args.obs_degree, degree)
            observe = common.grace_coefficients(
                fingerprint.codomain,
                state.model,
                reduced_args.obs_degree,
                args.grace_convention,
            )
            cheap = common.wmb_method(
                state, reduced_args
            ).load_to_potential_coefficient_operator(fingerprint.domain)
            transfer = physics.load_space.degree_transfer_operator(fingerprint.domain)
            # The sea level change of the model, as a field of its own and
            # as one of the problem's degree, and the ocean load it makes on
            # each: on the model's coarse ocean function, and on the
            # problem's.
            sea_level = fingerprint.codomain.field_operator("relative_sea_level_change")
            fine = physics.fingerprint.codomain.field_operator(
                "relative_sea_level_change"
            ).codomain
            ocean_loads = (
                sea_level,
                sea_level_change_to_load_operator(
                    state, sea_level.codomain, fingerprint.domain
                ),
                sea_level_change_to_load_operator(
                    physics.state, fine, physics.load_space
                )
                @ sea_level.codomain.degree_transfer_operator(fine),
            )
            self._reduced[key] = (fingerprint, transfer, observe, cheap, ocean_loads)
        return self._reduced[key]

    def grace_signal(self, mode, surrogates, n_jobs, probes=None) -> np.ndarray:
        """
        The covariance of GRACE's coefficients under the true prior, with
        physics that is cheaper than the true one: the whole matrix, one
        application a datum, or with `probes`, an array (data, k), its
        product with them alone, k applications. `mode` is

        - ("exact",): the true physics, nothing cheaper;

        - ("wmb",): WMB's physics at the problem's degree, W L, the loads L
          of the fields taken to the coefficients by the load Love numbers;
        - ("model", n, flags): everything on a model of degree n, its
          own masks, loads and sea level equation, with the prior's
          independent fields truncated at n (`common.ocean_prior` with
          a reference): O_n F_n L_n Q_n L_n* F_n* O_n*. No operation is
          at the problem's degree, so the cost of a row does not depend
          on it. Coefficients above degree n have no signal;
        - ("reduced", n, flags): the true physics of a model of degree n,
          sea level equation included, O_n F_n T L: the loads are those of
          the problem, masks and all, truncated at n by T. Coefficients
          above degree n have no signal;
        - ("wmb-sle", n, flags): W L + (O_n F_n - W_n) T L, WMB's physics at
          the problem's degree plus what the sea level equation adds on the
          model of degree n;
        - ("wmb-sl", n, flags): the model of degree n supplies the sea level
          change alone, a smooth field, and the ocean load it makes is
          formed on the problem's own ocean function: W (L + D P S_n T L)
          + (O_n F_n - W_n (1 + D_n S_n)) T L, with S_n the model's sea
          level change, P its prolongation to the problem's degree, D the
          ocean load of a sea level change on the problem's ocean function
          and D_n that on the model's. The ocean load then has its
          coefficients at every degree of the data, which the coarse ocean
          function cannot give it; the last term is what the model predicts
          beyond the loads' own attraction, the rotational part, at its own
          degrees.

        `flags` is a set of: "taper", the truncated loads fall smoothly to
        zero towards n; "smooth", the model of degree n has a band-limited
        ocean function (`reduced_model`); "fields", the fields are
        truncated before they are masked, on the model's own coarse masks,
        and not the loads after (the first form tried: the low-degree load
        then lacks what the masks bring down from the fields' higher
        degrees).
        """
        physics, args = self.physics, self.args
        setup = physics.setup
        covariance = self.prior.covariance
        domain = self.grace_noise.domain

        def matrix_of(operator):
            # The coefficients are ordered by degree: an operator on fewer
            # than all of them is on those that lead, and the rest have no
            # signal.
            count = operator.domain.dim
            if probes is None:
                block = assemble(operator, n_jobs)
                if count == domain.dim:
                    return block
                matrix = np.zeros((domain.dim, domain.dim))
                matrix[:count, :count] = block
                return matrix
            matrix = np.zeros((domain.dim, probes.shape[1]))
            matrix[:count] = images(operator, probes[:count], n_jobs)
            return matrix

        if mode[0] == "model":
            reduced_args = copy.copy(args)
            reduced_args.lmax = mode[1]
            reduced_args.obs_degree = min(args.obs_degree, mode[1])
            model = common.ocean_physics(reduced_args, points=physics.points)
            reference = {
                name: measure.covariance.eigenvalues
                for name, measure in self.calib["independent"]["measures"].items()
            }
            prior, _ = common.ocean_prior(
                model, reduced_args, np.random.default_rng(0), reference=reference
            )
            grace = model.grace(reduced_args)
            return matrix_of(grace @ prior.covariance @ grace.adjoint)
        wmb = None
        if mode[0] != "reduced":
            wmb = (
                common.wmb_method(
                    physics.state, args
                ).load_to_potential_coefficient_operator(physics.load_space)
                @ setup.load_operator
            )
        if mode[0] == "wmb":
            return matrix_of(wmb @ covariance @ wmb.adjoint)
        if mode[0] == "exact":
            grace = physics.grace(args)
            return matrix_of(grace @ covariance @ grace.adjoint)
        _, degree, flags = mode
        if "fields" in flags:
            surrogate, _ = surrogates.get(degree, "all", None, True)
            transfers = []
            for name in setup.names:
                space = setup.component(name).space
                target = surrogate.setup.component(name).space
                transfer = space.degree_transfer_operator(target)
                if "taper" in flags:
                    transfer = (
                        target.spectral_operator(taper(target, degree)) @ transfer
                    )
                transfers.append(transfer)
            transfer = inf.BlockDiagonalLinearOperator(
                transfers, domain=setup.space, codomain=surrogate.setup.space
            )
            reduced_args = copy.copy(args)
            reduced_args.obs_degree = min(args.obs_degree, degree)
            full = (
                common.grace_coefficients(
                    surrogate.fingerprint.codomain,
                    surrogate.state.model,
                    reduced_args.obs_degree,
                    args.grace_convention,
                )
                @ surrogate.fingerprint
                @ surrogate.setup.load_operator
                @ transfer
            )
            cheap = (
                common.wmb_method(
                    surrogate.state, reduced_args
                ).load_to_potential_coefficient_operator(surrogate.load_space)
                @ surrogate.setup.load_operator
                @ transfer
            )
        else:
            fingerprint, transfer, observe, cheap, ocean_loads = self.reduced_model(
                degree, "smooth" in flags
            )
            if "taper" in flags:
                target = fingerprint.domain
                transfer = target.spectral_operator(taper(target, degree)) @ transfer
            loads = transfer @ setup.load_operator
            if mode[0] == "wmb-sl":
                sea_level, coarse_load, fine_load = ocean_loads
                level = sea_level @ fingerprint @ loads
                wmb_load = common.wmb_method(
                    physics.state, args
                ).load_to_potential_coefficient_operator(physics.load_space)
                count = observe.codomain.dim
                pad = domain.coordinate_selection(np.arange(count)).adjoint
                rest = observe @ fingerprint @ loads - cheap @ (
                    loads + coarse_load @ level
                )
                forward = wmb + wmb_load @ fine_load @ level + pad @ rest
                return matrix_of(forward @ covariance @ forward.adjoint)
            full = observe @ fingerprint @ loads
            cheap = cheap @ loads
        count = full.codomain.dim
        if mode[0] == "reduced":
            return matrix_of(full @ covariance @ full.adjoint)
        pad = domain.coordinate_selection(np.arange(count)).adjoint
        forward = wmb + pad @ (full - cheap)
        return matrix_of(forward @ covariance @ forward.adjoint)

    def sketched_signal(self, mode, rank: int, surrogates, n_jobs) -> np.ndarray:
        """
        `grace_signal` of a mode from `rank` applications in place of one a
        datum: the randomised Nystrom approximation (Frangella, Tropp and
        Udell, 2023) in the metric of GRACE's noise.

        With R = L L^T the noise, the signal in the noise's metric is
        M = L^-1 S L^-T, whose eigenvalues above one are the directions the
        data resolve. Its sketch is M G for G white, that is L^-1 S applied
        to the probes L^-T G: `rank` applications of S. From it comes
        M ~ U E U^T of that rank, the best the sketch can give, and the
        signal L U E U^T L^T. What is left out is signal below the noise.

        Returned as a matrix on the data, as `grace_signal` is: GRACE's
        noise is one already. Nothing needs it formed, the inverse of the
        sum being L^-T (1 - U E (1 + E)^-1 U^T) L^-1.
        """
        dim = self.grace_noise.domain.dim
        noise = np.zeros((dim, dim))
        self.add_grace_noise(noise)
        lower = linalg.cholesky(noise, lower=True)
        white = np.random.default_rng(0).standard_normal((dim, rank))
        probes = linalg.solve_triangular(lower, white, lower=True, trans="T")
        sketch = linalg.solve_triangular(
            lower,
            self.grace_signal(mode, surrogates, n_jobs, probes=probes),
            lower=True,
        )
        # The stable form: a small shift makes the core definite to rounding,
        # and is taken off the eigenvalues after.
        shift = np.sqrt(dim) * np.finfo(float).eps * np.linalg.norm(sketch, 2)
        sketch += shift * white
        core = white.T @ sketch
        factor = linalg.cholesky(0.5 * (core + core.T), lower=True)
        basis, singular, _ = np.linalg.svd(
            linalg.solve_triangular(factor, sketch.T, lower=True).T,
            full_matrices=False,
        )
        half = lower @ (basis * np.sqrt(np.maximum(singular**2 - shift, 0.0)))
        return half @ half.T

    def short_noise(
        self,
        split_degree,
        cutoff_km,
        n_jobs,
        grace_split=True,
        exact_grace=False,
        grace_signal=None,
        vecchia=None,
    ):
        """
        The local noise with the short-wavelength part of the signal added
        to it, as a measure given by its covariance and its inverse.

        Each field's prior spectrum is split by `low_pass` at
        `split_degree`; what is not long-wavelength is the short part (all
        of it if `split_degree` is None). For the altimetry it is the
        covariance of eta's short part between the points, tapered beyond
        `cutoff_km`: a sparse matrix. For GRACE it is the covariance of the
        coefficients under the short parts of the loads with WMB's physics:
        a dense matrix of the size of GRACE's data; without `grace_split`
        it takes the whole of each spectrum. With `exact_grace` it is the
        covariance of the coefficients under the whole prior with the true
        physics. With `grace_signal`, a pair of a key and a matrix, it is
        that matrix (`Problem.grace_signal`). With `vecchia`, a number of
        neighbours, the altimetry's part is not tapered: its inverse is the
        sparse one of `vecchia.VecchiaInverse`, with no cut-off.
        """
        key = (
            split_degree,
            cutoff_km,
            grace_split,
            exact_grace,
            None if grace_signal is None else grace_signal[0],
            vecchia,
        )
        if key in self._short:
            return self._short[key]
        physics, args = self.physics, self.args
        setup = physics.setup
        by_name = common._stds_by_name(self.calib["derived_stds"])
        spectra = {
            name: common._field_measure(
                physics, args, name, by_name[name]
            ).covariance.eigenvalues
            for name in setup.names
        }
        measures = []
        if self.case != "grace":
            space = physics.field_space
            points = physics.points
            cutoff = cutoff_km * 1000.0 / physics.state.model.parameters.length_scale
            high = space.invariant_measure(
                spectra[common.ETA] * (1.0 - low_pass(space, split_degree)), unsafe=True
            )
            count = len(points)
            if vecchia is None:
                rows, columns, distances = space.pairs_within_distance(
                    points, cutoff, with_distances=True
                )
                values = space.covariance_function(high, distances) * gaspari_cohn(
                    distances / (0.5 * cutoff)
                )
                matrix = sparse.csc_matrix(
                    (values, (rows, columns)), shape=(count, count)
                )
                matrix = matrix + sparse.diags(self.alt_local.covariance.diagonals()[0])
                print(
                    f"    altimetry: {matrix.nnz / count:.0f} entries per point "
                    f"within {cutoff_km:.0f} km",
                    flush=True,
                )
                solve = SparseSolve(matrix)
            else:
                # No taper and no cut-off: each point regressed on its
                # nearest neighbours among those before it (`vecchia.py`).
                approximate = VecchiaInverse(
                    unit_vectors(points),
                    common.tabulated_kernel(space, high),
                    self.alt_local.covariance.diagonals()[0],
                    neighbours=vecchia,
                )
                print(
                    f"    altimetry: {approximate.entries_per_point:.0f} neighbours "
                    "per point, no cut-off",
                    flush=True,
                )
                solve = _Inverse(approximate)
            domain = self.alt_local.domain
            measures.append(
                inf.GaussianMeasure(
                    domain,
                    covariance=definite(
                        inf.LinearOperator.from_callables(
                            domain, domain, solve.apply, adjoint=solve.apply
                        )
                    ),
                    precision=definite(
                        inf.LinearOperator.from_callables(
                            domain, domain, solve.solve, adjoint=solve.solve
                        )
                    ),
                )
            )
        if self.case != "altimetry":
            wmb = common.wmb_method(physics.state, args)
            forward = (
                wmb.load_to_potential_coefficient_operator(physics.load_space)
                @ setup.load_operator
            )
            short = inf.BlockDiagonalLinearOperator(
                [
                    setup.component(name).space.spectral_operator(
                        spectra[name]
                        * (
                            1.0
                            - low_pass(
                                setup.component(name).space,
                                split_degree if grace_split else None,
                            )
                        )
                    )
                    for name in setup.names
                ],
                domain=setup.space,
                codomain=setup.space,
            )
            signal = forward @ short @ forward.adjoint
            if exact_grace:
                grace = physics.grace(args)
                signal = grace @ self.prior.covariance @ grace.adjoint
            if grace_signal is not None:
                matrix = grace_signal[1].copy()
            else:
                matrix = signal.matrix(form="components", by="rows", n_jobs=n_jobs)
                matrix = 0.5 * (matrix + matrix.T)
            self.add_grace_noise(matrix)
            domain = self.grace_noise.domain
            measures.append(
                inf.GaussianMeasure(
                    domain,
                    covariance=definite(
                        inf.LinearOperator.from_matrix(
                            domain, domain, matrix, form="components"
                        )
                    ),
                    precision=definite(
                        inf.LinearOperator.from_matrix(
                            domain, domain, np.linalg.inv(matrix), form="components"
                        )
                    ),
                )
            )
        if len(measures) == 1:
            result = measures[0]
        else:
            result = inf.GaussianMeasure.from_product(measures, domain=self.space)
        self._short[key] = result
        return result

    def fingerprint_factors(self, surrogate):
        """Low-rank factors of what the mass sources add to the altimetry
        through the sea level equation: the surrogate's altimetry operator
        on each source, times the square root of that source's prior."""
        factors = []
        by_name = common._stds_by_name(self.calib["derived_stds"])
        for name in surrogate.sources:
            space = surrogate.setup.component(name).space
            spectrum = common._field_measure(
                self.physics, self.args, name, by_name[name]
            ).covariance.eigenvalues
            root = space.invariant_measure(spectrum[: space.dim]).covariance_factor
            factor = surrogate.altimetry @ surrogate.setup.space.inclusion(name) @ root
            if self.case == "joint":
                factor = self.space.inclusion(0) @ factor
            factors.append(factor)
        return factors

    def local_noise(self, surrogate_degree=None):
        """The noise the surrogate sees: the local altimetry noise and
        GRACE's, each with the variance of the signal above
        `surrogate_degree` added if that is given."""
        args, physics = self.args, self.physics
        stds = self.calib["derived_stds"]
        alt, grace = self.alt_local, self.grace_noise
        if surrogate_degree is not None:
            alt = common.with_added_variance(
                alt,
                common.unresolved_altimetry_variance(
                    physics, args, stds["dyn_std"], surrogate_degree
                ),
            )
            grace = self.grace_noise_with(
                common.unresolved_grace_variance(physics, args, stds, surrogate_degree)
            )
        if self.case == "joint":
            return inf.GaussianMeasure.from_product([alt, grace], domain=self.space)
        return alt if self.case == "altimetry" else grace


class Surrogates:
    """The surrogate problems, built once for each set of options."""

    def __init__(self, physics, calib, args):
        self.physics, self.calib, self.args = physics, calib, args
        self._cache = {}

    def low_pass_prior(self, surrogate, split_degree, only=None):
        """The surrogate's prior for the two-level preconditioner: each
        field has the long-wavelength part of the true prior's spectrum,
        but for the density, which is kept whole. With `only`, a list of
        names, the other fields are kept whole too."""
        by_name = common._stds_by_name(self.calib["derived_stds"])
        measures = []
        for name in surrogate.setup.names:
            space = surrogate.setup.component(name).space
            reference = self.physics.setup.component(name).space
            spectrum = common._field_measure(
                self.physics, self.args, name, by_name[name]
            ).covariance.eigenvalues
            if name != common.RHO and (only is None or name in only):
                spectrum = spectrum * low_pass(reference, split_degree)
            measures.append(space.invariant_measure(spectrum[: space.dim]))
        return inf.GaussianMeasure.from_product(measures, domain=surrogate.setup.space)

    def get(self, degree, loads, iterations, matched=False):
        key = (degree, loads, iterations, matched)
        if key not in self._cache:
            args = copy.copy(self.args)
            args.surrogate_degree = degree
            surrogate = common.ocean_physics(
                args,
                points=self.physics.points,
                surrogate=True,
                surrogate_loads=loads,
                surrogate_iterations=iterations,
            )
            prior = common.product_prior(
                surrogate,
                args,
                self.calib["derived_stds"],
                reference=self.physics if matched else None,
            )
            self._cache[key] = (surrogate, prior)
        return self._cache[key]

    def forward(self, case, surrogate):
        altimetry, grace = surrogate.altimetry, surrogate.grace(self.args)
        if case == "joint":
            return inf.ColumnLinearOperator([altimetry, grace])
        return altimetry if case == "altimetry" else grace


def specification(name: str, args: argparse.Namespace) -> dict:
    """The options a variant's name stands for."""
    spec = {
        "kind": "surrogate",
        "degree": args.surrogate_degree,
        "alpha": 0.1,
        "loads": "sources",
        "iterations": 1,
        "unresolved": False,
        "low_rank": False,
        "matched": False,
    }
    # A suffix asks for the altimetry block to be the nearest-neighbour
    # sparse inverse, in place of the tapered covariance.
    neighbours = None
    if "+vecchia-" in name:
        name, count = name.rsplit("+vecchia-", 1)
        neighbours = int(count)
    spec["vecchia"] = neighbours
    # A prefix asks for the surrogate's prior to be the true one, truncated.
    if name.startswith("matched"):
        spec["matched"] = True
        name = name[len("matched") :].lstrip("-") or "base"
    if name in ("none", "noise", "wmb"):
        spec["kind"] = name
    elif name in ("local", "local-low-rank", "local-fingerprint"):
        spec |= {
            "kind": "local",
            "low_rank": name != "local",
            "fingerprint": name == "local-fingerprint",
        }
    elif name in ("exact", "exact-low-rank", "exact-fingerprint"):
        spec |= {
            "kind": "local",
            "exact": True,
            "low_rank": name != "exact",
            "fingerprint": name == "exact-fingerprint",
        }
    elif name.startswith("sketch-"):
        # sketch-<k> or sketch-<k>-<block>: k applications of the block's
        # physics (the exact one if none is named).
        _, rank, *inner = name.split("-", 2)
        inner = specification(inner[0] if inner else "exact-block", args)
        spec |= {
            "kind": "local",
            "low_rank": True,
            "fingerprint": False,
            "mode": ("sketch", int(rank), inner["mode"]),
        }
    elif name in ("wmb-block", "exact-block") or name.startswith(
        ("model-", "reduced-", "wmb-sle-", "wmb-sl-")
    ):
        stem, flags = name, set()
        while stem.rsplit("-", 1)[-1] in ("taper", "smooth", "fields"):
            stem, flag = stem.rsplit("-", 1)
            flags.add(flag)
        if stem == "wmb-block":
            mode = ("wmb",)
        elif stem == "exact-block":
            mode = ("exact",)
        else:
            kind, degree = stem.rsplit("-", 1)
            mode = (kind, int(degree), frozenset(flags))
        spec |= {"kind": "local", "low_rank": True, "fingerprint": False, "mode": mode}
    elif name in ("hybrid", "hybrid-low-rank"):
        spec |= {"kind": "hybrid", "low_rank": name != "hybrid", "alpha": 0.0}
    elif name in ("two-level", "two-level-low-rank"):
        spec |= {"kind": "two-level", "low_rank": name != "two-level", "alpha": 0.0}
    elif name in ("cheap", "cheap-low-rank", "cheap-fingerprint"):
        spec |= {
            "kind": "cheap",
            "low_rank": name != "cheap",
            "fingerprint": name == "cheap-fingerprint",
            "matched": True,
        }
    elif name.startswith("spectral-"):
        spec |= {"kind": "spectral", "rank": int(name.split("-", 1)[1])}
    elif name == "base":
        pass
    elif name.startswith("alpha-"):
        spec["alpha"] = float(name.split("-", 1)[1])
    elif name == "degree-half":
        spec["degree"] = args.surrogate_degree // 2
    elif name.startswith("degree-"):
        spec["degree"] = int(name.split("-", 1)[1])
    elif name == "all-loads":
        spec["loads"] = "all"
    elif name == "full-sle":
        spec["iterations"] = None
    elif name in ("unresolved", "unresolved-alpha-0"):
        spec["unresolved"] = True
    elif name == "low-rank":
        spec["low_rank"] = True
    elif name in ("both", "both-alpha-0"):
        spec |= {"unresolved": True, "low_rank": True}
    else:
        raise ValueError(f"Unknown variant '{name}'.")
    if name.endswith("-alpha-0"):
        spec["alpha"] = 0.0
    return spec


# Options that leave the signal part of a GRACE block unchanged.
_NOT_OF_THE_BLOCK = (
    "variants",
    "cases",
    "recycle",
    "seed_directions",
    "seed_power",
    "max_iterations",
    "serial_threads",
    "block_cache",
    "spacing",
    "cutoff_km",
    "seed",
)


def grace_block(mode, problem: Problem, surrogates: Surrogates, args) -> np.ndarray:
    """`Problem.grace_signal`, kept on disk with --block-cache for the other
    variants and cases that use the same block. It does not depend on the
    case."""

    def label(mode):
        return "-".join(str(part) for part in mode[:2]) + "".join(
            f"-{flag}" for flag in sorted(mode[2] if len(mode) > 2 else ())
        )

    sketched = mode[0] == "sketch"
    name = f"sketch-{mode[1]}-{label(mode[2])}" if sketched else label(mode)
    path = None
    if args.block_cache:
        options = sorted(
            (key, repr(value))
            for key, value in vars(args).items()
            if key not in _NOT_OF_THE_BLOCK
        )
        digest = hashlib.sha1(repr(options).encode()).hexdigest()[:12]
        path = os.path.join(args.block_cache, f"{name}-{digest}.npy")
        if os.path.exists(path):
            print(f"    GRACE block read from {path}", flush=True)
            return np.load(path)
    start = time.perf_counter()
    if sketched:
        matrix = problem.sketched_signal(
            mode[2], mode[1], surrogates, common.jobs(args)
        )
    else:
        matrix = problem.grace_signal(mode, surrogates, common.jobs(args))
    print(
        f"    GRACE block {name} built in {time.perf_counter() - start:.0f} s",
        flush=True,
    )
    if path is not None:
        os.makedirs(args.block_cache, exist_ok=True)
        np.save(path, matrix)
    return matrix


def build(spec: dict, problem: Problem, surrogates: Surrogates, args, rng):
    """The preconditioner of a specification, or None for none."""
    if spec["kind"] == "none":
        return None
    if spec["kind"] == "noise":
        return problem.local_noise().precision
    if spec["kind"] == "wmb":
        # Every degree counts as unresolved: the signal's variance is added
        # to the noise throughout, and nothing else is used.
        return problem.local_noise(-1).precision
    if spec["kind"] == "hybrid":
        jobs = common.jobs(args)
        if problem.case == "grace":
            preconditioner = problem.short_noise(None, args.cutoff_km, jobs).precision
        else:
            surrogate, _ = surrogates.get(spec["degree"], "all", 1, True)
            forward = surrogate.altimetry
            if problem.case == "joint":
                forward = inf.ColumnLinearOperator(
                    [
                        forward,
                        inf.LinearOperator.zero(
                            forward.domain, codomain=problem.grace_noise.domain
                        ),
                    ]
                )
            preconditioner = common.surrogate_preconditioner(
                problem.normal,
                forward,
                surrogates.low_pass_prior(surrogate, spec["degree"], only=[common.ETA]),
                problem.short_noise(
                    spec["degree"], args.cutoff_km, jobs, grace_split=False
                ),
                max_jobs=jobs,
                alpha=0.0,
            )
        if spec["low_rank"] and problem.factors:
            preconditioner = common.low_rank_correction(preconditioner, problem.factors)
        return preconditioner
    if spec["kind"] == "two-level":
        surrogate, _ = surrogates.get(spec["degree"], "all", 1, True)
        noise = problem.short_noise(spec["degree"], args.cutoff_km, common.jobs(args))
        preconditioner = common.surrogate_preconditioner(
            problem.normal,
            surrogates.forward(problem.case, surrogate),
            surrogates.low_pass_prior(surrogate, spec["degree"]),
            noise,
            max_jobs=common.jobs(args),
            alpha=spec["alpha"],
        )
        if spec["low_rank"] and problem.factors:
            preconditioner = common.low_rank_correction(preconditioner, problem.factors)
        return preconditioner
    if spec["kind"] in ("cheap", "local"):
        if spec["kind"] == "local":
            signal = None
            if spec.get("mode") is not None and problem.case != "altimetry":
                signal = (
                    spec["mode"],
                    grace_block(spec["mode"], problem, surrogates, args),
                )
            preconditioner = problem.short_noise(
                None,
                args.cutoff_km,
                common.jobs(args),
                exact_grace=spec.get("exact", False),
                grace_signal=signal,
                vecchia=spec.get("vecchia"),
            ).precision
        else:
            preconditioner = problem.cheap_inverse(common.jobs(args))
        factors = list(problem.factors) if spec["low_rank"] else []
        if spec["fingerprint"] and problem.case != "grace":
            surrogate, _ = surrogates.get(spec["degree"], "sources", 1, True)
            factors += problem.fingerprint_factors(surrogate)
        if factors:
            preconditioner = common.low_rank_correction(preconditioner, factors)
        return preconditioner
    if spec["kind"] == "spectral":
        return inf.SpectralPreconditioner(
            rank=spec["rank"], rng=rng, n_jobs=common.jobs(args)
        )(problem.normal)
    surrogate, prior = surrogates.get(
        spec["degree"], spec["loads"], spec["iterations"], spec["matched"]
    )
    noise = problem.local_noise(spec["degree"] if spec["unresolved"] else None)
    preconditioner = common.surrogate_preconditioner(
        problem.normal,
        surrogates.forward(problem.case, surrogate),
        prior,
        noise,
        max_jobs=common.jobs(args),
        alpha=spec["alpha"],
    )
    if spec["low_rank"] and problem.factors:
        preconditioner = common.low_rank_correction(preconditioner, problem.factors)
    return preconditioner


def solve(problem: Problem, preconditioner, max_iterations: int):
    """Iterations to each tolerance, and the time of the solve."""
    progress = inf.ProgressCallback()
    solver = inf.CGSolver(
        rtol=TOLERANCES[-1],
        preconditioner=preconditioner,
        callback=progress,
        max_iterations=max_iterations,
        strict=False,
    )
    threads = problem.args.serial_threads
    if threads is not None:
        set_num_threads(threads)
    start = time.perf_counter()
    solver(problem.normal)(problem.data)
    elapsed = time.perf_counter() - start
    if threads is not None and problem.args.parallel:
        set_num_threads(1)
    residuals = np.asarray(progress.residuals, dtype=float)
    relative = residuals / residuals[0]
    counts = []
    for tolerance in TOLERANCES:
        reached = np.flatnonzero(relative <= tolerance)
        counts.append(int(reached[0]) if reached.size else None)
    return counts, elapsed


def seeded_solve(problem: Problem, preconditioner, args, rng) -> str:
    """The iterations of the data's solve with the preconditioner corrected
    by --seed-directions random directions (`RecycledSolver.seed`)."""
    solver = recycling.RecycledSolver(problem.normal, preconditioner)
    start = time.perf_counter()
    solver.seed(args.seed_directions, power=args.seed_power, rng=rng)
    built, used = time.perf_counter() - start, solver.applications
    _, residuals = solver.solve(
        problem.data,
        rtol=TOLERANCES[-1],
        max_iterations=args.max_iterations,
        recycle=False,
    )
    return (
        f"    seeded with {solver.rank} directions ({used} applications, "
        f"{built:.0f} s): {recycling.iterations_to(residuals, TOLERANCES)}\n"
    )


def recycled_solves(problem: Problem, preconditioner, args, rng) -> str:
    """The iterations of a sequence of solves under one preconditioner,
    each reusing the search directions of those before it, against the same
    solves without reuse. The first is for the data."""
    reused = recycling.RecycledSolver(problem.normal, preconditioner)
    plain = recycling.RecycledSolver(problem.normal, preconditioner)
    sides = [("data", problem.data)] + problem.right_hand_sides(args.recycle, rng)
    lines = ""
    for index, (label, side) in enumerate(sides):
        kept = reused.rank
        _, residuals = reused.solve(
            side, rtol=TOLERANCES[-1], max_iterations=args.max_iterations
        )
        with_reuse = recycling.iterations_to(residuals, TOLERANCES)
        without = with_reuse
        if index > 0:
            _, residuals = plain.solve(
                side,
                rtol=TOLERANCES[-1],
                max_iterations=args.max_iterations,
                recycle=False,
            )
            without = recycling.iterations_to(residuals, TOLERANCES)
        lines += (
            f"    solve {index + 1:<3}{label:<18}kept {kept:<5}"
            f"reused {str(with_reuse):<22}not reused {without}\n"
        )
    return lines


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)

    print(f"Building the physics and the prior (lmax={args.lmax})...")
    physics = common.ocean_physics(args)
    print(f"  {len(physics.points)} altimetry points")
    prior, calib = common.ocean_prior(physics, args, rng)
    common.print_calibration_report(calib, physics)
    surrogates = Surrogates(physics, calib, args)

    header = (
        f"{'preconditioner':<22}{'build (s)':>11}"
        + "".join(f"{f'its to {t:.0e}':>14}" for t in TOLERANCES)
        + f"{'solve (s)':>11}\n"
    )
    for case in args.cases:
        print(f"\nCase: {case}")
        problem = Problem(case, physics, prior, calib, args, rng)
        ranks = [factor.domain.dim for factor in problem.factors]
        text = (
            f"\n\nCase: {case}; {problem.space.dim} data; low-rank noise parts "
            f"of rank {ranks if ranks else 'none'}\n"
            + "=" * len(header)
            + "\n"
            + header
        )
        text += "-" * len(header) + "\n"
        for name in args.variants:
            spec = specification(name, args)
            if spec["low_rank"] and not problem.factors and name == "low-rank":
                continue
            start = time.perf_counter()
            preconditioner = build(spec, problem, surrogates, args, rng)
            if preconditioner is not None:
                # Factorisations are made on first use: make them now.
                preconditioner(problem.data)
            built = time.perf_counter() - start
            counts, elapsed = solve(problem, preconditioner, args.max_iterations)
            row = f"{name:<22}{built:>11.1f}" + "".join(
                f"{'--' if c is None else c:>14}" for c in counts
            )
            row += f"{elapsed:>11.1f}\n"
            print("  " + row, end="", flush=True)
            text += row
            if args.seed_directions > 0:
                lines = seeded_solve(problem, preconditioner, args, rng)
                print(lines, end="", flush=True)
                text += lines
            if args.recycle > 0:
                lines = recycled_solves(problem, preconditioner, args, rng)
                print(lines, end="", flush=True)
                text += lines
        common.write_metrics(METRICS_FILE, text)
    print(f"\nsaved {METRICS_FILE}")


if __name__ == "__main__":
    main()
