# Porting pyslfp to pygeoinf 2

This note maps every pygeoinf 1 name pyslfp uses to its pygeoinf 2
equivalent. It also lists the changes in behaviour that a rename does not
cover, and the gaps to be fixed in pygeoinf rather than worked around here.

The port is released as pyslfp 3.1.0. The minimum versions become
`pygeoinf>=2.1` and `planetmodel>=1.4.1`. The release notes should say that
the spaces, operators and measures pyslfp returns are pygeoinf 2 objects.

The v1 names were taken from the `v1-final` tag of pygeoinf and the v2
names from `main` at 2.1.0. Items marked *verified* were run against v2.
The area columns count uses in:

- **lib**: `pyslfp/`;
- **tst**: `tests/`;
- **tut**: `tutorials/scripts/`;
- **H26**: `Heathcote2026/`.

## 1. Name map

### Spaces

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `sphere.Lebesgue(lmax, radius=, grid=)` | `sphere.Lebesgue(lmax, radius=, grid=, sampling=, extend=)` | lib | See §2.1: the grid options must be passed. |
| `sphere.Sobolev(lmax, order, scale, ...)` | `sphere.Sobolev(lmax, order, length_scale, ...)` | lib | Positional order unchanged. |
| `space.scale` | `space.length_scale` | lib, H26 | |
| `MassWeightedHilbertSpace` | `MassWeightedSpace` | lib | No longer needed; see §2.2. |
| `space.underlying_space` | `space.with_order(0)` | lib | Sobolev is not mass-weighted in v2; see §2.2. |
| `space.spatial_multiplication_operator(f)` | `space.multiplication_operator(f)` | lib | Correct adjoint on Sobolev (*verified*). |
| `space.l2_products_operator(ws)` | unchanged | lib, tut, H26 | |
| `space.point_evaluation_operator(points, matrix_free=)` | `space.point_evaluation_operator(points, route=)` | lib | `route` is `"auto"`, `"dense"` or `"local"`. Refused at order ≤ 1 unless `unsafe=True`. |
| `space.to_coefficient_operator(lmax, lmin=)` | `space.coefficient_operator(lmax=, lmin=)` | lib, H26 | Keyword-only. Scaled by the radius; see §2.3. |
| `space.from_coefficient_operator(lmax, lmin=)` | `space.from_coefficient_operator(lmax=, lmin=)` | lib | As above. |
| `space.order_inclusion_operator(order)` | `space.order_inclusion_operator(target)` | lib | Takes the target space, not an order. |
| `space.identity_operator()` | `LinearOperator.identity(space)` | lib, H26 | |
| `space.zero_operator(codomain=)` | `LinearOperator.zero(space, codomain=)` | lib, H26 | |
| `space.vector_multiply(a, b)` | `space.multiply(a, b)` | H26 | |
| `space.vector_sqrt(a)` | `space.sqrt(a)` | H26 | |
| `space.dirac_representation(p)` | `space.dirac(p).representer` | H26 | `dirac` returns the functional. |
| `space.zero` (a property) | `space.zero()` (a method) | tst, H26 | The property form passes the bound method on silently. |

### Direct sums

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `HilbertSpaceDirectSum(spaces)` | `DirectSum(spaces, labels=)` | lib, tst, H26 | |
| `.subspaces`, `.subspace(i)` | unchanged | lib | `subspace` also takes a label. |
| `.number_of_subspaces` | `len(space.subspaces)` | lib | |
| `.subspace_projection(i)` | `.projection(i)` | lib, tut, H26 | |
| vectors as lists | vectors as tuples | lib, tst, H26 | |

### Operators

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `LinearOperator(dom, cod, mapping, adjoint_mapping=)` | `LinearOperator.from_callables(dom, cod, value, adjoint=)` | lib | |
| subclass calling `super().__init__(dom, cod, op, adjoint_mapping=)` | `super().__init__(dom, cod)` and override `_value`, `_adjoint_value` | lib | `FingerPrintOperator`. |
| `LinearOperator.from_formal_adjoint(dom, cod, op)` | unchanged | lib | Recurses through direct sums with Euclidean summands (*verified*, adjoint defect 1e-13). |
| `LinearOperator.from_matrix(dom, cod, M)` | `LinearOperator.from_matrix(dom, cod, M, form=)` | lib | `form` is required: `"components"` here. |
| `RowLinearOperator`, `ColumnLinearOperator`, `BlockDiagonalLinearOperator`, `BlockLinearOperator` | unchanged | lib, H26 | |
| `DiagonalSparseMatrixLinearOperator.from_diagonal_values(dom, cod, v)` | `DiagonalLinearOperator(dom, v)` | lib, tst | Endomorphisms only, which is every pyslfp use. |
| `op.extract_diagonal(parallel=, n_jobs=)` | `op.diagonals(n_jobs=)[0]`; `.eigenvalues` on a diagonal operator | lib | |
| `op.matrix(dense=True)` | `op.matrix(form=)` | tut, H26 | Always dense. `form="auto"` gives the Galerkin form for a self-adjoint operator; say `"components"` where that is meant. |
| `op.check(n_checks=, check_rtol=, domain_measure=, codomain_measure=)` | `pygeoinf.testing.check_operator(op, rng=, measure=, trials=, rtol=)` | tst, tut | Returns a `CheckReport`. There is no codomain measure; see §3. |

### Solvers

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `CholeskySolver(galerkin=True)` | `CholeskySolver()` | lib, tut, H26 | The form is fixed by the class. Requires `POSITIVE_DEFINITE`; see §2.4. |
| `LUSolver(galerkin=True, parallel=, n_jobs=)` | `LUSolver(n_jobs=)` | H26 | |
| `CGSolver(rtol=)` | `CGSolver(rtol=, preconditioner=, callback=, max_iterations=, strict=)` | tut, H26 | The preconditioner goes on the solver. |
| `solver.iterations` | `SolveResult.iterations`, or a callback | tut, H26 | Solvers keep no state; see §2.5. |
| `ProgressCallback()` | `ProgressCallback(report=print)` | H26 | Prints only when given `report`; `.iterations` counts the last solve. |

### Measures

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `GaussianMeasure.from_standard_deviation(s)` | unchanged | lib, tst, tut, H26 | |
| `GaussianMeasure.from_direct_sum(ms)` | `GaussianMeasure.from_product(ms, labels=)` | lib, tst, H26 | |
| `InvariantGaussianMeasure` | no class: `space.invariant_measure(...)` returns a `GaussianMeasure` with a diagonal covariance | lib, tst | The GRACE type check goes; see §3. |
| `measure.spectral_variances` | none public | lib | GRACE uses a bespoke operator instead; see §2.6 and §3. |
| `CorrelatedInvariantGaussianMeasure.from_invariant_measures(ms, c)` | `space.correlated_measure_from_correlations([m.covariance.eigenvalues for m in ms], R)` | H26 | `R` stacks `c(k)` over `space.laplacian_eigenvalues`, the argument v1 passed. |
| `space.invariant_gaussian_measure(f)` | `space.invariant_measure(f)` | H26 | `f` acts on the Laplacian eigenvalues, as before. |
| `space.heat_kernel_gaussian_measure(s)` | `space.heat_measure(s)` | lib, tst, H26 | Identical spectrum (*verified*). |
| `space.point_value_scaled_heat_kernel_gaussian_measure(s, std=)` | `space.heat_measure(s, pointwise_std=)` | H26 | Identical (*verified*). |
| `space.point_value_scaled_sobolev_kernel_gaussian_measure(o, s, std=)` | `space.sobolev_measure(o, s, pointwise_std=)` | tut, H26 | Identical on a Sobolev space (*verified*). |
| `measure.affine_mapping(operator=, translation=)` | `measure.affine_map(operator, translation=)` | tut, H26 | |
| `measure.inverse_covariance` | `measure.precision` | H26 | |
| `measure.with_dense_covariance(parallel=, n_jobs=)` | `measure.with_dense_covariance(n_jobs=)` | tut, H26 | |
| `measure.sample()`, `.samples(n, parallel=, n_jobs=)` | `measure.sample(rng=)`, `.samples(n, rng=, n_jobs=)` | tst, H26 | See §2.7. |

### Inference and geometry

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `LinearForwardProblem(op, data_error_measure=)` | `LinearForwardProblem(op, error=)` | tut, H26 | |
| `problem.data_error_measure` | `problem.error_measure` | H26 | |
| `LinearBayesianInversion(problem, prior)` | `LinearGaussianInversion(problem, prior, solver=)` | tut, H26 | |
| `inversion.model_posterior_measure(data, solver, preconditioner=)` | `inversion.with_solver(solver)(data)` | tut, H26 | Returns a `GaussianPosterior`. |
| `inversion.normal_operator` | unchanged | tut | |
| `inversion.surrogate_woodbury_data_preconditioner(solver, alternate_...)` | `WoodburyPreconditioner(A_s, Q_s.covariance, R_s.covariance, solver=, prior_inverse=Q_s.precision, noise_inverse=R_s.precision)(inversion.normal_operator)` | H26 | The surrogate's pieces; applied to the true normal operator it gives the data form. |
| `inversion.kalman_operator(solver, preconditioner=)` | `inversion.with_solver(solver).kalman_operator` | H26 | A property; the preconditioner goes on the solver. |
| `AffineSubspace.from_linear_equation(op, value, solver=)` | unchanged | H26 | |
| `LinearSubspace.from_kernel(op)` | unchanged | H26 | |
| `subspace.condition_gaussian_measure(prior)` | `prior.condition(op.with_traits(Traits.SURJECTIVE), value)` | H26 | Exact conditioning needs a positive-definite prior; see §3. |

### Plotting

| v1 | v2 | Used in | Notes |
|---|---|---|---|
| `sphere.create_map_figure(figsize=, projection=)` | `pygeoinf.plotting.subplots(space, projection=, figsize=)` | lib | |
| `sphere.plot(f, ax=, projection=, ...)` | `pygeoinf.plot(space, f, ax=, ...)` | lib | See §2.8. |
| `sphere.plot_points(points, data=, ...)` | `pygeoinf.plot_points(space, points, ...)` | lib | |
| `plot_corner_distributions(post, prior_measure=, true_values=, labels=, title=)` | `plot_corner(post, prior=, truth=, labels=, title=)` | tut, H26 | |
| `plot_1d_distributions(...)` | `plot_densities(...)` | H26 | |

Unchanged and used: `HilbertSpace` (type hints only), `EuclideanSpace`,
`LinearOperator`, `.adjoint`, `@` and scalar `*`, `.domain`, `.codomain`,
`.dim`, `GaussianMeasure.from_standard_deviations`, `.covariance`,
`.expectation`, `problem.synthetic_model_and_data`, `problem.chi_squared`.

## 2. Changes in behaviour

1. **Grid options.** v2 spheres default to `extend=False` and refuse
   `grid="DH2"`, which is written `grid="DH", sampling=2`. pyslfp's grids
   default to `extend=True`, and a default v2 space refuses them
   (*verified*). The space factories in `linear_operators/physics.py` must
   pass `grid`, `sampling` and `extend` from the `EarthModel`. The v1
   factories passed only `grid`, so a model built with `extend=False` would
   not have matched its spaces there either.

2. **Sobolev spaces are not mass-weighted.** In v1 a Sobolev space wrapped
   a Lebesgue space, which `underlying_space` returned. In v2 both are
   diagonal-metric spaces over the same components, and
   `space.with_order(0)` gives the Lebesgue space with the same grid
   options (*verified*). The pattern used throughout `linear_operators`
   stays as it is:
   - write the action and the adjoint on the Lebesgue spaces;
   - lift them with `from_formal_adjoint`.

   `underlying_space` becomes `with_order(0)`, applied to each summand of
   a direct sum.

3. **Coefficient scaling.** Both versions expand in pyshtools'
   orthonormal harmonics Y_lm, normalised so that ∫ Y_lm² dΩ = 1 over the
   unit sphere, without the Condon-Shortley phase. Both use the same
   ordering: by degree, then m = 0, 1, …, l, −1, …, −l. They differ in
   which basis the components refer to:
   - **v1:** the components are the coefficients c_lm of u = Σ c_lm Y_lm.
     This is also pyslfp's own convention (`EarthModel` uses `"ortho"`,
     `csphase=1`). On a sphere of radius b the basis is not orthonormal:
     ∫ Y_lm² dS = b², so the Lebesgue Gram matrix is b² I.
   - **v2:** the components refer to Y_lm / b, which is orthonormal on the
     sphere of radius b. The component is therefore b c_lm, and the
     Lebesgue Gram matrix is I.

   The fields and their inner products are the same in both versions;
   only the component vector changes (*verified*). With pyslfp's default
   parameters, b = 0.99952911630827, so the change is a relative 4.7e-4.
   That is small enough to pass a loose test unnoticed. In pyslfp only the
   GRACE code reads components, through `coefficient_operator`,
   `from_coefficient_operator` and the prior's spectral variances.
   Elsewhere everything is free of coordinates and unaffected.

   The GRACE operators keep the v1 output: coefficients that are
   orthonormal on the unit sphere. This keeps the output definite and
   independent of the space's internal choices. The port therefore divides
   `coefficient_operator` by b explicitly and multiplies
   `from_coefficient_operator` by b. The factor is never dropped because b
   is close to 1. An option for output in geodetic conventions is in §5.

4. **Cholesky needs a claim of definiteness.** `CholeskySolver` requires
   `POSITIVE_DEFINITE`. A composition `B @ B.adjoint` is given only
   `SELF_ADJOINT | POSITIVE_SEMIDEFINITE`, so it is refused (*verified*).
   This affects `ice_sheet_basis_operator`. The operator is definite there
   because the basin masks are linearly independent. The claim is added
   with `with_traits(Traits.POSITIVE_DEFINITE)` and checked in a test with
   `testing.check_traits`.

5. **Solvers keep no state.** `solver.iterations` no longer exists. An
   iteration count comes from the `SolveResult` returned by `solve`, or
   from a `callback`, which is the route when the solve happens inside an
   inversion. Tutorial 05 prints the iteration counts and must count them
   with a callback.

6. **Spectral variances.** Both versions describe an invariant measure by
   the eigenvalues σ_k of its covariance, relative to the space's inner
   product, in which ‖Y_k‖² = g_k. A coefficient's variance is σ_k / g_k
   (*verified* in v1 and v2). v1's `load_measure_to_observation_measure`
   took σ_k as the variance. That overstates it by g_k:
   - by b² on a Lebesgue space;
   - by b² (1 + L² l(l+1)/b²)^s on a Sobolev space, which at the
     Heathcote2026 defaults (s = 2, L = 500 km) is 2.8 at degree 10, 56 at
     degree 32, 709 at degree 64 and 4000 at degree 100.

   `grace_inversion.py`, `grace_bias.py` and `joint_inversion.py` build
   their GRACE data-noise measure with this function, from a spatial noise
   measure on the Sobolev load space; pyslfp 2.0.3, pinned by the record
   repository, has the same code. The synthetic noise and the likelihood
   share that measure, so those inversions are consistent for the noise
   they used. That noise is g_k times the push-forward of the stated
   spatial noise, so the amplitude SNR in the data is the calibrated one
   divided by √g_k. Run on the corrected function, the scripts will have
   less noise at high degrees than the paper's runs.

   The port forms the observation measure from the spectrum, with
   variances k_l² σ_k / g_k. This agrees with the exact push-forward
   through the WMB operator to rounding (*verified*), and takes 2 ms at
   lmax 256 and observation degree 100. A measure that is not invariant
   is refused, as in v1. Above the load space's truncation the operator
   gives zero, but the observed degrees keep a variance, that of the
   highest resolved degree times k_l², so the measure stays usable as a
   data error measure.

7. **Random numbers.** Sampling takes `rng=` and otherwise draws a fresh
   generator seeded by the operating system. `np.random.seed` no longer
   makes a run repeatable. This affects five tests and tutorial 05.

8. **Plotting.** The v2 functions dispatch on the space, take axes made by
   `subplots(space, projection=)`, and return `(axes, mappable)`, as v1
   did. `pyslfp.plot` and `pyslfp.plot_points` keep their own signatures:
   - they build the `Sphere` from the field's own `lmax`, grid, sampling
     and `extend`;
   - they make Robinson axes when none are given.

   `num_levels` becomes `levels`. Whether `contour_lines_kwargs` has a v2
   equivalent is to be checked during the port.

9. **Point evaluation** on a space of order ≤ 1 is refused unless
   `unsafe=True`. `check_load_space(point_values=True)` already requires
   order > 1, so nothing changes in pyslfp.

10. **Priors are unchanged.** The Sobolev, heat and pointwise-scaled
    measures give the same spectrum in v1 and v2 on a Sobolev space, to all
    printed digits (*verified*). Prior parameters in tutorials and scripts
    carry over as they are.

## 3. Gaps in pygeoinf

1. **A codomain measure in `check_operator`.** v1's `check` drew the
   codomain probes from a measure. pyslfp's tests use smooth response
   measures, because the fingerprint operator's adjoint is accurate only to
   the solver tolerance. v2 probes the codomain with white noise; pygeoinf's
   `CODE_NOTES.md` already records this. It is wanted later, but does not
   block the port. In the meantime the operator checks run on the
   analytical test states (`EarthState.for_testing`), where the solves
   converge tightly.

2. **Exact conditioning of a semidefinite prior.** `GaussianMeasure.condition`
   requires A C A* to be known positive definite, and infers that only from
   a positive-definite C and a surjective A. A masked prior P C P* is
   semidefinite, so conditioning it is refused even where A C A* is
   positive definite, and there is no way to claim that. v1 allowed it.
   Heathcote2026 avoids it because its mass constraint carries the ocean
   function and so A P = A: conditioning the unmasked prior and masking
   afterwards gives the same measure. The models submodule will constrain
   masked components where that identity need not hold.

3. **A tensor product of spaces.** It is not needed for the port. It is
   needed later for model spaces that vary in time and for the viscoelastic
   operators on ice histories.

Pushing the load prior into coefficient space for GRACE
(`load_measure_to_observation_measure`,
`bayesian_normal_operator_preconditioner`) is not treated as a gap. The
GRACE module reads the spectrum off the measure's diagonal covariance and
scales it degree by degree; see §2.6.

## 4. Order of work

1. Raise the minimum versions in `pyproject.toml` and update the lock.
2. `linear_operators/utils.py` and the space factories in `physics.py`:
   grid options, `with_order(0)`, and the direct-sum and checking helpers.
3. The remaining operator modules, with `FingerPrintOperator` as a
   subclass of `LinearOperator` in the v2 form.
4. `grace.py`, with its bespoke prior operator and the decision in §2.3.
5. The tests, with explicit generators.
6. `plot.py`.
7. Tutorials 04 and 05, then regenerate their notebooks.
8. The Heathcote2026 scripts, later, against this map.

## 5. GRACE output as Stokes coefficients

The GRACE operators can return what Level-2 products and ICGEM files
publish: dimensionless Stokes coefficients ΔC̄_lm, ΔS̄_lm. Those products
write the geopotential as

    V(r, θ, λ) = (GM/a) Σ_l (a/r)^(l+1) Σ_m P̄_lm(cos θ) (C̄_lm cos mλ + S̄_lm sin mλ).

The default output is the coefficient Φ_lm of pyslfp's potential change on
the sphere of radius b, orthonormal on the unit sphere and in pyslfp's
units. It differs from ΔC̄_lm in four ways:

1. **Normalisation.** "Fully normalised" harmonics have mean square one
   over the sphere, pyshtools' `"4pi"`, against `"ortho"`'s
   ∫ Y_lm² dΩ = 1, so a coefficient divides by √(4π). ICGEM files take
   `norm fully_normalized` by default (ICGEM format, 2023), GRACE Level-2
   headers state "fully normalized", and pyshtools reads ICGEM files as
   `"4pi"`, without the Condon-Shortley phase. Schmidt semi-normalisation
   is the geomagnetic convention.
2. **Sign.** pyslfp's potential is negative near added mass; the geodetic
   V is positive, with C̄_00 = 1 giving V = GM/r.
3. **Units.** Stokes coefficients are dimensionless: divide by GM/a. The
   result does not depend on pyslfp's units.
4. **Radius.** The coefficients refer to the product's reference radius a,
   while pyslfp's potential is on r = b. Upward continuation gives a
   factor (b/a)^(l+1): about 0.91 at degree 60.

Together:

    ΔC̄_lm = −(a / GM) (b/a)^(l+1) Φ_lm / √(4π),

with ΔS̄_l|m| at the negative orders, so no reordering is needed.

`stokes_coefficient_operator` is this diagonal operator. `GraceObservationModel`
and `WMBMethod` take `convention="stokes"`, with `gravitational_parameter`
and `reference_radius` in SI. The defaults are GM = 3.986004415e14 m³ s⁻²
and a = 6378136.3 m, those of GGM05C, to which the CSR and JPL releases
refer; GFZ's EIGEN-based releases use a = 6378136.46 m. These are not yet
checked against a CSR or JPL file header, and a product's header gives
its own. In `WMBMethod` the
conversion is folded into the per-degree scaling, so its operators, its
observation measure and its preconditioner all follow. The default
convention is unchanged.

A test synthesises the geopotential with pyshtools from chosen Stokes
coefficients, evaluates it on r = b, and recovers them through the GRACE
operator. It fails if the sign is flipped or if GFZ's reference radius is
used against CSR's.
