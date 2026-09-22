# Moving the Love number solver into pyslfp

Written 15 September 2026. Decisions taken with David the same day are
marked as such. planetmodel 1.0.0 is installed in the pyslfp environment
and pinned in pyproject.toml.

## What exists

**planetmodel.loading** (three modules, about 920 lines, numpy and scipy
only) solves the loading and tidal problem degree by degree on a
`RadialMesh` by the reduced weak form of Al-Attar & Tromp (2014), for any
model holding density and the five moduli, real or complex:

| module | lines | contents |
| :-- | --: | :-- |
| `material.py` | 184 | `NodalModuli`, `nodal_moduli`, `Material`: a model read onto a mesh once (rho, drho, g, fluid flags, A C F L N, G, radius, surface gravity, omega) |
| `assembly.py` | 391 | `DegreeSystem`: the banded symmetric degree-l system, its dof map, load and tide right-hand sides, Cholesky or LU solve, `expand` to nodal arrays |
| `love.py` | 342 | `DegreeSolution` (U, V, phi of one degree, `evaluate(radii)`), `solve_degree`, `LoveNumbers` (nine numbers by degree with `Scales`, `conventional()`, `tidal()`, `reciprocity_residual()`, `converted()`, `write()`), `read_love_numbers`, `love_numbers` |

Everything it imports from planetmodel is public in 1.0.0, so the copied
code runs against the released wheel with its import lines changed.
Nothing in planetmodel outside `loading/` depends on it: tutorials 10 and
12, `tests/loading/test_love.py`, and a few README and docstring lines are
the whole coupling.

The tests (about 330 lines) carry two reference tables for PREM at degrees
1, 2, 3, 10, 20: the archived solver's values, matched to 1e-8, and the
Fortran table pyslfp ships, matched to 2 per cent; also the incompressible
homogeneous sphere closed forms, Maxwell limits of a frozen model, the
non-dimensional model equivalence, and the file round trip.

**pyslfp** consumes Love numbers through `core.LoveNumbers(lmax, params, /,
*, file=None)`: it reads the seven-column file (default
`~/.pyslfp_data/love_numbers/PREM_4096.dat`, fetched from Zenodo), slices
to lmax, non-dimensionalises by hand from `EarthModelParameters`, and
exposes `h_u, k_u, h_phi, k_phi, h, k, ht, kt` plus the Green's functions
and their plots. `EarthModel` holds one and `with_degree` rebuilds it. The
consumers are `physics.SeaLevelEquation` (all eight arrays, lines 137 to
153) and `linear_operators/grace.py` (`k` only, two places).

The two non-dimensionalisations agree. pyslfp scales h_u by
load_scale / length_scale, k_u by load_scale / potential_scale, ht by
potential_scale / length_scale; planetmodel's `Dimensions` for the same
columns are (mass -1, length 3), (mass -1, length 4, time -2) and
(length -1, time 2). So
`Scales(length=p.length_scale, mass=p.mass_scale, time=p.time_scale)`
fed to `LoveNumbers.converted` reproduces pyslfp's arrays exactly.

## Timings (PREM without ocean, five GLL nodes, this machine)

| lmax | elements | all degrees |
| --: | --: | --: |
| 64 | 657 | 0.4 s |
| 256 | 2575 | 2.6 s |
| 512 | 5137 | 6.6 s |

The full 0 to 4096 table should take a few minutes. On-the-fly
computation is practical for any lmax a fingerprint uses; the file
default stays because it is instant and needs no model.

## Decisions (David, 15 September 2026)

- The sub-package is `pyslfp.love_numbers`.
- The tidal numbers take the load notation: `h_t, k_t, l_t` beside
  `h_u, h_phi`; the consumers in `physics.py` change accordingly.
- Degree 0 is new functionality, not a discrepancy. It never mattered
  before because every direct load conserves mass, so its degree-0 part
  vanishes. The generalised loads of Al-Attar et al. (2024), eq. 55 (the
  displacement load zeta^u, the potential load zeta^phi, and the adjoint
  loads built from them) carry no such constraint, and the computed table
  gives their degree-0 response. The implicit assumption that degree 0
  drops out is to be removed from the solver.
- `EarthModelParameters` is rebuilt on planetmodel's `Scales` and
  `Dimensions` rather than its own hand-rolled derived scales, and G
  comes from `planetmodel.G_SI` (CODATA 2018) as the single source. The
  old value 6.6723e-11 differs by 3e-5; regression numbers move by that.
- The generalised (traction and potential split) Love numbers are
  essential and stay first-class: the adjoint theory is written in them.
- Horizontal displacements are wanted as an option, not always on.
- The packaging of the solver's outputs (the four-tuple) may be worth
  reconsidering, but any change has to be designed together with the
  pygeoinf operators, whose codomain is that direct sum; not in this
  round.
- The Fortran table is believed to be off by of order a per cent (an
  independent C++ code agrees with planetmodel), which matters for
  reproducibility rather than for results.
- Tutorials as `# %%` scripts rather than notebooks, taking a reader
  through the core functionality including the new material.

## Found on the way: the body belongs to the Love numbers

With G changed to planetmodel's, the adjoint identity of
`FingerPrintOperator` with rotation failed at 3e-5, and a scan showed
the mismatch proportional to the change in G and, separately, in g, but
not in G M. The rotational adjoint pairs the table's tidal numbers
against the parameters' G and g, so the identity holds only when the
Earth model uses the g and G the table was computed with. With a table
computed by the solver and the parameters taking g and G from it, the
identity closes to 5e-13; with the shipped table it closes to 3e-7 at
best, with g = 9.825652323 and G = 6.6723e-11, which are therefore the
constants of the code that produced it (`SHIPPED_TABLE_BODY`).

So `EarthModelParameters` now has `raw_gravitational_constant` (default
`planetmodel.G_SI`), and `EarthModel` replaces the raw sea-floor radius,
surface gravity and G of whatever parameters it is given by the table's
(`EarthModelParameters.with_body`), keeping the scales and everything
else. The default model therefore runs with the shipped table's
constants, exactly as before; a model from a computed table runs with
planetmodel's. This is the one-source-per-fact answer and it is tested
(`tests/love_numbers/test_earth_model.py`).

Measured for a ten per cent West Antarctic melt at degree 64, shipped
against computed table: sea level over the oceans differs by 0.1 per
cent (0.35 with the real ICE-7G state at degree 256), vertical
displacement by 0.7 per cent, the angular velocity by 0.2 per cent;
degree 0 plays no part for a mass-conserving load.

## Extensions, and how they fit

None of these goes in with the migration, but the class, the file format
and the solver are shaped so that they can.

**Degree 0 through the generalised equation.** `SeaLevelEquation`
already multiplies every coefficient, degree 0 included, by the Love
numbers, so once the table carries non-zero degree-0 values the
displacement and potential loads get their uniform response with no code
change. What needs a look is the mass-conservation step
(`slc += mean_slc - ocean_average(slc)`), which is correct for the direct
load and is where the assumption that the degree-0 load is zero lives;
with a displacement or potential load the uniform sea level is still set
by conservation of the direct load, so the step stays, but the test that
checks the adjoint identity of `FingerPrintOperator` is where a wrong
degree-0 treatment would show, and is the test to extend.

**The axial rotation component.** Superseded by `axial_rotation_plan.md`
(22 September 2026): the degree-0 numbers do not enter, and what the
exact treatment needs is an interior moment the table does not hold.
The paragraph is kept as written. The theory (eq. 27 onwards) keeps all
three components of omega; the numerics drop omega_3 because the change
in the polar moment of inertia is dominated by I_3 and because the
degree-0 deformation it needs was not available. The centrifugal
potential perturbation psi = -(Omega x x).(omega x x) has, for
omega_3, a degree-0 and a (2, 0) part; for omega_1, omega_2 the (2, 1)
parts used now. To add omega_3: `_rotation` becomes a 3-vector fixed
point using h, k at (0, 0) and (2, 0) and h_t, k_t at degree 2 (the
tidal numbers at degree 0 are exactly zero: a uniform external
potential is a gauge); `rotation_factor` and `inertia_factor` gain their
axial counterparts in `EarthModelParameters`;
`centrifugal_potential_operator` maps from `EuclideanSpace(3)`; the
returned angular velocity becomes length 3, which is an API change and
wants a release note. The Love number table needs nothing beyond
non-zero degree 0.

**Horizontal displacements.** The solver already computes V at the
surface for every forcing: `l_u, l_phi, l_t` are in the dataclass and are
only missing from the seven-column file. Two pieces:

1. Output. The tangential displacement is
   u_h = sum_lm V_lm grad_1 Y_lm with V_lm = l_l sigma_lm (plus the
   generalised terms), a surface gradient of a scalar field. pyshtools'
   `MakeGradientDH` gives the theta and phi components on the DH grid
   from a scalar coefficient array, so this is one synthesis step on
   the converged load, with the normalisation and the 1/a factor
   checked against a single-harmonic load. Since the four-tuple the
   solvers return is public, this is best a method on
   `SeaLevelEquation` (and an operator in `linear_operators`) that
   takes the converged sea level and rebuilds the load, rather than a
   fifth return value.

2. Adjoints. The paper limits functionals to the vertical displacement
   (section 4.1). A functional of horizontal displacement needs a
   tangential traction load in the generalised problem, and its Love
   numbers are the response of (U, V, phi) to a unit tangential
   traction on V at the surface. In the solver this is one more
   right-hand side beside `load(part="force")`, `load(part="potential")`
   and `tide()`: the surface V dof scaled by g a^2 and the l(l + 1)
   normalisation of grad_1 Y_lm. The bilinear form is symmetric, so the
   full response is, per degree, a symmetric 3 x 3 matrix of surface
   (U, V, phi) against unit (traction on U, traction on V, potential
   source) up to the factors of g, of which eq. 64 (g h^phi = k^u) is
   one off-diagonal entry; the new column gives two more identities
   (l^u against the U response to tangential traction, and k against
   l^phi) and the reciprocity test generalises to checking the whole
   matrix. Adding this column to the dataclass (`h_v, l_v, k_v`) and
   to `love_numbers` is a small change to make before PREM_4096 is
   regenerated, so the file needs regenerating once.

**The file format.** The current file is seven positional columns with
no names. The written file should name its columns in a header line
(`# columns: l h_u l_u k_u h_phi l_phi k_phi h_v l_v k_v h_t l_t k_t`),
and the reader should take the names from the header, falling back to
the seven-column layout for the shipped table (tangential and
tangential-forced columns NaN, as planetmodel's reader does now). Then
the regenerated PREM_4096 carries everything and any later column is a
header change, not a format change.

## Design

**One class, not two.** planetmodel's `LoveNumbers` dataclass becomes
pyslfp's `LoveNumbers`, replacing the file-reading class in `core.py`. It
gains `truncated(lmax)`, and the Green's functions and their plots move
onto it: they need only `h`, `k`, the radius and surface gravity, all of
which it already carries. `EarthModel` holds
`table.converted(parameters.scales).truncated(lmax)`.

Constructors, all classmethods so the dataclass constructor stays plain:

    LoveNumbers.from_file(path, *, lmax=None)          # the pyslfp file, SI
    LoveNumbers.default(*, lmax=None)                  # fetch PREM_4096 from Zenodo
    LoveNumbers.from_model(model, lmax, *, mesh=None, ngll=5, eps=1e-8)
    love.write(path)                                    # refuses complex numbers

`LoveNumbers(lmax, params, file=...)` stops working. Only `tests/test_core.py`
calls it; the tutorials go through `EarthModel`.

**EarthModel.** Add `love_numbers: LoveNumbers | str | Path | None`. A
path reads the file; None keeps the download default; a `LoveNumbers`
is used as given, converted and truncated (refused if its lmax is too
small or if it is complex). `love_number_file` stays as an alias for a
release. `with_degree` passes the same object on.
`EarthModel.from_planet_model(model, lmax, *, parameters=None, ...)`
computes the numbers and builds the model in one call.

**EarthModelParameters on Scales.** The constructor keeps
`length_scale`, `density_scale`, `time_scale` and the `raw_*` inputs, so
`from_defaults`, the ice models and the tests are untouched. In
`__post_init__` it builds `scales = Scales(length, density length^3,
time)` and every non-dimensional value is `raw / scales.factor(dims)`
with the `Dimensions` constants of `planetmodel.units`; the derived
`*_scale` fields (`mass_scale`, `load_scale`, `velocity_scale`,
`acceleration_scale`, `gravitational_potential_scale`,
`moment_of_inertia_scale`, `frequency_scale`) are only used inside
`core.LoveNumbers`, which goes, and are dropped. `gravitational_constant`
becomes `G_SI / scales.factor(GRAVITATIONAL_CONSTANT)`. Deriving the raw
parameters (mass, surface gravity, moments of inertia) from a planetmodel
model is possible with its field algebra and is left for later.

**Package layout.** `pyslfp/love_numbers/`:

| file | from | contents |
| :-- | :-- | :-- |
| `material.py` | copied | `NodalModuli`, `nodal_moduli`, `Material` |
| `assembly.py` | copied | `DegreeSystem` |
| `solver.py` | copied `love.py` | `DegreeSolution`, `solve_degree`, `love_numbers` |
| `table.py` | split from `solver.py`, plus `core.py` | the `LoveNumbers` dataclass, file read and write, conversion, truncation, Green's functions |
| `plot.py` | new | `plot_love_numbers` (h', l', k' and the tidal numbers against degree), `plot_degree_solution` (U, V, phi against radius with the skeleton boundaries drawn), the Green's function plots moved from `core.py` |

`core.py` keeps `EarthModelParameters` and `EarthModel` and imports
`LoveNumbers` from the sub-package; `pyslfp.LoveNumbers` is unchanged as a
name. `pyslfp.__init__` adds `love_numbers` beside `ice` and
`linear_operators`.

**Units in the solve.** Solve in the model's own units (SI for `PREM()`)
and convert the result; conversion is exact.

**Style.** `ruff format` reformats the copied files; the dense module
docstrings stating the weak form and the conventions are kept verbatim
as the only statement of the sign conventions. `pyslfp/love_numbers/*`
is on the E741 ignore list.

## Work packages

Each ends with the fast suite and ruff green. David commits.

**WP1: copy and test. Done 15 September.** `pyslfp/love_numbers/` with
`material.py`, `assembly.py`, `solver.py` and `__init__.py` copied from
planetmodel with imports pointed at the installed package;
`tests/love_numbers/test_solver.py` ported with both reference tables.
16 tests pass in 1.6 s. `core.py` untouched.

**WP2: one LoveNumbers, parameters on Scales. Done 15 September.**
`table.py` split out of `solver.py`; `core.LoveNumbers` replaced by the
dataclass with `from_file`, `default`, `from_model`, `truncated`, the
Green's functions and their two plots; the header-named file format
with the seven-column fallback; `EarthModelParameters` on `Scales`, G a
raw parameter defaulting to `G_SI`. The hand non-dimensionalisation is
reproduced to 1e-12 (`tests/test_core.py`).

**WP3: EarthModel and the consumers. Done 15 September.**
`love_numbers=` (a table or a path), `love_number_file` kept as an
alias, `with_degree` carrying the table, `from_planet_model`, `with_body`
(see above); `h_t, k_t` in `physics.py`. The adjoint identity with a
computed table is tested to 1e-9; the fingerprint differences are
recorded above rather than tested.

**WP4: the tangential-traction column. Done 15 September.**
`DegreeSystem.tangential()`: a traction -g grad_1 Y_lm per unit
amplitude, so the V dof takes -g a^2 l(l + 1); `solve_degree(...,
forcing="load_tangential")`; `h_v, l_v, k_v` in the dataclass and the
file, twelve columns in all; `reciprocity_residual` checks
g h^phi = k^u, h^v = l(l+1) l^u and k^v = g l(l+1) l^phi, all to 1e-12
on PREM (1e-10 for the near-fluid frozen case).

**WP5: visualisation. Done 15 September.** `plot_love_numbers` (load
numbers h', l l', l k' and the tidal numbers against degree, NaN columns
skipped, complex tables by real part) and `plot_degree_solution` (U, V,
phi against radius, boundaries drawn); the Green's function plots are
methods of `LoveNumbers`. Smoke tests.

**WP6: docs and tutorials. Done 15 September, bar the version.** README:
layout row, a Love numbers section, planetmodel among the dependencies,
the tutorials table; the same section on the Sphinx index page (apidoc
picks the sub-package up by itself). Five `# %%` scripts under
`tutorials/scripts/`: the four notebooks rewritten as scripts (1, 2, 4,
5) and a new one on Love numbers (3). `tests/test_tutorial_scripts.py`
runs them headless under the `slow` marker; all five run here (script 5
in about a minute). The notebooks are left in place. The version bump to
2.1.0 is David's, with the commit.

**WP7, first half done 15 September: the table.** The solver already
truncated each degree to the sub-mesh above its 1e-8 radius, but built
one uniform mesh sized for lmax (41 000 elements at 4096) that the low
degrees then solved on almost entirely. `graded_mesh` replaces it as the
default: uniform at the lmax rule's width down to lmax's own truncation
depth, then widening as 0.1 a ln(1 - d/a) / ln(eps), about d / 184, so
that every degree sees at least the uniform rule's resolution within its
own sub-mesh; 1083 elements for PREM at 4096, layer boundaries kept, the
last element of a span shortened or split rather than widened. Checked
against a uniform mesh with seven nodes and a quarter of the width:
degree 1000 and 4096 surface values agree to 3e-11; against the uniform
rule at degree 64, to 2e-6 (the discretisation error of either).

`~/.pyslfp_data/love_numbers/PREM_4096_planetmodel.dat`: PREM without
ocean, degrees 0 to 4096, the twelve columns with header and body line,
1.2 MB, 29 s to compute; reciprocity residual 1e-15, k_0 to 8e-9 of
-4 pi G a, k'_1 = -1 exactly. `LoveNumbers.default` now uses a file's
own header when it has one, so renaming this file to PREM_4096.dat in
the Zenodo zip is all that is needed to make it the default; the
SHIPPED_TABLE_BODY constants then apply to nothing and can go.

**WP7, second half done 15 September.** David uploaded the table as
`PREM_4096.dat` in a new version of the record, 22770094 (DOI
10.5281/zenodo.22770094); the zip now unpacks into
`pyslfp_love_numbers/`, so the downloader's folder map changed with it,
and the record number is set in the downloader, README and docs.
`ensure_data(key, refresh=True)` and `LoveNumbers.default(refresh=True)`
delete a cached copy and download again, since the cache is only checked
for presence and anyone with the old file would otherwise keep it.
`SHIPPED_TABLE_BODY` and the legacy branch of `default` are gone; the
seven-column reader stays for old files. The tests that described the
old table now describe the new one.

**Still David's:** in planetmodel remove `loading/`, tutorial 10, the loading half of
tutorial 12, `tests/loading`, and the README, CHANGELOG, `examples/README`
and `__init__` docstring lines, and release.

**Later, separately:** the axial rotation component; horizontal
displacement output as an option (a method on the solver and an
operator, not a fifth return value) and its adjoints, for which the
`h_v, l_v, k_v` column is now in place; any repackaging of the solver
outputs, designed with the pygeoinf operators.

## Out of scope, noted

- Parallelising degrees across processes via `pyslfp.parallel`.
- `EarthModelParameters.from_planet_model`.
- Complex (viscoelastic) Love numbers in the sea level solver.
- The `pyslfp.models` reorganisation; separate plan, separate branch.
