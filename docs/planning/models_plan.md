# The models submodule: a component vocabulary for inversion spaces

Status: design agreed, revised against pygeoinf 2 (see "Against
pygeoinf 2" below), and to ship in 4.0.0 with the pygeoinf 2 port. The
steric and mass-conservation pieces it builds on are in
pyslfp/models/loads.py.

## Goal

Give the library a vocabulary of physically named load components and
an assembler that turns a list of them into a ready inversion space,
so that common set-ups are one-liners while full generality stays
where it already lives, in pygeoinf. The three-component space of the
Heathcote2026 scripts — grounded ice, sterodynamic sea level, ocean
density — is one instance, not the design: land water storage,
glaciers separated from ice sheets, and firn separated from ice (the
ice-altimetry case, where the two load with different densities but
are observed as one height change) must drop in as further components
without touching the assembler.

## Settled decisions

1. **Organise by physical approximation, not instrument**; small base,
   composition over inheritance (the standing decision of the August
   design, which this plan revises — see decision 5).
2. **Full generality is pygeoinf's.** Direct sums, row and block
   operators, subspace conditioning are the framework; pyslfp
   contributes the physics vocabulary on top, never a wrapper around
   the framework.
3. **Concrete generalisation, not a schema.** Components are named
   physical choices (a glacier, not a "field with a mask"); the open
   family is the point, arbitrary abstraction is not.
4. **Set-ups are passive.** No priors, no calibration, no solvers, no
   conditioning choices — those stay with the caller, as decided at
   the harvest. A set-up is a frozen bundle of spaces, operators and
   functionals, with accessors only. This keeps the August design's
   central correction of the paper's utils — hold `state`, the spaces
   and the operators in an object rather than thread them through
   every call — without giving the object behaviour.
5. **Components replace enumerated model-space submodules.** The
   August design laid the package out as one submodule per model space
   (`models/{ice,ice_ocean,ice_ocean_density}.py`), with a rule that a
   component only earning "a column in the load operator" (land water
   storage was the example) would be a constructor option, to stop the
   enumerated names growing combinatorially. The family now in view —
   ice sheets, glaciers, land water storage, firn, eta, delta rho —
   makes the combinatorics the normal case rather than the edge case,
   so the vocabulary-plus-assembler replaces the enumeration, and the
   pre-built set-ups take over the role the named submodules would
   have played. The August decisions that stand unchanged: the
   submodule is called `models` (matching pygeoinf's `model_space`,
   disambiguated in docstrings from `EarthModel` and the ice models);
   no deep inheritance even though the spaces nest physically; no
   design for viscous or time-dependent physics (time dependence
   arrives as linked single-epoch models via filtering);
   `AltimetryObservationModel` and `JointAltimetryObservationModel`
   are removed in 4.0, a set-up's `geocentric_sea_level` replacing
   them.
6. **Direct observable contributions belong to the set-up.** The
   physics that made the August reorganisation necessary is that
   altimetry is model-dependent: sea surface height carries eta's
   direct term, which bypasses the sea level equation, and ice-sheet
   altimetry reads the height sum of ice and firn while the two load
   differently. A component therefore carries, besides its load, its
   direct contributions to named observables (eta to sea surface
   height; ice and firn to surface height), and the assembler derives
   each observable's direct operator the way it derives the load row.
   Observation models remain consumers of the set-up — they add
   sampling, averaging and noise, never physics.

7. **Scalar diagnostics map into Euclidean space.** GMSL functionals and
   the other scalar diagnostics are operators into `EuclideanSpace(1)`
   (or `EuclideanSpace(k)` for a split), pyslfp's existing convention,
   so that they compose with operators, push-forwards and dense
   covariances as they do now. pygeoinf's `LinearFunctional` is not
   used here.

8. **Fields are global; supports are part of the physics.** A
   component's space holds functions on the whole sphere, and a prior is
   a measure on that space as it stands. A prior is never masked by
   pushing it forward through a support projection: the masked samples
   jump at the support's edge, so the result is not a measure on a
   Sobolev space of order 1/2 or more, and it is only semidefinite. The
   support enters through the operators, as their regularity allows.
   Multiplication by the indicator of a Lipschitz region is bounded on
   H^s for |s| < 1/2 and on no H^s with s >= 1/2, and the sea level
   equation gains one order, so the spaces follow the chain

       components H^t --load--> H^s, 0 <= s < 1/2 --SLE--> H^(s+1)

   with s above 0 when the response is sampled at points.
   - Loads carry the supports (the ocean function, the ice and land
     indicators), so the load space has an order below 1/2.
   - Pointwise observables (sea and ice surface heights) take the field
     unmasked, into an observable space of order above 1, normally the
     response's field space; the observation points carry the support,
     since a field and its masked version agree inside it.
   - Integrated diagnostics (GMSL, regional averages, the steric sea
     level they are taken of) carry the supports; they are bounded on
     L^2.

   Loads, integrals and averages see a field only on its support, and
   point observations do when their points lie in it, so posteriors are
   those of a masked prior. The field off its support keeps its prior,
   and plots show `support(u)`. The intermediate spaces' metrics cancel
   in the adjoint of a chain from the fields to data, so the choice of
   load and response orders changes no result: the chain from H^2 fields
   to altimetry points built with H^2 loads and response, and with
   H^(1/4) loads and an H^(5/4) response, agrees exactly in the forward
   map and to 1e-13 in the adjoint at lmax 32 and 64.

## Against pygeoinf 2

- **Component names are labels.** The joint space is
  `DirectSum(spaces, labels=names)`, so `projection(name)`,
  `inclusion(name)`, `subspace(name)` and a product measure's
  `marginal(name)` work by name, and the assembler keeps no name-to-index
  bookkeeping of its own. Labels do not enter a direct sum's identity, so
  the row and block operators, which carry none, compose with it.
- **Constraints claim surjectivity.** Exact conditioning in pygeoinf 2
  needs the constraint operator to claim `SURJECTIVE`, so the set-up's
  constraints carry it. A prior on the global fields is positive
  definite (decision 8), so it is conditioned on a constraint as it
  stands; pygeoinf's refusal to condition a semidefinite prior does not
  arise.
- **Supports are multiplications by an indicator**
  (`multiplication_operator`), idempotent on any space and self-adjoint on
  a Lebesgue space; the traits they claim are verified with
  `testing.check_traits`. They serve display and diagnostics.
- **One product per load.** Each load is a single pointwise product,
  the support's indicator folded into its multiplier, never a support
  projection composed with a product. A product is left on the grid
  untruncated, and the next product acts on all of it, while a lifted
  operator's adjoint passes through the space's components, which keep
  only the part up to the truncation degree. The computed adjoint of
  g (f u) is therefore the exact adjoint of g T(f u), with T the
  truncation, and the composition fails the adjoint identity by the
  pairing of the two products' parts above the truncation degree. For
  the ice mask and load the failure, relative to the norms, is about
  2e-3 at lmax 32, 3e-4 at 64, 1e-4 at 128 and 1e-5 at 256: a
  discretisation error that falls with the degree. A single product
  satisfies the identity to rounding at every degree.
- **The steric map is a direct observable contribution.** The density
  component contributes to the observable `steric_sea_level` through
  `steric_sea_level_operator`, so the assembler derives the steric GMSL
  the way it derives every other observable, from the components alone.
- **Reference operators.** The Heathcote2026 scripts run on pygeoinf 2,
  so WP2 is cross-checked against their hand-built operators at small
  lmax by a one-off script, outside the test suite since the folder is
  to go; the suite tests the assembler against the same operators built
  from `loads.py`. Their `gmsl_split_operators` and
  `regional_decomposition_operators` are ported and ready for WP3.

## Design

**The component record.** A frozen `LoadComponent`: name, Hilbert
space, the to-load operator (None for a component that loads nothing),
its support (decision 8), its direct contributions to named observables
(decision 6), each into that observable's space, and its role. The
pointwise factories take the observable space as an argument. The role is the organising principle and
takes one of two values:

- **mass source** (grounded ice, glaciers, land water storage, firn):
  lives on land, adds or removes ocean mass. Each mass source admits a
  barystatic GMSL functional, -<load>_S / (rho_w A_O), derived from
  its load operator — the bookkeeping generalises to any number of
  sources without hand-wiring.
- **ocean-internal** (dynamic topography eta, vertically averaged
  density delta rho): rearranges ocean mass. These are the components
  the ocean mass-conservation constraint couples
  (`ocean_mass_conservation_operator`), and the density component
  carries the steric map (`steric_sea_level_operator`) and the
  effective steric scale.

**Component factories**, each a few lines on the existing lego
(state projections plus the load operators):

- grounded ice sheets: rho_i (1 - C) times the state's ice-sheet
  indicator, the ice sheets selected by rule (`ice_sheet_projection`):
  all ice south of 60S, and all ice within GTN-G region 5 (Greenland);
- glaciers: the RGI 7.0 glaciers as point masses at their grid nodes,
  rho_i a h with a the glacier area density and h a regional thickness
  change. Glaciers are far smaller than the grid's resolution and sea
  level is wanted well away from them; the whole area loads, with no
  (1 - C), since the glaciers are grounded whatever the ocean function
  says of their node. Glaciers at ice-sheet nodes belong to the ice
  sheets, which hold them in the ice model. Spreading the largest
  glaciers over their own area is the refinement to make if degree 512
  needs it;
- land water storage: rho_w times the ice-free land indicator,
  water-equivalent thickness;
- firn: the ice rule with a caller-supplied firn density — the reason
  it exists as a separate component is exactly that altimetry observes
  the height sum while the loads differ;
- dynamic topography: rho_w C eta;
- ocean density: D C delta rho.

**The assembler.** `ModelSetup(components)` derives: the joint space
(direct sum in the given order), the row load operator, the block
support, the per-component and total barystatic GMSL
functionals, the steric GMSL functional and barystatic/steric split
where the roles admit them, and the available constraints. Everything
is derived from the components' roles; adding a component never means
editing the assembler.

**Pre-built set-ups** as one-line factories over the vocabulary, which
double as worked examples and fix the terminology (Gregory et
al. 2019; note the zeta clash — the scripts' zeta is the dynamic
manometric sea level change, the library's zeta is the load):

- ice + eta (the joint fingerprint space already served by
  `joint_ice_ocean_to_load_operator`);
- ice + eta + delta rho (the Heathcote2026 space);
- ice + glaciers + land water + eta + delta rho (the modern sea-level
  budget space);
- ice + firn (ice-sheet altimetry).

## Work packages

**WP1 — the record, the roles, the factories**, with tests: every
factory's load operator adjoint-checked; ice and firn load with their
distinct densities; the mass-source barystatic functionals integrate a
uniform change to the bathtub answer.

**WP2 — the assembler and derived objects**, with identity tests: the
assembled three-component set-up reproduces the hand-built operators
of the harvest exactly; a steric (eta, delta rho) pair lies in the
assembled constraint's kernel; the per-component GMSL functionals sum
to the total; a global prior conditions exactly on the mass
constraint. Built as `pyslfp.models.ModelSetup`. On the real state at
lmax 32 and 64, on the scripts' own spaces, the continuous sea level and
sea surface height operators and the steric GMSL agree with the
scripts' exactly in the forward map. The adjoints differ: the scripts
compose the ice mask with the ice load, and their sea level and sea
surface height operators fail the adjoint identity, relative to the
norms, by 2e-3 and 1e-3 at lmax 32 and by 2e-4 and 6e-5 at lmax 64,
where the set-up's hold to the solver's tolerance. The scripts'
barystatic GMSL pairs the weighting with the masked field spectrally,
while the set-up's is the grid integral of the load, the mass the sea
level equation sees; at worst over H^2 fields the two differ by 7% at
lmax 32, 1.5% at 64, 0.5% at 128 and 0.09% at 256.

**WP3 — the pre-built set-ups and the remaining harvest** (built): the
scripts' `gmsl_split_operators` and the SSLC/DMSLC/BMSLC regional
decomposition land as derived diagnostics of a `ModelSetup` rather
than free functions, completing the Heathcote2026 harvest list. The
scripts move onto set-ups, which drops their prior masking (decision 8)
and with it the composed mask-and-load rows and the spectral barystatic
pairing, and puts their loads in a space of order below 1/2.

- **Diagnostics that need the sea level equation** (the true GMSL, the
  residual form of the steric GMSL, the SSLC/DMSLC/BMSLC decomposition)
  take the fingerprint operator as an argument; a set-up holds none and
  stays passive. The decomposition is written through the roles: BMSLC
  is the response to the mass sources' loads, DMSLC the response to the
  ocean-internal loads plus the sterodynamic sea level change less its
  steric part,
  SSLC the steric sea level. The scripts' zeta (dynamic manometric
  change) is renamed, zeta being the load in the library.
- **Ready-made set-ups**: ice + eta; ice + eta + delta rho; ice + firn;
  the budget (ice + glaciers + land water + eta + delta rho). Each takes
  the load space, the field space and the observable space, as the
  factories do.
- **The scripts are replaced by new, simplified ones** built on set-ups,
  sharing `Heathcote2026/common.py` and `figures.py`. They were checked in
  quick mode against a local record of the old ones; every difference
  traces to a fix or to the change of formulation. The formulation is in
  `Heathcote2026/FORMULATION.md`, and the points the paper's revision must
  take up are in `Heathcote2026/PAPER_NOTES.md`.
- **Order.** WP3 follows the clean-up of the hard-coded regions (see the
  roadmap), so that the ready-made set-ups and the paper's revision runs
  use the physical definitions of the ice sheets, the glaciers and the
  ocean.

## Non-goals

- Priors, calibration chains, solver and conditioning choices.
- Observation models: they consume a set-up's spaces and components,
  they are not part of one.
- Time dependence; the set-ups describe single-epoch (rate) spaces.
  Revisit once pygeoinf has a tensor product of spaces.
- Priors: a separate layer that takes a set-up, built later.

## Risks

- Record scope creep: the component holds name, space, load,
  support, role and direct observable contributions — nothing
  else. Sampling, averaging and noise belong to the observation
  operators.
- The firn density is a modelling choice, so it is an argument, never
  a constant.
- Land-mask subtleties (the Caspian, lakes) are inherited from the
  state's projections; the factories reuse them rather than redefining
  masks.
- A support supplied by the caller (a mask, named regions or custom
  glacier positions) would serve any component whose support the state's
  projections do not describe.
