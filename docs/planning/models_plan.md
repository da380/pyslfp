# The models submodule: a component vocabulary for inversion spaces

Drafted 16 October 2026 from the model-spaces discussion with David.
Status: design agreed in outline, not started; the steric and
mass-conservation lego it builds on landed with the Heathcote2026
harvest (see pyslfp/linear_operators/loads.py).

## Goal

Give the library a vocabulary of physically named load components and
an assembler that turns a list of them into a ready inversion space,
so that common set-ups are one-liners while full generality stays
where it already lives, in pygeoinf. The three-component space of the
Heathcote2026 scripts — grounded ice, dynamic topography, ocean
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
   become deprecated aliases for a release when their replacements
   land.
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

## Design

**The component record.** A frozen `LoadComponent`: name, Hilbert
space, the to-load operator (None for a component that loads nothing),
the support projection, its direct contributions to named observables
(decision 6), and its role. The role is the organising principle and
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

- grounded ice sheets: rho_i (1 - C), ice mask excluding glaciers;
- glaciers: the same rule on the glacier mask;
- land water storage: rho_w on the land mask, water-equivalent
  thickness;
- firn: the ice rule with a caller-supplied firn density — the reason
  it exists as a separate component is exactly that altimetry observes
  the height sum while the loads differ;
- dynamic topography: rho_w C eta;
- ocean density: D C delta rho.

**The assembler.** `ModelSetup(components)` derives: the joint space
(direct sum in the given order), the row load operator, the block
projection operator, the per-component and total barystatic GMSL
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
to the total.

**WP3 — the pre-built set-ups and the remaining harvest**: the
scripts' `gmsl_split_operators` and the SSLC/DMSLC/BMSLC regional
decomposition land as derived diagnostics of a `ModelSetup` rather
than free functions, completing the Heathcote2026 harvest list.

## Non-goals

- Priors, calibration chains, solver and conditioning choices.
- Observation models: they consume a set-up's spaces and components,
  they are not part of one.
- Time dependence; the set-ups describe single-epoch (rate) spaces.

## Risks

- Record scope creep: the component holds name, space, load,
  projection, role and direct observable contributions — nothing
  else. Sampling, averaging and noise belong to the observation
  operators.
- The firn density is a modelling choice, so it is an argument, never
  a constant.
- Land-mask subtleties (the Caspian, lakes) are inherited from the
  state's projections; the factories reuse them rather than redefining
  masks.
