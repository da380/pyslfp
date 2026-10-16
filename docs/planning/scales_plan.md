# One internal frame: the scales refactor

Drafted 16 October 2026 from the units discussion with David. Status:
WP0 to WP4 implemented 16 October 2026 (WP3 revised, see below; WP4's
constructor helpers deferred, see below); WP5's sl3dpp side remains.

## Goal

Adopt a single internal non-dimensionalisation across pyslfp —
planetmodel's `Scales.geophysical` — with SI accepted and offered only
at the user-facing rim. This resolves docs/open-issues issue 2 (pyslfp,
AdGIA and sl3dpp sharing one convention) and removes the bug class
behind the continuous-adjoint itau blow-up: with one frame, unit
covariance of hand-derived formulas no longer has to be earned term by
term, because there is nothing to be covariant against.

The present architecture is unit-agnostic rather than non-dimensional:
`Material` carries whatever units its model came in (the cross-code
fixtures feed raw SI decks), and `SeaLevelEvolution` bridges to the
EarthModel's own scales through `ratio()` conversion factors. That
generality produced two unit bugs in one week and serves no purpose a
normalising ingestion step would not serve better.

## Settled decisions

1. **The frame is `Scales.geophysical`**: length the mean radius,
   density the Earth's mean, time 1/sqrt(G rho_bar) ≈ 1648 s, so
   G = 1 and g = 4 pi / 3. This is planetmodel's documented scheme and
   sl3dpp already follows it; pyslfp's current default (time
   sqrt(a/g) ≈ 806 s, so g = 1) is the one that changes, per the
   decision recorded in open-issues issue 2.

2. **The time scale stays gravitational**, although the physics here
   evolves on viscoelastic relaxation times. There are only three base
   units to spend; G = 1 with O(1) moduli and densities uses them, and
   keeps every assembled spatial operator — the degree systems, the
   Love number solves, the SLE coupling — O(1), which is where
   conditioning and readability live. Time enters the evolution only
   through the pure ratio h = dt itau and the kiloyear rim conversion,
   so relaxation times of ~1e7 in the frame cost nothing; a
   relaxation-based time scale would instead push G rho_bar T^2 ~ 1e13
   into the gravitational terms of every stiffness matrix. The
   relaxation scale is honoured at the rim: histories stay in
   kiloyears, and relaxation diagnostics are reported in years.

3. **Ingestion normalises, never refuses.** Models and Love number
   tables convert to the frame at construction: `LoveNumbers` already
   does (`table.converted(parameters.scales)` in `EarthModel`, which
   is how the SI Zenodo tables come in), and planetmodel's
   `Model.converted(scales)` does the same for models. The SI deck
   fixtures keep working unchanged; internally there is then only one
   frame.

4. **Outputs stay non-dimensional, with explicit SI converters.**
   SI-by-default outputs would put a conversion at every operator
   joint and weaken composition; the SI boundary is the user's rim,
   not every operator's rim.

5. **Objects keep carrying their `Scales`.** The convention becomes a
   cheap construction-time assertion and self-describing data, not an
   implicit assumption.

## Work packages

**WP0 — the migration gate, before anything changes.** A two-frame
covariance harness at the public-API level: the same physical body and
history run under the current default parameters and under geophysical
parameters (`parameters=` is already injectable everywhere), comparing
SI-converted outputs of the elastic fingerprint, the generalised SLE,
a viscoelastic run, and both adjoint kernel methods.
`test_continuous_viscosity_kernels_unit_covariance` is the template.
This catches latent g = 1 assumptions — algebra that silently dropped
a gravity factor because it was unity — which are the real migration
risk, before the default flips.

**WP1 — flip the default.** `EarthModelParameters.from_defaults`
switches its time scale to 1/sqrt(G rho_bar) (equivalently, builds
from `Scales.geophysical`). Chase the fallout: tests asserting
non-dimensional magnitudes, anything the WP0 harness flags.

**WP2 — normalise ingestion on the radial side.** `Material` (or the
`LoveNumbers.from_model` / `ViscoelasticOperator` path) converts its
model to the frame via `Model.converted`, so the material's scales and
the EarthModel's scales always agree. Assert the agreement where the
two meet.

**WP3 — collapse the bridge.** Revised on David's rule that a factor
is never dropped because the current scheme makes it one: the
`ratio()` plumbing in `SeaLevelEvolution` (`_to_mat_*`, `_to_nd_*`) is
KEPT as the written dimensional structure of the formulas, with the
WP2 assertion guaranteeing every factor is exactly one. The collapse
is the assertion, not a deletion; the code stays correct for any
scales should the assertion ever be relaxed, and the covariance gates
stay meaningful. The kiloyear conversion (`KYR /
scales.factor(TIME)`) is rim and stays.

**WP4 — the rim.** The convention page is written
(docs/source/units.rst): the frame, the rim, the time-scale argument
of decision 2, and the never-drop-a-unit-factor rule. The constructor
helpers — viscosity in Pa s and relaxation times in years at the
`Rheology` constructors, diagnostics reported in years — are deferred
pending an API decision: changing the meaning of existing arguments is
silent breakage, so they want either new keyword names (`eta_si=`,
`tau_years=`) or planetmodel-style tagged quantities.

**WP5 — fixtures and cross-code.** Regenerate the sl3dpp stage
fixtures without their SI anchors (the anchors exist only to bridge
the two conventions; with one frame they are redundant), coordinate
the matching sl3dpp changes, align AdGIA, and delete open-issues
issue 2. The outstanding stage-5 itau regeneration (open-issues
issue 1, step 4) is independent of this refactor and happens first,
so the fixture only churns once more.

## Non-goals

- No change to planetmodel: its scheme is the one being adopted.
- No SI-by-default outputs, and no change to the kiloyear rim for
  histories.
- No renaming or API redesign beyond what the rim helpers need.

## Risks

- Latent g = 1 assumptions are the main one; WP0 exists to find them
  before WP1 makes them wrong.
- External scripts and notebooks that read non-dimensional numbers
  from pyslfp objects will see different values; the Heathcote2026
  scripts on develop should be checked when harvested.
- sl3dpp's gates compare against committed fixtures; WP5 needs a
  coordinated window so both sides regenerate together.
