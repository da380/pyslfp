# Ice-history adjoints for the GIA solver: the plan

Drafted 14 October 2026 from David's direction and a reading of
Yu, Al-Attar, Syvret & Lloyd (2025, GJI 240, 329-348; ggae388 in
~/dev/AdGIA/doc/Elasticity), which states the first-order adjoint GIA
problem in full generality with rotational feedbacks.  Status: plan.

## Scope

Sensitivity kernels of a chosen objective functional with respect to
the ICE THICKNESS HISTORY, K_I on the sphere x time, and with respect
to the INITIAL SEA LEVEL, K_SL0 -- their eqs (64) and (65) -- for the
forward problem WITH migrating shorelines (David, 14 Oct: shorelines
are important, the elastic case already implements them, and the
frozen restriction is not to be baked into the adjoint).  NOT in
scope: viscosity kernels (that is what AdGIA is being written for, and
sl3d does in a more restricted sense); second-order adjoints.

**Prerequisite: migrating shorelines in the forward evolution** (the
plan's WP-G5, pulled forward).  The nonlinear solver exists in the
elastic machinery (`solve_nonlinear_equation`); it needs the same
generalisations the linear one received (the response channels, the
inertia-moment term, the tri-state rotation), and the evolution gains
a per-step nonlinear solve with the ocean function updated by
flotation.  The forward run records the ocean-function history C(t),
one grid per step (~2 MB per step at lmax 256 as floats, less as
masks): that history is all the adjoint needs of the trajectory.

## Why this is cheap here

1. **The adjoint problem is the forward problem with more general
   forces** (their Section 3.3): D_SL J forces like an ice load; D_u J
   and D_phi J enter as inhomogeneous surface terms, eqs (57)-(58);
   D_omega J acts as an angular momentum perturbation.  These are
   EXACTLY the channels `solve_generalised_equation` already carries
   (displacement_load, gravitational_potential_load,
   angular_momentum_change), built for the elastic adjoint; the
   viscoelastic adjoint step reuses them with the step's effective
   Love numbers.  The elastic adjoint mapping in
   `linear_operators/physics.py` (zeta_d = xi-dagger, zeta^u =
   -u-dagger, zeta^phi = -g phi-dagger, L = -g(omega-dagger - ...)) is
   the per-step template.
2. **The adjoint ocean function is not determined dynamically**: it is
   the time-reversed C(t) of the forward run (their Section 3.3), so
   the adjoint problem is LINEAR with time-dependent coefficients,
   whatever the forward nonlinearity.  Only C(t) is stored -- no other
   forward state enters the adjoint operator; the kernels depend on
   the trajectory through C(t) alone (the shoreline-motion singular
   terms cancel, their eq. 62 ff.).
3. **Every discrete building block transposes within existing
   machinery**: the degree systems are symmetric; the memory-force
   assembly is the mu-weighted transpose of the strain map (same GLL
   quadrature), so the strain map's transpose is the same assembly
   routine with different nodal weights; the scheme weight updates are
   diagonal; the surface extraction transposes to point forcings at
   the surface dofs, which are the cached unit solutions' RHS.

## The approach: both adjoints (David, 14 Oct)

Two implementations, deliberately:

1. **The discrete transpose** -- the exact transpose of the
   time-stepped forward map.  Machine-exact gradients for the discrete
   model, gateable by dot-product tests; but tied to the specific
   numerics (change the stepper and it must be re-derived).
2. **The continuous adjoint, then discretised** -- David's preferred
   formulation: the adjoint equations of Yu et al. have the SAME form
   as the forward problem with more general forces, so the adjoint run
   IS the forward machinery in reversed time, stepper-agnostic by
   construction.  The commute error (discretise-then-transpose vs
   transpose-then-discretise) is second order for the trapezoid and,
   for this weakly nonlinear problem, expected minimal; MEASURE it by
   comparing the two implementations' gradients -- that comparison is
   itself a deliverable.

The engine-level piece is shared: `SeaLevelEquation.
adjoint_generalised_solve`, the exact reverse-mode of the generalised
solve returning duals of all four step channels (direct load, the two
response channels, the inertia moment), gated by a dot-product
identity at machine precision in all three rotation modes (DONE,
14 Oct; the rotation-block transpose must dualise the FULL degree-2
objects disp_2/grav_2, not the forcings -- the first attempt's bug).

First rung of measurements: SEA LEVEL functionals only (the dominant
GIA observable); D_u, D_phi and D_omega measurements need two further
radial forcing channels (the split load vectors, `load(part=...)`)
and come after.  Point-in-time measurements enter the discrete
transpose naturally at their time node; the continuous realisation of
the same (impulsive adjoint forces, the jump conditions of AT14
D81-D82) is informed by it.

Forward step (exponential trapezoid; backward Euler analogous), as a
linear map with state (m^n, d^n):

    m~          = E m^n + A d^n                 (diagonal weights)
    x_mem       = K_eff^{-1} M m~               (memory solve; M = D* W_mu)
    (U, phi)_mem, J0 = P x_mem                  (surface values, inertia moment)
    (xi, omega, sigma_tot, psi) = L_eff(ice_{n+1}, (U, phi)_mem, J0)
                                                 (the per-step SLE)
    x           = x_mem + X_load sigma_tot + X_tide psi   (unit solutions)
    d^{n+1}     = D x
    m^{n+1}     = m~ + B d^{n+1}

Transpose, run in reverse over the steps: adjoint state (m-dagger,
d-dagger); K_eff^{-1} self-transposed; L_eff^T is the adjoint SLE via
the generalised channels with the same effective table; M^T and D^T by
the shared assembly with swapped weights; E, A, B diagonal.  Per step
the cost is the forward step's: one symmetric banded solve per degree
plus one SLE solve.

The per-step transposition must be derived once on paper, carefully,
and recorded in gia_notes.md before coding -- the chain rule over the
start-strain cache (d^n enters both the next step's m~ and nothing
else) and the right-limit initialisation at t0 both participate.

## Kernels

Their eqs (64) and (65), with the stored C(t):

    K_I(x, t_n)  = -rho_i g (1 - C(t_n)) T(SL1-dagger)(x, t_n),
    plus the singular baseline term at t0:
    K_I(x, t0)  += +rho_i g (1 - C0) int T(SL1-dagger) dt,
    K_SL0(x)     = int [ D_SL J - rho_w g (C - C0) T(SL1-dagger) ] dt.

In the discrete setting the time integrals are quadratures of the step
grid.  Under frozen shorelines these reduce (C = C0, the K_SL0
correction vanishes) and the t0 term is minus the sum of the others:
a delta I constant in time then annihilates J's variation -- verify
that identity as a gate in the frozen limit rather than assume it.

## Interface sketch

- A measurement object giving the derivatives of J at the forward time
  nodes: time series of (D_SL J, D_u J, D_phi J, D_omega J), grids and
  vectors; the common case is a weighted sea-level reading at one or
  several times.
- `run_adjoint(measurement)` on the evolution (name to David's taste),
  returning the kernel history K_I (per date, SHGrids) and K_SL0, and
  the adjoint fields if wanted.
- No pygeoinf coupling at the outset (David, 14 Oct): kernels are
  plain grids; the operator wrapping waits for pygeoinf v2.

## Status (14 Oct, late)

DONE: the discrete transpose, end to end, frozen shorelines,
sea-level measurements:

- `SeaLevelEquation.adjoint_generalised_solve` -- the engine
  transpose; dot-product gate at machine precision, all three
  rotation modes.
- The operator's transpose surface: `strain_transpose` (D^T, sharing
  the memory-force assembly), `apply_weights` (W), `surface_dual`
  (P^T), `moment_row` (q), and public accessors for the degree
  workspaces, effective systems, unit responses and scheme weights.
- `SeaLevelEvolution.adjoint_ice_history(dates, measurement)` -- the
  reversed sweep; needs NO forward run (linearity).  Returns
  `IceKernels`: per-date load duals and thickness kernels with the
  baseline (delta at t0) term folded in.
- Gradient identity gate: arbitrary per-node ice perturbations, J
  recovered to 8e-13 (trapezoid) and 2e-13 (backward Euler) relative,
  rotational feedbacks on.

ALSO DONE: the singular structure of the ice kernels (David, 14 Oct;
Yu et al. Section 3.7 and eq. 81).  `IceKernels` carries two views:
the exact DISCRETE per-node duals (which satisfy the gradient
identity but lump the singular parts in with the continuous one),
and the DISTRIBUTIONAL decomposition stored separately -- the
continuous density per kiloyear; one delta amplitude per observation
time, computed as the instantaneous (unrelaxed) fingerprint adjoint
of that measurement, grid-independent by construction (their
"generalised fingerprint problem"); and the t0 baseline delta.
Gated by grid refinement: the density and baseline converge, the
lumps are identical to all digits, the raw node duals merely scale
with the time weights.

AND the QUELLED view (David, 14 Oct): the kernel with respect to the
thickness RATE -- the cumulative tails of the node duals, Backus's
device -- is bounded with finite jumps at the observation times only,
pairs with thickness histories continuous in time, and makes the
initial-thickness dual vanish identically under frozen shorelines (a
uniform shift loads nothing; the node duals sum to zero exactly).
Gates: exact against per-interval rate perturbations (1e-6 relative,
full feedbacks); refinement consistency (a coarse interval's value is
the mean of its halves to 1.3 per cent); the uniform-shift identity
to round-off.  Remaining inversion-side smoothing (Sobolev duals in
the manner of pygeoinf, needing specialisation) acts on these views.

DONE (15 Oct): the continuous adjoint
(`continuous_adjoint_ice_history`) -- the reversed forward run, free
evolution between observations by the unchanged step machinery, the
eq.-(80) jump m-dagger += d-tilde-dagger / tau at each observation
from the instantaneous fingerprint problem, and the node quadrature
split across the density's finite jumps.  THE COMPARISON (the point
of doing both): on a stiff configuration (steps of several relaxation
times) the continuous gradients carry 19 -> 5 -> 2 per cent error
under refinement while the discrete transpose stays at ~1e-13; once
the steps resolve the relaxation the continuous error is 5e-4 falling
super-quadratically to 1e-5.  The doctrine (David, 15 Oct): a discrete
adjoint is always accurate, but can be accurate for the WRONG
problem -- it is exactly faithful to the discretised functional,
which in an under-resolved run is a poor stand-in for the continuum
one.  The continuous and discrete gradients differ exactly when the
calculation is not accurate enough, so their DISAGREEMENT is a
resolution diagnostic: when it is large, refine the forward run
rather than prefer either gradient.  The discrete transpose remains
what an optimiser of the discrete model needs (its exactness makes
the optimisation loop consistent); the continuous formulation is the
stepper-agnostic statement of the continuum problem.  The two methods' delta amplitudes, computed
by different realisations (the engine transpose against the forward
channel mapping), agree to 1e-8, cross-validating both.  Two faults
the comparison caught in the continuous driver: the smooth adjoint
fields are densities per MATERIAL time (the jump divides by tau in
material units) and must be converted before the date quadrature; and
a measurement's legacy lives in the jumped internal variables, not in
a persistent load.

DONE (15 Oct): the migrating-shoreline kernels.  Both drivers take
`ocean_functions` -- the C(t) history off a migrating forward run's
snapshots -- and the engine's forward and adjoint solves take an
`ocean_function` override; the adjoint is the linearisation about the
trajectory, linear with time-dependent coefficients, the grounding
factors following C(t).  Gates: a constant history reproduces the
frozen kernels exactly; the discrete transpose matches central
differences about the moving trajectory to 1e-10 (the forward is
precisely the linearised map while no flotation cell flips); the
continuous method converges at second order once the rheology is
resolved (1.2 -> 0.3 per cent), and on the stiff under-resolved body
the two disagree by tens of per cent -- the resolution diagnostic
doing its job, not a defect of either.  The t0 node's kernel under a
perturbed INITIAL thickness needs the initial state rebuilt (its
flotation changes); that piece is deferred with the initial-sea-level
kernels.

DONE (15 Oct): measurements of all four outputs.  For the discrete
transpose no new radial machinery was needed -- the engine adjoint
already carries the u, phi and omega duals, and the radial reverse
consumes only its returned input-duals; gradient identity exact with
all four dual types.  The continuous jump realises the elastic
adjoint's channel mapping (zeta^u = -g_u, zeta^phi = -g g_phi,
L = -g (g_omega - Psi* g_phi)) with the SPLIT radial load responses
(the force-only and attraction-only unit solutions); 6e-4 in the
resolved regime.  A `Measurement` carries the four duals; a bare
grid remains a sea-level dual.

DONE (15 Oct): the initial-state kernels.  Deriving eq. (65)
discretely showed K_SL0 assembles from the ALREADY-COMPUTED per-node
load duals -- the initial sea level enters each step's load as
rho_w (C_n - C0) delta SL0 -- at zero extra cost, identically zero
under frozen shorelines; `IceKernels.initial_sea_level` carries it
(for objectives on the sea level CHANGE; absolute-sea-level
objectives add the sum of their own duals).  The derivation also
caught a factor fault: the t0 baseline must ground with the INITIAL
(1 - C0), not the per-node grounding (the frozen case masked it).
Both initial-state duals are gated by central differences with
rebuilt initial states: SL0 to 1e-10 and I0 to 1e-13, step-size
independent.  The Newton-type present-day calibration of SL0 now has
its gradient available; wiring it into
`calibrate_initial_sea_level` is an application-layer follow-up.

DONE (15 Oct): kernel outputs at the ice model's dates.
`IceKernels.at_dates(model_dates)` applies the transpose of the
model's linear interpolation, mapping the run-grid duals onto the
model's own slices; gated exactly (1e-9) against slice perturbations
under a finer run grid.  Pure post-processing, no driver changes.

DONE (15 Oct): the adaptive exponential trapezoid, radial and
coupled.  `ViscoelasticOperator.adaptive_step`/`integrate` and
`SeaLevelEvolution.run_adaptive`; the embedded ETD1 companion costs
nodal work only (err = RMS of (m - m_hat)/(atol + rtol max|m|),
dt_next = dt clamp(0.9 err^{-1/2}, 0.2, 4)).  Lessons that matter:
the atol default must follow the state's own magnitude (the internal
variables carry the degree-basis scalars' units), so atol=None means
1e-6 rtol max|m|; the quantised ladder uses FLOOR plus a stagnation
promoter (climb a rung after eight healthy accepts) -- rounding to
the nearest rung thrashes, flooring alone freezes; and the estimate
is the FIRST-ORDER companion's, so rtol_step is conservative for the
propagated second-order solution by two to three orders (measured:
rtol_step 1e-1 -> 3e-4 global, 1e-2 -> 7e-6).  Gated radially
(tolerance ordering, striding, ladder, elastic striding) and coupled
(ordering vs fixed reference, ladder <= 12, striding).  Variable
steps compose with both adjoints: the accepted dates become the
kernel grid.

TO DO, in order: (exhausted -- see LATER)

DONE (16 Oct): spherically symmetric viscosity kernels, both
methods.  The forward records its trajectory on request (`record=True`
attaches the internal states to the snapshots); `adjoint_ice_history`
given `internal_states` accumulates the EXACT DISCRETE dJ/d itau(r)
per branch -- per step the scheme-weight derivatives in h = dt itau
(weights.py grew them, series-switched like the weights) pair the
sweep's duals with the forward variables: mtilde-dual with
(de m + dalpha d0), the entry m-dual with dbeta d1, and the effective
moduli's dependence as dbeta <strain(y), W_k d1> through the branch
bilinear (backward Euler analogous via dw).  Gated: FD residual falls
at exactly second order in eps and Richardson lands on the kernel to
1e-6, both schemes.  `continuous_adjoint_ice_history` with
`internal_states` accumulates int W_k m-dagger / (g itau) (d - m) dt
over the reversed nodes with the density's split trapezoid: the
multiplier of the m-dynamics is lambda_k = W_k m-dagger / (g itau),
the g (material units) because the self-adjoint channel mapping poses
the adjoint in the g-weighted energy form -- pinned down empirically
by the rho-dependence of the mismatch constant (4 pi rho / 3).
Gated: converges to the discrete kernel at second order when
resolved (1.8% -> 0.46% -> 0.11%), disagrees at 38% on the stiff
body (the resolution diagnostic).  `IceKernels.itau` per branch,
nodal; `IceKernels.viscosity(rheology)` maps by -itau^2/mu.
Migrating trajectories are covered too (16 Oct): `_run_migrating`
records on request (the start state's cache is the zero solve's), and
both kernel variants gate about the marine trajectory -- discrete FD
residual second order with Richardson landing on the kernel to 1e-5,
continuous converging to discrete at second order (1e-3 -> 2.3e-4).
Checkpointing (16 Oct, built as the trial for AdGIA's 3-D
problem): the exact inverse of an exponential-trapezoid step is the
same update at -dt with the strain roles swapped (alpha(-h) =
-beta(h)/e(h) and conversely), so `ViscoelasticOperator.step_back`
reuses the two-phase machinery with a negative-dt flavour (its
effective-modulus weight phi1(-h) > 1 stays positive definite);
backward Euler inverts explicitly, m = (1+h) m' - h d'.  Recording
takes a stride: `record=M` keeps internal states every M nodes (and
the last) while the per-node load and centrifugal coefficients are
always kept (surface-sized), and `ForwardTrajectory`/_TrajectoryCursor
reconstruct the gaps during the reverse sweep in lockstep with the
adjoint.  Dissipation backwards amplifies roundoff by exp(h) per
branch-node, so the accumulated h = dt/tau across a gap is the digit
budget -- measured: resolved gaps 1e-12 agreement with full
recording, stiff gaps (sum h = 6.4, amplification ~600) 3e-11; the
unit gate inverts a step to 1e-11.  Works for frozen and migrating
trajectories; the adjoints accept snapshots directly, a
ForwardTrajectory, or a full state list as before.

LATER, from David (15 Oct):

- (done above) **Spherically symmetric viscosity kernels** K_eta(r).  No 3-D
  viscosity kernels here (no generalised spherical harmonics in
  Python, and the cost is sl3d/AdGIA territory); but the 1-D
  reduction of Yu's eq. (61) collapses degree by degree -- the
  contraction is our (4/3, k^2, k^2(k^2-2)) scalar pairing, AT14
  (D41)-(D50) -- and accumulates during the reverse sweep.  Needs the
  forward trajectory (d - m)(t) stored (in memory at test scales,
  checkpointed at production).  The discrete variant differentiates
  the scheme weights in tau; the resolution diagnostic applies.
- **Decay-time functionals** (Hudson Bay and the like, usually
  finite-differenced in the literature): the decay time is defined by
  a nonlinear least-squares fit to a sea level curve; differentiate
  the fit's stationarity conditions locally to get the decay time's
  derivative with respect to its data, which maps to time-dependent
  weights on sea-level measurements -- then any of the kernel
  machinery applies.  An appendix of AT14 works the idea in a simpler
  case; the construction is not unique and could be done differently.
  Build once the basic kernel machinery (above) is in.

## Gates

1. **The gradient identity to solver precision in the frozen limit**:
   there J is linear in the ice history, so <K_I, delta I> +
   <K_SL0, delta SL0> equals J(perturbed) - J(base) EXACTLY (up to
   solve tolerances) for arbitrary perturbations on the step grid.
   Run with random perturbations, rotational feedbacks on and off.
2. **Central finite differences about the migrating-shoreline
   trajectory**: the kernels predict the linearised change of J; the
   FD residual must converge at second order in the perturbation size
   (the Crawford et al. 2018 verification pattern).
3. The constant-in-time delta I identity in the frozen limit.
4. Degree-wise dot-product test of the transposed step:
   <F x, y> = <x, F^T y> for a single step at machine precision.
5. The elastic limit: with itau = 0 the kernels must reduce to the
   elastic fingerprint adjoint's (linear_operators), which is already
   gated against its forward operator.

## Sequencing

1. The v3 restructuring (naming to David).
2. Migrating shorelines in the forward evolution: the nonlinear
   solver's generalisations, the per-step flotation update, the C(t)
   record, and the forward gates (mass conservation with moving
   shorelines; the frozen limit recovered when the flotation update
   is suppressed; the elastic nonlinear solver recovered when every
   tau is infinite).
3. The adjoint, as above, landing beside the evolution.

Also queued (David, 15 Oct): a review of ALL docstrings across the
library for correctness and house style -- detailed, no development
narrative, every argument documented Google-style except trivial
arguments self-described by the function's statement of purpose.

Design note, recorded (David, 15 Oct): continuous-in-time
measurements are a SEPARATE kind from the point-in-time ones (as in
AdGIA) -- a density-flagged measurement whose duals are per unit
time; the discrete driver folds quadrature-weighted duals into the
nodes but excludes them from lump extraction (a density produces no
deltas), and the continuous driver carries them as in-step forcings
with no eq.-(80) jump.  Not built: such measurements are not made in
practice, and the decay-time functionals are fits to discrete
samples; the point type covers the known uses.
