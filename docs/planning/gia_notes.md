# GIA implementation notes: the degree-basis internal-variable system

Working notes for the implementation of `docs/planning/gia_plan.md`,
transcribed from Al-Attar & Tromp (2014) Appendix D on 14 October 2026
and rederived for the generalised Maxwell (Prony) body.  Equation
numbers are theirs.  This file records the algebra the code implements,
so that the kernels can be checked line by line against it.

## The spheroidal internal-variable components (D30)-(D33)

For one (l, m), with k^2 = l(l + 1), the trace-free symmetric internal
variable m of a branch reduces to three radial scalars per (l, m):

    M_lm  (the purely tangential component, weight k^2 (k^2 - 2)),
    R_lm  (the radial-radial component, weight 4/3),
    S_lm  (the radial-tangential component, weight k^2).

The toroidal scalars (N, T) stay zero under spheroidal forcing (D88)
and are not carried.

## The deviatoric strain scalars

The scalars that drive (M, R, S), read from (D71)-(D73), with the
assembly's operators X = r d_r U, Y = 2U - k^2 V, S_op = U + r d_r V - V:

    e_M = V,
    e_R = r d_r U - U + (1/2) k^2 V   = X - Y / 2,
    e_S = r d_r V - V + U             = S_op.

They are evaluated nodally by GLL collocation: per element,
r d_r f = (r / jac) D f with the mesh's differentiation matrix, exactly
as the assembly builds its B operator.

Consistency check against the elastic form: the mu-part of the
isotropic degree-l bilinear form is

    (4/3) mu e_R e_R' + k^2 mu e_S e_S' + k^2 (k^2 - 2) mu V V',

and expanding (4/3)(X - Y/2)(X' - Y'/2) + ... reproduces the
transversely isotropic assembly C X X' + (A - N) Y Y' + F(X Y' + Y X')
+ k^2 L S S' + k^2 (k^2 - 2) N V V' under A = C = kappa + 4 mu / 3,
F = kappa - 2 mu / 3, L = N = mu.  Verified by hand 14 Oct 2026.

## The ODEs, branch-general

AT14 write the Maxwell case (D71)-(D73).  For the generalised Maxwell
body with branches (mu_k, tau_k) and long-term shear mu_inf,

    T = kappa tr(eps) I + 2 mu_inf d + sum_k 2 mu_k (d - m_k),

each branch carries its own (M_k, R_k, S_k) with the SAME driving
scalars and its own time:

    dM_k/dt = (e_M - M_k) / tau_k,
    dR_k/dt = (e_R - R_k) / tau_k,
    dS_k/dt = (e_S - S_k) / tau_k.

The branch-wise structure is AdGIA's; the components are AT14's.  The
classical Maxwell body is one branch with mu_inf = 0, tau = eta / mu.
Crawford et al. (2017), eqs (2.23) and (2.39)-(2.48), give the same
generalised Maxwell internal-variable convention in the continuum
(m_i the tau_i-exponential average of d, branch stress 2 mu_i
(d - m_i)), confirming the branch algebra; their Appendix A nonlinear
extension is out of scope here (it belongs with lateral viscosity
variations, hence with the 3-D code).

## The memory force

The quasi-static equilibrium at fixed m reads

    A_l(U, V, phi | tests) = F_load(tests) + b_mem(m | tests),

with A_l the elastic bilinear form (D76) assembled with the UNRELAXED
moduli mu_U = mu_inf + sum_k mu_k, and, from the pairing (D65)/(D66)
read one-sided (the mu-part of the quadratic form applied to m):

    b_mem(U', V') = sum_k int mu_k [ (4/3) R_k e_R(U', V')
                                     + k^2 S_k e_S(U', V')
                                     + k^2 (k^2 - 2) M_k V' ] dr,

a GLL quadrature per element; the integrand pairs the nodal branch
state with the same strain operators as above applied to the test
functions.  At l = 0 only the R term survives (k^2 = 0); at l = 1 the
M term drops (k^2 - 2 = 0) though M still evolves.

## Scheme eliminations (ported from AdGIA `viscoelastic.cpp`)

With h_k = dt / tau_k per node and branch:

Backward Euler: m^{n+1} = (m^n + h d^{n+1}) / (1 + h), so

    mu_eff = mu_inf + sum_k mu_k / (1 + h_k),
    memory state  m~_k = m_k^n / (1 + h_k),

and the step's equilibrium is the elastic solve with mu_eff and
b_mem(m~).

Exponential trapezoid: with e = exp(-h), phi1 = (1 - e)/h (series
1 - h/2 + h^2/6 - h^3/24 + h^4/120 for h < 1e-3), alpha = phi1 - e,
beta = 1 - phi1:

    m^{n+1} = e m^n + alpha d^n + beta d^{n+1},
    mu_eff = mu_inf + sum_k (1 - beta_k) mu_k = mu_inf + sum_k phi1_k mu_k,
    memory state  m~_k = e_k m_k^n + alpha_k d^n.

Exponential Euler (ETD1): m^{n+1} = e m^n + (1 - e) d^n; no effective
operator, the solve at t^n uses the unrelaxed moduli and b_mem(m^n).

Exactness properties the gates rely on: BE reproduces its discrete
relaxation formula exactly; the exponential schemes are exact for
relaxation under prescribed strain at any dt (ETD1 for piecewise
constant, the trapezoid for piecewise linear strain).

tau = infinity (an elastic region) is stored as itau = 0: h = 0 gives
e = 1, alpha = beta = 0, phi1 = 1, so m stays put (zero) and the
effective modulus is the unrelaxed one.  Relaxation weights lie in
(0, 1] at the nodes; pure GLL collocation never interpolates them, so
the AdGIA clamping lesson (14 Oct) does not bite here, but any future
interpolation of weights must clamp.

## The SLE reduction per step

With effective moduli the step's instantaneous response to a surface
load is characterised by per-dt effective Love numbers (h_eff, k_eff,
and the tidal and axial families), computed by the existing
`love_numbers` machinery on the effective material; the memory field
(the per-degree surface response (U, phi) to b_mem with zero load)
enters the existing generalised SLE as a ready-made response
coefficient pair.  The ocean and rotation feedbacks close by the
existing Picard/Anderson iteration.  After the step's load converges,
the (l, m)-resolved radial solutions give d^{n+1} at the nodes and the
state update completes the step.

## Scope of verification (David, 14 Oct 2026)

pyslfp carries its own analytic checks and benchmarks only: the
weight-algebra exactness tests, the elastic and relaxed limits, the
Wu-Peltier correspondence gate and the scheme-order gates.  The AdGIA
cross-comparison (the plan's WP-G4) is run from the AdGIA repository,
with pyslfp as the simpler, independently verified reference in its
benchmarking chain; no cross-benchmark scripts live here.

A further reference is available in ~/dev/sl3d: `gia1d` is Crawford
(2018)'s own 1-D viscosity GIA code (Fortran, CMake build), itself
carefully benchmarked against Mitrovica's standard viscoelastic
Love-number code; this library is intended to replace the need to go
back to such codes, and a viscoelastic Love-number comparison against
gia1d is the natural extra gate when wanted.

## Status at the end of the 14 October 2026 session

Implemented, all tests passing (220, none committed; David commits):

- `pyslfp/gia/weights.py` — the scheme weight algebra, ported exactly
  from AdGIA, vectorised; prescribed-strain exactness tests.
- `pyslfp/gia/rheology.py` — `Rheology`: branch data as nodal arrays;
  constructors `maxwell` (constant / callable / per-layer viscosity,
  so two-layer is the mapping form), `from_relaxation_time` (the
  tau(r) parameterisation) and `from_model` (planetmodel's `viscosity`
  field); TI refused where branches relax; `effective_moduli` by the
  isotropic perturbation rule; `long_term_material`.
- `pyslfp/gia/operator.py` — `ViscoelasticOperator`: per-degree
  sub-mesh state (M, R, S per branch, columns = independent RHS),
  nodal strain scalars, the memory force, flavour-cached effective
  `DegreeSystem`s, the three steppers, and the two-phase
  `start_step`/`ViscoelasticStep.finish` that the coupled loop
  interleaves with the SLE solve (`memory_surface`, `inertia_moment`,
  `effective_love_numbers`).  The l = 0 tide channel is the
  centrifugal (r/a)^2 forcing, matching `love_numbers.solver`.
- `pyslfp/gia/evolution.py` — `SeaLevelEvolution`: the frozen-C
  coupled loop; initialises with the UNRELAXED elastic SLE solve at
  times[0] (the right limit — without this a switched-on load costs
  the trapezoid its second order; measured 1.1, fixed to 2.0).
  Units bridge material <-> model scales at the boundary.
- `pyslfp/love_numbers/material.py` — `Material.with_moduli`.
- `pyslfp/physics.py` — generalised SLE grew the ready-response
  channels (`displacement_response`, `gravitational_potential_response`)
  and `inertia_moment_change` for the axial trace term.

Gates in `tests/gia/` and `tests/love_numbers/test_analytic.py`:
weight exactness; elastic and relaxed limits; Wu-Peltier
correspondence (worst 9.3e-4 at kappa/mu = 1600); scheme orders
(2.00 / 0.92 / 0.98); slaving/elimination consistency to 1e-10;
load-jump invalidation; Kelvin-sphere elastic closed forms for the
EXISTING solver (worst 1.3e-3, the compressibility floor);
frequency-domain steady-state against `planetmodel.frozen` (7e-5);
coupled-loop elastic fingerprint recovery, mass conservation,
second order, rotation coupling.

Theory note: `docs/theory/pyslfp_theory.{tex,bib,pdf}` (15 pp,
builds with latexmk), written by a delegated session and checked
against the code; cites Crawford et al. (2018).

Example: `tutorials/scripts/06_forward_gia.py` — a Laurentide-scale
cap melting over 10 kyr on isotropic PREM with a two-layer Maxwell
mantle and elastic lithosphere; maps of the final sea level change
and uplift (colour scales saturated so the far field reads) and the
uplift history at the cap centre, which keeps rising after the
melting stops.  The disc load is tapered in degree space against
Gibbs ringing, mass untouched.  Runs headless in ~25 s at degree 64
and is covered by the slow tutorial-script test.

## The second 14 October session (the train)

- **Performance.** The load and tide right-hand sides are rank one, so
  each flavour keeps one unit solve per degree and the steps add their
  responses by outer products; `finish` reuses the memory solve the
  SLE already triggered, so a coupled step costs ONE banded solve per
  degree.  The memory-force scatter is a precomputed sparse matrix.
  Per step: 0.63 -> 0.40 s at lmax 64, 2.95 -> 1.86 s at lmax 128;
  the remaining floor is the memory solve itself.  Accepted cost
  (David: "not a light calculation"); the Fortran code OpenMPs such
  loops, and threading the degree loop is the later lever.
- **Ice models.** `BaseIceModel` slimmed to the thickness-only
  contract (histories interpolate; a consumer asks for any date);
  `BaseIceSeaLevelModel` is the richer type for inversion products
  (ICE-NG) that also carry sea level, from which
  `EarthState.from_ice_model` reads a background state directly.
- **The driver.** `SeaLevelEvolution.run_ice_history(dates, ice_model)`
  takes dates in kiloyears before present, grounds the thickness
  change over the frozen ocean function, and dates its snapshots.
  `calibrate_initial_sea_level` recovers the initial sea level of a
  thickness-only history by the standard present-day iteration
  (SL(t0) <- observed - xi(present)); round-trip gated.
- **Rotation.** Long steps drove the transverse feedback unstable on
  the homogeneous test body; diagnosis: the per-step effective k_t2
  approaches the model's hydrostatic fluid limit, and the specified
  moments (Earth's) sustained less oblateness than that limit --
  the wrong side of the secular resonance.  Resolution per
  Yu, Al-Attar, Syvret & Lloyd (2025, GJI 240, eq. 15): the closure
  is quasi-static with SPECIFIED equilibrium moments, never computed
  from the spherical model.  Added: `fluid_tidal_number` (the
  zero-rigidity solid solve; Kelvin 3/(2(l-1)) to 1e-9 on homogeneous
  bodies, 0.9347 for isotropic PREM),
  `EarthModelParameters.with_hydrostatic_moments` (Mitrovica et al.
  2005: C - A = (1+delta) k_hyd Omega^2 a^5 / 3G, delta = 0.8 per
  cent default, margin delta/(1+delta) for any rheology), the
  tri-state `rotational_feedbacks` (True / "axial" / False) through
  the generalised SLE, and a stability guard in the evolution that
  refuses an unsustainable transverse feedback.  Verified stable with
  consistent moments at two step sizes.  Two same-session bugs in
  physics.py were caught by the new gates (the Picard-loop _rotation
  call missed the new kwargs; a variable clobber in the
  angular-momentum branch).
- Theory note gains "Rotational stability and the choice of moments"
  (citing Yu et al. 2025 and Mitrovica et al. 2005); rebuilt, 15 pp.
- The forward-GIA example now uses `EarthState.from_ice_model`; 17 s.

## The capstone: ICE-7G deglaciation, 26 ka to present

Isotropic PREM, two-layer Maxwell mantle (5e20 / 2e21 Pa s about
670 km) under the elastic lithosphere, ICE-7G at half-kiloyear steps,
full rotational feedback, frozen 26 ka shorelines.  At lmax 128 the
run takes 97 s single-core and lands where it should:

- ocean-mean sea level rise 26 ka -> present 133.2 m, against
  128.4 m from ICE-7G's own inverted record (4 per cent, with a
  crude viscosity and the linear equation);
- peak present-day uplift rate 13.6 mm/yr at 54 N, 74 W (southeast
  Hudson Bay; observed ~13 near Richmond Gulf); Gulf of Bothnia
  7.9 mm/yr (observed ~9).

Known caveat: the run cold-starts at 26 ka with no internal memory
(the instantaneous elastic state), so the first few kiloyears carry
the usual spin-up bias; standard practice starts a glacial cycle
earlier or repeats cycles.

### Next steps

- WP-G2 remainder: the Talbot inverse-Laplace synthesis gate if
  wanted (the steady-periodic gate already cross-checks the frozen
  path); the gia1d (sl3d) viscoelastic Love-number comparison.
- Glacial-cycle spin-up for production runs (start a cycle earlier,
  or repeat the deglaciation saw-tooth), removing the cold-start
  bias at 26 ka.
- DONE (14 Oct, after the restructure): migrating shorelines.
  `solve_nonlinear_equation` grew the response channels, the
  inertia-moment term and the tri-state rotation;
  `run_ice_history(..., shorelines="migrating")` solves the nonlinear
  equation per step, with every snapshot recording its ocean function
  -- the C(t) history the adjoint reads back.  Gates: migrating equals
  frozen when no shoreline can move (which caught a units slip in the
  step size -- the per-step fields match the engine exactly); the
  melting cap floods and the water budget balances its grounded loss;
  the elastic limit reproduces the elastic nonlinear solver date by
  date.  Note: under migrating shorelines the per-step slaving is
  consistent to the nonlinear iteration's tolerance, not eliminated
  exactly; and the convergence-order gate stays on the frozen path,
  where the discretisation is smooth.
- Next: the ice-history adjoints -- see
  docs/planning/gia_adjoint_plan.md (Yu et al. 2025 basis, the
  discrete transpose, the stored-C(t) design and the gates).  The
  per-step transpose derivation goes in these notes first.
- Performance beyond the rank-one and sparse-scatter wins: thread the
  per-degree loop (the Fortran precedent OpenMPs it; pyslfp/parallel
  is the natural home).  `effective_love_numbers` still assembles its
  own `DegreeSystem`s once per flavour — minor, share if it ever
  matters.
- WP-G5: migrating shorelines; the adaptive trapezoid.
- The version-3 rearrangement (core / elastic / gia with the SLE
  engine in core) as discussed on 14 Oct; nothing written today
  fights it.

## The adjoint step: the discrete transpose

The reverse-mode of one viscoelastic step, derived from the code's own
update equations; the dot-product gate <F x, y> = <x, F^T y> at machine
precision is the arbiter of every factor below.

### The radial pieces

Per degree, with K the (symmetric) effective or unrelaxed system, D the
strain map, W the nodal weight diag(mu_k wq w_c) with w_c in
(k^2(k^2-2), 4/3, k^2), E/A/B the diagonal scheme weights, P the
surface-value rows and q the l = 0 inertia-moment row:

    forward:  m~ = E m + A d-,   x_m = K^{-1} D^T W m~,
              (U_m, phi_m, J0) = (P x_m, q . x_m),
              x = x_m + x_sigma sigma^T + x_psi psi^T,
              d+ = D x,   m+ = m~ + B d+.

    reverse:  x-dagger   = D^T (B m+^dagger + d+^dagger),
              sigma-dagger = x_sigma . x-dagger,  psi-dagger = x_psi . x-dagger,
              x_m-dagger = x-dagger + P^T (U_m, phi_m)-dagger + q J0-dagger,
              m~-dagger  = m+^dagger + W D K^{-1} x_m-dagger,
              m-dagger   = E m~-dagger,   d-^dagger = A m~-dagger.

One symmetric solve per degree per reverse step.  D^T is the
memory-force assembly with unit weights (M = D^T W exactly, same GLL
quadrature); P^T places the dual values at the surface dofs; the unit
solutions x_sigma, x_psi are the flavour cache's.

### The sea-level piece

The engine's generalised solve (frozen C) is, symbolically, the load
fixed point l = A[f + rho_w C S xi(l)] with, per evaluation,

    xi_raw = -(H + K/g) l - (s_u + s_phi/g),
    the rotation block (explicit scalars at degrees 0 and 2),
    the conservation projection x -> x + (mean - <C, x>/A_O),

and the outputs u = H l + s_u, phi = K l + s_phi with their degree-2/0
rotational replacements.  Every piece is one of: a diagonal Love
multiplication (self-transposed per degree), a multiplication by the
fixed ocean function (self-transposed in the grid pairing), the
synthesis/analysis pair (mutual transposes up to the quadrature
weights), a rank-one projection, and the closed-form rotation scalars
(transposed directly).  The adjoint is therefore implemented as the
exact reverse-mode of the same code path -- a transposed fixed point in
the load dual, run through the same Anderson mixer -- returning the
duals of ALL the step's inputs: the direct load, the two response
channels, and the inertia moment.  The house channel-swap realisation
(linear_operators) exposes only the direct-load dual, which is why the
engine grows an explicit adjoint method instead.

Under migrating shorelines the forward per-step map is the nonlinear
solver; its linearisation about the converged state has the same
structure with C(t) read from the stored history, which is why the
adjoint stays linear (Yu et al. 2025, Section 3.3).

## Storage

State per branch: three scalars x (l, m) pairs x radial nodes, real
coefficients in the pyshtools layout (2, lmax+1, lmax+1) per nodal
(nspec, ngll) field, ordered so each degree-l solve is one banded
back-solve with the orders as right-hand-side columns.
