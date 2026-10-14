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

## The approach: the discrete transpose

The DISCRETE adjoint, the exact transpose of the time-stepped forward
map, not a discretisation of the continuous adjoint equations -- the
house style (the elastic adjoint is discrete-exact; AdGIA's strain
projection exists to make the step's adjoint the transposed step).

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
