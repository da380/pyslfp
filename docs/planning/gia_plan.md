# GIA forward modelling in pyslfp: the plan

Drafted 14 October 2026 from the design discussion with David; this
document is the brief for the implementing session. Status: plan,
awaiting review; nothing below is implemented.

## Goal

Extend pyslfp from elastic sea-level fingerprints to full forward GIA
modelling: viscoelastic (generalised Maxwell) loading of a spherically
symmetric Earth, coupled to the sea level equation and rotational
feedback, driven by an ice history, integrated in time. Forward
modelling only — no adjoints, no generalised spherical harmonics, no
lateral heterogeneity. Each spherical-harmonic degree decouples, which
is the whole computational economy.

The second purpose is reciprocal benchmarking with the AdGIA C++
library (`~/dev/AdGIA`): a fully independent discretisation (1-D GLL
spectral elements per degree + spherical harmonics, against 3-D finite
elements) of the same continuum formulation with the same time-stepping
algebra, so that disagreements isolate spatial discretisation cleanly.

## Settled design decisions (David, 14 Oct 2026)

1. **Internal variables, not the rate formulation.** The theory basis
   is Al-Attar & Tromp (2014) — the internal variable m with
   ṁ = (d − m)/τ and stress T = κ∇·u I + 2μ₀(d − m) (their eqs
   2.29–2.32) — but NOT their (or Crawford et al. 2018's) literal
   numerical approach. No rate formulation.
2. **Slave the elastic problem to the internal-variable ODE.** m is
   the state; u (and φ, and the sea level) are slaved: evaluating the
   ODE right-hand side, or a scheme's implicit/exponential update,
   contains the quasi-static degree solves. The steppers see only an
   ODE in m. This is the AdGIA `ViscoelasticOperator` architecture
   transplanted; its dividends carry over: m-only state (checkpoints,
   restarts), stepper-agnostic core, and the unrelaxed/effective
   operator flavour pair with factorisations cached per Δt.
3. **Time-step with the AdGIA scheme family**: exponential trapezoid
   (default), backward Euler, exponential Euler; the adaptive
   trapezoid later if wanted. The effective-weight algebra is ported
   exactly (see "AdGIA context" below).
4. **Branch-general (Prony) from day one**, not Maxwell-only: the
   weight algebra is branch-wise identical, and SLS/Burgers parity
   with AdGIA's test families comes free. A Maxwell mantle is the
   one-branch, μ_∞ = 0 case.
5. **Spherically symmetric, forward only** (restated for the record).
6. **Frozen shorelines first** (the linear SLE path); the migrating
   ocean function is a later rung, mirroring the AdGIA ladder.
7. **Rheology data lives on planetmodel models** through planetmodel's
   own flexibility (add fields to a model, or a class with additional
   fields); all the work stays in pyslfp, planetmodel itself is edited
   only if a real need arises.
8. **Rheology convenience constructors**: constant viscosity;
   two-layer viscosity; and the constant (or piecewise-constant)
   RELAXATION TIME parameterisation — τ(r) prescribed, η = μ τ
   inheriting its depth dependence from the shear modulus. The last is
   arguably more physical and has not been done before; treat it as a
   first-class option, not a toy.
9. **Naming**: `pyslfp/gia/`, at least initially; this work is a
   large enough extension that a rearrangement may come with a
   version-3 release.
10. **Frame**: centre of mass, fixed through time (the existing l = 1
   dof removal). Cross-code comparisons are made on physically
   meaningful quantities, never on frame-dependent raw fields.
11. **Rotation feedback from day one**, and — with the per-degree
   elastic solves practically free — the feedbacks (rotation, ocean)
   may close by SIMPLE Picard iteration inside the slaved solve. No
   bordered-system machinery is needed on this side; that economics
   belongs to the 3-D code.
12. **Ice histories provide values at arbitrary times; how is their
   business.** The ice-ng classes already interpolate linearly; the
   time loop consumes the interface, aligning its step grid with any
   breakpoints the history declares.
13. **lmax**: 256 is the sensible global-scale target, but development
   and gating run much lower — the library's idealised test
   configurations are to be adapted to the GIA problem for low-lmax
   testing.

## What already exists (do not rebuild)

- `pyslfp/love_numbers/assembly.py` — the degree-l system IS
  Al-Attar & Tromp (2014) eq. (D76): transversely isotropic elastic
  block in the variables X = r∂ᵣU, Y = 2U − k²V, S = U + r∂ᵣV − V, V;
  fluid regions as potential-only with the stratification term and
  fluid–solid couplings; DtN closure; l = 0 and l = 1 special cases
  (centre-of-mass frame, axial numbers). Banded symmetric Cholesky.
- `pyslfp/love_numbers/material.py` — `NodalModuli` takes per-node
  arrays, so per-step effective moduli μ_eff(r) cost nothing new; the
  complex "frozen at a frequency" path gives a frequency-domain
  verification route.
- `pyslfp/love_numbers/solver.py` — per-degree solves for the load,
  tide and centrifugal forcings; the Love-number table conventions.
- `pyslfp/core.py` / `physics.py` — the SLE with Anderson mixing,
  rotational feedback via the axial numbers, the operator layer.
- `pyslfp/ice` — ice histories.

## Architecture

New subpackage `pyslfp/gia/` (name to David's taste):

- **State.** m per branch, per (l, m) pair, per radial node, per
  deviatoric component in the degree basis. Storage order chosen so
  the per-degree solves are multi-RHS banded back-solves vectorised
  over m (one factorisation per l, BLAS-friendly). Order-of-magnitude
  memory: lmax 256 ≈ 240 MB, lmax 128 ≈ 30 MB.
- **The slaved solve.** Given m and the loads at time t: assemble the
  degree systems with unrelaxed or effective moduli, add the
  internal-variable force (the μ-part of the existing quadratic form
  applied one-sided to m), solve all degrees, and close the sea-level
  coupling INSIDE the solve. The SLE reduction: with effective moduli
  the step's instantaneous response is characterised by per-Δt
  EFFECTIVE Love numbers, so the existing `SeaLevelEquation`
  machinery runs unchanged with (h_eff, k_eff) plus a memory field
  (the per-degree response to m with zero load) folded into the
  direct-effect term. The ocean and rotation feedbacks close by plain
  Picard iteration inside this solve (decision 11): per-degree
  back-solves are practically free, so the 3-D code's bordered-system
  and elimination economics simply do not apply here.
- **The ODE layer.** Mirroring AdGIA's `ViscoelasticOperator`:
  `Rate(m, t)` for explicit/exponential-Euler use, plus the
  per-scheme updates (BE weight 1/(1 + Δt/τ); exponential trapezoid
  weights e = e^(−Δt/τ), α = φ₁ − e, β = 1 − φ₁ with
  φ₁ = (1 − e^(−Δt/τ))/(Δt/τ), series-switched at small argument —
  port `detail::ExponentialTrapezoidWeights` exactly, including the
  effective operator weight 1 − β = φ₁).
- **The internal-variable components in the degree basis are already
  in the paper** — no new derivation needed. Al-Attar & Tromp (2014)
  Appendix D, eqs (D68)–(D73): the spheroidal subsystem's internal
  variables are the three scalars (M_lm, R_lm, S_lm) per (l, m), with
  ODEs driven by the degree-basis deviatoric strain scalars (e.g.
  r∂ᵣU − U + k²V and r∂ᵣV − V + U), and eqs (D75)–(D76) give the
  spheroidal weak form with the memory force. Transcribe, checking
  factors against the assembled quadratic form; the ODEs must also be
  rederived once for the GENERALISED Maxwell case (per-branch M, R, S
  with μ_∞ and branch moduli — the branch-wise structure is AdGIA's,
  the components AT14's). The toroidal subsystem (D69) is skipped:
  surface loading and the centrifugal perturbation are spheroidal.
  The strain map evaluates nodally (GLL collocation — the analogue of
  AdGIA's interpolation strain map).

## Work packages and gates

- **WP-G1 — the radial viscoelastic core.** The state, the strain
  scalars, the slaved degree solves with effective moduli and
  m-forcing, the flavour-pair factorisation cache, the three
  steppers. Gate: a homogeneous Maxwell sphere against the
  Wu–Peltier / correspondence-principle closed form (the instrument
  of AdGIA's `TestAdjointLove`); backward-Euler discrete relaxation
  exact at its known formula; exponential schemes exact for
  relaxation under prescribed strain at any Δt.
- **WP-G2 — degree-wise verification.** The elastic limit (τ → ∞, or
  t → 0⁺) must reproduce the existing Love-number table to digits;
  the fluid-core and l = 0/1 paths unchanged. Frequency-domain
  cross-check: viscoelastic Love numbers synthesised from the
  existing complex frozen-moduli solves via numerical inverse Laplace
  (Talbot; the AAA pitfalls at extreme contrast are recorded in
  AdGIA's box-benchmark docs) against the time-stepped h_l(t).
  ALSO (David, 14 Oct): pyslfp's Love numbers themselves have never
  been gated against analytical models — build that into this rung:
  elastic Love numbers against the homogeneous/incompressible-sphere
  closed forms (the instruments of AdGIA's `love_benchmark` analytic
  gates), and viscoelastic h_l(t) against Laplace-domain closed forms
  via the correspondence principle (Wu–Peltier; the constant-free
  construction in AdGIA's `TestAdjointLove` shows how to make such a
  gate normalisation-independent). This hardens the EXISTING library,
  not just the new code.
- **WP-G3 — the coupled time loop.** Ice history in, frozen-C SLE
  per step via effective Love numbers + memory field, rotation
  feedback through the existing axial machinery, outputs (sea level,
  displacement, potential histories). Gate: step-convergence orders
  per scheme (ExpTrap 2, BE 1) against a fine-step reference;
  mass conservation; the elastic fingerprint recovered when every
  τ → ∞.
- **WP-G4 — the AdGIA cross-benchmark.** Viscoelastic Love-number
  histories against AdGIA's `love_benchmark` viscoelastic family;
  a deglacial SLE-coupled run against `ice_age_loading`. Recorded on
  the AdGIA side in `doc/benchmarks.tex`; this closes AdGIA's
  known gap of having no independent viscoelastic reference, and
  retires the "pyslfp unverified as derivative reference" caveat.
- **WP-G5 (later) — migrating shorelines**, mirroring the AdGIA
  ladder; and the adaptive stepper if the fixed-step costs annoy.

## AdGIA context for the implementing session

Read before writing code:

- `~/dev/AdGIA/doc/viscoelasticity.md` — the internal-variable
  formulation, the scheme derivations, the exactness properties the
  gates rely on.
- `~/dev/AdGIA/src/viscoelastic.cpp` — `ExponentialTrapezoidWeights`
  (top of file), `SetEffectiveModulus`, `Rate`,
  `LocalExponentialUpdate`, and the `ElasticUpdate`/`ImplicitSolve`
  slaving pattern to mirror. NOTE the fresh lesson (14 Oct, see
  `doc/planning/solvers.md` "Resolved"): nodal relaxation weights lie
  in (0, 1], but anything that INTERPOLATES them (not an issue for
  pure GLL collocation, but keep it in mind wherever τ(r) varies by
  decades within an element) can overshoot the bounds; AdGIA now
  clamps at evaluation.
- `~/dev/AdGIA/doc/BenchmarkPapers/alattar2014.pdf` §2.2 (internal
  variables) and Appendix D — in particular eqs (D68)–(D76), which
  hold the degree-basis internal-variable components, their ODEs and
  the weak form the implementation transcribes; `Crawford2018.pdf`
  only for what we are NOT doing (the rate formulation).
- AdGIA benchmark conventions for the cross-checks:
  `benchmarks/love_numbers/`, `examples/ice_age_loading.cpp`.

## Deferred (deliberately unsettled)

- The central operator class's name (within `pyslfp/gia/`).
- Whether the per-Δt effective Love numbers get a public API (they
  are useful in their own right; default internal until wanted).
- Where the cross-benchmark scripts live (pyslfp tests/tutorials,
  AdGIA benchmarks/, or both ends of a shared data format).
- Migrating shorelines and the adaptive stepper (later rungs).

Everything else raised at the design stage was settled on
14 October 2026 and is folded into "Settled design decisions" above.
