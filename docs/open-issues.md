# Open issues

Precise, reproducible items found from the outside (sl3dpp's stage-5
cross-code gates) and parked here for someone to pick up. Each carries a
diagnosis, a minimal repro and a concrete fix plan.

## 1. [FIXED in pyslfp — sl3dpp regeneration outstanding] Continuous-adjoint itau kernels blow up (~1e31) on layered Maxwell rheologies

**Resolution (16 Oct 2026).** The prime suspect below was wrong: the
branch support is clean (no tiny-positive itau nodes; the smallest
positive itau on the stage-3 body is within a factor 4 of the branch
maximum).  The real cause is a units-covariance error in
`continuous_adjoint_ice_history`: the adjoint radial state seeded by
`apply_jump` drives the forward unit-response machinery, whose load
pairing lives on the material's surface (g a^2, material units), while
the functional pairs its dual with sea level on the model's surface
(r^2, model units).  The ratio of the two pairings carries dimensions
mass x length, so it is unity on a non-dimensionalised body (every
test body) and ~9.1e30 on an SI one (the stage-3 deck material is SI).
Layering was a red herring: the same body with an elastic lid in nd
units agrees with the discrete kernel at the resolution level.
Fixed by scaling `accumulate_itau` with the pairing ratio; verified
covariant to 2e-12 across four unit systems, and the stage-3 repro now
converges to the discrete kernel at second order (2.7e-2 / 7.2e-3 /
1.8e-3 at 6 / 11 / 21 nodes).  Regression gate added
(`test_continuous_viscosity_kernels_unit_covariance`, plus a finite
ceiling on the stiff branch).

**Remaining:** step 4 only — flip `with_itau` to True for the
continuous method in `sl3dpp/scripts/make_stage5_fixtures.py`, delete
its stale comment pointing here, regenerate the fixture, and run the
sl3dpp stage-5 gate (compares at 1e-5).

---

Original report:

**Found:** 16 Oct 2026, while generating sl3dpp's stage-5 kernel fixture
(`sl3dpp/scripts/make_stage5_fixtures.py`).

**Symptom.** `SeaLevelEvolution.continuous_adjoint_ice_history(...,
internal_states=...)` returns `itau` kernels of order **1e31–1e32**
(model units) on a PREM-deck Maxwell body whose viscosity lives on the
mantle layers only (fluid core; viscous 3480–6151 km; elastic above),
lmax 16, GLQ grid, the analytic test state, 10 kyr histories. In
identical runs `adjoint_ice_history` (the exact discrete transpose)
returns O(1)–O(20) kernels, and its FD/Richardson gradient identity is
clean.

**Key facts the diagnosis must honour:**

- The magnitude is **independent of eta** (measured at 1e21 and 1e22 —
  so not a stiffness/exp(sum h) amplification),
- **independent of the measurement type** (a sea-level, displacement,
  potential or angular-velocity dual alone each reproduce it),
- and **independent of the node count — a single step suffices**
  (max |K| is ~constant from 2 nodes to 6: 1.5e32, 1.4e32, 1.4e32,
  1.4e32), so it originates at the observation jump / equilibrium
  accumulation stage, not in the reversed stepping.
- The kernel's argmax sits at flat node index 837 of 880 (ngll = 5 →
  element 167, r ≈ 6050–6080 km): **near the upper edge of the viscous
  region** (6151 km).
- The formulation is not at fault: sl3dpp's independent implementation
  of the same continuous algebra agrees with its own exact discrete
  kernel at the per-cent level in the resolved regime and converges
  under step refinement, on this same body.

**Why the test suite cannot see it:**
`test_continuous_viscosity_kernels` checks the resolved regime with
`Rheology.from_relaxation_time` — a **uniform tau over the whole
body** — and its stiff branch only asserts `disagreement > 0.2`, which
is unbounded above; a 1e31 kernel passes it.

**Minimal repro** (the sl3dpp fixture setting; `ETA = 1e21` or `1e22`,
`LMAX = 16`):

```python
# Build the stage-3 layered Maxwell body (see
# sl3dpp/scripts/make_stage5_fixtures.py for the full construction:
# PREM deck -> Model -> RadialMesh(lmax=16, ngll=5) -> Material,
# Rheology.maxwell(material, {viscous mantle layers: 1e22}),
# LoveNumbers.from_model, EarthModel(16, grid="GLQ", extend=False),
# the AnalyticalIceModel state).
evolution = SeaLevelEvolution(op, state)
dates = np.linspace(10.0, 0.0, 2)          # ONE step suffices
times = (dates[0] - dates) * kyr           # kyr in model time units
snapshots, _ = evolution.run(times, ice_load, rtol=1e-12, record=True)
m = {0.0: g}                               # any smooth sea-level dual
kd = evolution.adjoint_ice_history(dates, m, internal_states=snapshots)
kc = evolution.continuous_adjoint_ice_history(dates, m,
                                              internal_states=snapshots)
print(np.max(np.abs(kd.itau[0])))          # O(10)
print(np.max(np.abs(kc.itau[0])))          # O(1e32)  <-- the bug
```

**Prime suspect.** In `continuous_adjoint_ice_history`'s
`accumulate_itau`, the kernel density divides by the branch itau:

```python
inv = np.where(itau_k > 0.0, itau_k, 1.0) * g_mat
g = np.where(itau_k > 0.0, g / inv, 0.0)
```

The strict `> 0` guard is only safe when the branch support is exact
zeros. A layered `Rheology.maxwell` builds nodal `mu` (and `itau =
mu / eta`) from the model's layer fields; any node where itau comes out
**positive but tiny** (edge nodes of the viscous region, spline
evaluation at a boundary) passes the guard and the division explodes —
consistent with the argmax at the region's upper edge, the
eta-independence (a tiny itau scales with 1/eta, and the jump that
seeds m-dagger scales with itau, cancelling), and the uniform-tau test
gap. The jump `m-dagger += itau * d-tilde` at the same nodes is bounded
and less likely. This is a hypothesis to verify first, not a
conclusion.

**Fix plan:**

1. Instrument: in the repro, print per branch `min(itau_k[itau_k > 0])`
   over the degree workspaces and max |weighted * dm| in
   `accumulate_itau`; confirm whether the explosion is the division at
   tiny-positive itau nodes or something else (e.g. the jump, or
   `g_mat`).
2. If tiny-positive nodes: make the branch support definition exact —
   either threshold the guard physically (`itau_k > floor` with a floor
   relative to the branch maximum; the continuous density's division by
   itau is only meaningful where the branch genuinely relaxes), or fix
   `Rheology.maxwell` to zero itau wherever it zeroes mu's support, so
   discrete and continuous agree on where the branch lives.
3. Add the missing regression gate: the layered Maxwell body (the
   stage-3 deck), resolved regime, asserting the continuous itau
   kernel's distance from the exact discrete one (e.g.
   `norm(Kc - Kd) < 0.1 norm(Kd)` at 11 nodes), and a finite ceiling on
   the stiff branch instead of the one-sided `> 0.2`.
4. Cross-check against sl3dpp once fixed: its stage-5 fixture's
   continuous block currently carries `itau 0` with a pointer to this
   issue; flip `with_itau` to True for the continuous method in
   `sl3dpp/scripts/make_stage5_fixtures.py`, regenerate, and the
   sl3dpp gate compares at 1e-5.

## 2. [FIXED in pyslfp — sl3dpp anchor removal outstanding] EarthModel scales deviate from planetmodel's `Scales.geophysical`

**Resolution (16 Oct 2026).** pyslfp changed, per the plan in
docs/planning/scales_plan.md: `EarthModelParameters.from_defaults` now
builds from `Scales.geophysical(MEAN_RADIUS)` (G = 1, T ≈ 1648 s), the
viscoelastic coupling refuses a material whose scales disagree with
the model's (models authored in other units convert at ingestion via
`Model.converted`), and two-frame covariance gates
(tests/test_scales_covariance.py, and the unit-covariance test in
tests/viscoelastic/test_evolution.py) hold the frame. The convention
is documented in docs/source/units.rst.

**Remaining:** sl3dpp's stage gates still carry the SI anchors for
potential and angular-velocity duals that worked around the old
mismatch; with both codes in one frame the anchors are redundant and
can come out when the fixtures are next regenerated (together with the
stage-5 itau regeneration of issue 1).

---

Original report:

Recorded for the convention unification David called for (16 Oct 2026:
pyslfp, AdGIA and sl3dpp should share one internal
non-dimensionalisation). pyslfp's `EarthModel` parameter scales use
T = 1/sqrt(G (4 pi / 3) rho_bar) ≈ 805 s (so g_nd ≈ 1), while
planetmodel's documented `Scales.geophysical` — which sl3dpp follows —
uses T = 1/sqrt(G rho_bar) ≈ 1648 s (so G_nd = 1). Lengths and
SI-converted outputs are invariant, but the non-dimensional potential
scales as T² (a clean factor 4 pi / 3 between the codes) and
frequencies as 1/T; this cost a day's debugging in sl3dpp's stage-5
gates, which now anchor potential and angular-velocity duals through SI
constants as a workaround. Whichever convention wins, the loser should
change; planetmodel's is the documented shared scheme.
