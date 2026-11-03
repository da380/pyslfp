# The response to a load, and horizontal displacement

## Decisions

1. **One object per physical quantity.** `GRDResponse` is the response of the
   sea level equation to a load: the sea level change xi, the surface
   displacement, the gravitational potential change phi and the angular
   velocity change omega. The elastic solvers, the fingerprint operator and
   the viscoelastic snapshots return it. Derived quantities (sea surface
   height, geoid height, centrifugal potential, Stokes coefficients) are
   methods or operators, not members.
2. **The displacement is one field.** `SurfaceDisplacement` holds
   u = U r_hat + grad_1 V, a spheroidal field: loads have no toroidal part, so
   there is no rigid rotation, and degree 1 is in the centre-of-mass frame.
   V has no degree 0. It offers the vertical part, the horizontal part as
   east and north components on a grid or at points, a projection onto a line
   of sight, and later frame changes.
3. **Optional members.** The horizontal part, which costs an adjoint channel,
   and omega, present only with rotational feedback. xi, U and phi are always
   present.
4. **Spectral storage.** Every member is band-limited: U, V and phi are
   Love-number multiples of the load's coefficients, and xi is a synthesis of
   degree-L coefficients plus a uniform shift, the ocean function entering
   only the load. Members are held as coefficients in pygeoinf's component
   convention for its sphere spaces (the radius times the orthonormal
   coefficients), so a member's space is the matching pygeoinf Sobolev space
   in component form and moving between them is the identity on components.
   Grids are made on demand, for display and for products with masks.
   Helpers carry members into pygeoinf's Sobolev spaces.
5. **Hilbert spaces.** A space for each class, wrapping pygeoinf rather than a
   bare `DirectSum`. The displacement space carries the vector-field Sobolev
   norm sum_l lambda_l (|U_lm|^2 + l(l + 1)|V_lm|^2), so U and V are weighted
   as parts of one field. The response space is built from its members'
   spaces, with projections and inclusions by name, and its vectors are
   `GRDResponse` objects.
6. **Only physical observables.** Horizontal displacement is observed as
   grad_1 V (east and north at points, integrals of the field); V itself is
   never point-evaluated.

## Horizontal displacement: forward and adjoint

The tables carry the tangential numbers: with the total load sigma, the load
channels zeta^u and zeta^phi and the centrifugal potential psi,

    V_lm = l_l sigma_lm + l^u_l zeta^u_lm + l^phi_l zeta^phi_lm
           + l^t_2 psi_2m (degree 2),

the same path as U with l in place of h. V feeds nothing back into the sea
level equation, so the forward solve gains one product per coefficient.

For the adjoint, a dual s on the coefficients of V gives the functional
J = sum s_lm V_lm. With d = l s, J = <d, sigma> + <l^t s_2, psi>. The total
load is sigma = zeta + rho_w C xi, so

    <d, sigma> = <d, zeta> + <rho_w C d, xi>,

and the dual on the direct load is d plus the existing fingerprint adjoint
applied to the sea level dual rho_w C d. The centrifugal term is
psi = P(omega), with P the centrifugal potential operator, so it adds the
angular velocity dual P*(l^t s_2). No new solver channel is needed for the
elastic adjoint; the tangential-traction numbers (h^v, l^v, k^v) are its
reciprocal statement and serve as the check. The viscoelastic adjoint needs
the memory field's horizontal part as well.

On grids and at points, grad_1 V comes from ducc0: `synthesis_general` with
spin 1 and E-mode coefficients sqrt(l(l + 1)) a_lm gives (d/dtheta,
(1/sin theta) d/dphi) at any points, and `adjoint_synthesis_general` is its
exact adjoint, m > 0 coefficients counting twice for a real field.

## The viscoelastic response (package 5)

### Snapshots

A `SeaLevelSnapshot` holds the response at its time as a `GRDResponse`,
`response`, in place of the grids `relative_sea_level_change`, `displacement`
and `gravity_potential_change` and the array `angular_velocity_change`. It
keeps `t`, `date`, `ocean_function` and the recorded trajectory
(`internal_state`, `load_coefficients`, `tide_coefficients`). Fields are read
by name, `snap.response.field("relative_sea_level_change")`, and the
displacement as a whole is `snap.response.displacement`.

The runs (`run`, `run_adaptive`, `run_ice_history`) take `space`, a
`GRDSpace` over the model's sphere, as `solve_response` does: it holds the
angular velocity change exactly when there is rotational feedback, and the
horizontal part when that is wanted. None gives the L2 space without the
horizontal part. The space sets only the inner product, since the components
are the same in every space over the sphere, so a Sobolev response space for
an inversion costs nothing in the forward run.

### The solver

Each step solves through the coefficient engine, and the response is built
from its coefficients with no grid in between:

- `SeaLevelEquation.solve_response` takes the generalised channels beside the
  direct load: the ready responses `displacement_response`,
  `gravitational_potential_response` and the new `horizontal_response`, with
  `inertia_moment_change` and `ocean_function`.
- `SeaLevelEquation.solve_nonlinear_response` solves the nonlinear equation
  and returns the final state and a `GRDResponse`; `solve_nonlinear_equation`
  wraps it and keeps its grids. The sea level change's coefficients are those
  of the last iteration with the eustatic shift added at degree 0, and V is
  l times the final load, as in the linear engine.

### The horizontal memory field

The memory state's response holds V beside U and phi, read at the surface V
dof by `step.memory_surface("V")`. It enters the step's V additively, through
`horizontal_response`, as the memory's U enters U:

    V = l~ sigma + l~^t psi_2 + V_mem,

with l~ the step's effective tangential numbers, sigma the converged total
load and psi_2 the degree-2 centrifugal potential. V feeds nothing back into
the sea level equation, so the solve gains one product per coefficient. The
check is the radial solution itself: `step.finish` returns the degree
solutions consistent with the new state, and their surface V must equal the
snapshot's to the solver's tolerance.

### Measurements

A measurement may be a `GRDResponse` g of any `GRDSpace` over the model's
sphere, standing for J = <g, response(t_i)> in that space. This is how the
horizontal part is observed, and how the adjoint of a pygeoinf observation
operator hands its dual to the kernels. With lambda the space's metric and
c the components of the response, J = sum lambda g c, so the duals the
engine takes are the metric-weighted components:

- the scalar members as grids paired by the surface integral, the duals the
  engine takes now;
- V as coefficients: its components are c = b V_lm and its metric is
  lambda_l l(l + 1), so s_lm = b lambda_l l(l + 1) g_lm, with
  J_V = sum s_lm V_lm;
- the angular velocity change as it stands.

`Measurement` keeps its grid duals, the L2 form of the scalar members, and its
fields take the members' names: `relative_sea_level_change`, `displacement`,
`potential_change` and `angular_velocity_change`. A bare grid is still a sea
level dual.

### The discrete adjoint

`adjoint_generalised_solve` takes `horizontal_dual`, the coefficient dual s
with J += sum s_lm V_lm. In the engine V = l sigma + V_static + l^t psi_2, so
s adds l s to the dual load read off the outputs and l^t s_2 to the dual of
the degree-2 centrifugal potential. The dual of `horizontal_response` is s
itself, so the engine needs no new output. In `adjoint_ice_history` the
memory's V is the read at the surface V dof, so s joins the radial right-hand
side as `surface_dual("V", s)` beside U and phi, with the length's factor
between the model's units and the material's.

### The continuous adjoint

The jump realises the adjoint as a forward problem, by reciprocity, as it does
for the other duals (zeta^u = -g_u, zeta^phi = -g g_phi). For V the forward
channel is a tangential traction, whose force per unit amplitude is
`DegreeSystem.tangential()`, -g a^2 l(l + 1) on V'(a). Matching the scaling of
the U channel, a dual s has the amplitude

    zeta^v_lm = -s_lm / (b^2 l(l + 1)),

which for an L2 dual is minus the V member as a field, as zeta^u is minus the
U dual. The engine gains it as `tangential_load`, answered through the
table's tangential numbers (h_v, l_v, k_v), so the instantaneous sea level
and rotation see it; the radial jump adds the unrelaxed solution of the
tangential force, a third column of `split_unit_responses`. Its sea level
static term is -(h_v + k_v / g) zeta^v = l s / b^2, since h_v = l(l + 1) l_u
and k_v = g l(l + 1) l_phi; this is the direct term d / b^2 of the elastic
adjoint above, and the fixed points of the two forms agree.

The fingerprint operator's adjoint moves onto the same channel, in place of
`_horizontal_dual`, which carries the V dual onto the sea level and rotation
duals by hand. Its existing dot-product check then tests the channel.

### Tests

- Forward: the V of every snapshot equals the surface V of the step's radial
  solution, for frozen and migrating runs, with and without rotation. With an
  elastic rheology every snapshot equals `solve_response` of the load,
  horizontal part included.
- Solver: the dot-product identity of `horizontal_dual` against the forward
  engine with V, with and without rotation; `solve_nonlinear_response`
  against the grids of `solve_nonlinear_equation`; the fingerprint operator's
  check with the horizontal part, after the change of channel.
- Kernels: `test_full_measurement_kernels` with a `GRDResponse` measurement
  holding a horizontal part, in L2 and in a Sobolev space: the discrete
  gradient identity to 1e-8 and the continuous one in the resolved regime. The
  instantaneous kernels of the two methods agree to the solver tolerance,
  both being the elastic adjoint.

### Steps

1. The solver: `solve_response` with the generalised channels,
   `horizontal_response`, `tangential_load`, `solve_nonlinear_response` and
   `horizontal_dual`; the fingerprint adjoint on `tangential_load`.
2. The snapshots and the horizontal memory field: the runs, the code that
   reads snapshots, tutorials 06, 07 and 09, and the tests.
3. The adjoints: `GRDResponse` measurements, the renamed `Measurement` fields,
   and V in the discrete and continuous adjoints.
4. The docs: the viscoelastic adjoint problem in theory.rst, release_4.md
   and this file.

## Work packages

Packages 1 to 4 are built: the solver's coefficient engine with V
(`SeaLevelEquation._solve_generalised`, and `solve_response` returning a
`GRDResponse`), `pyslfp.core.response`, the fingerprint operator onto a
`GRDSpace` with the horizontal part on request, and its consumers (sea surface
height, GRACE, tide gauges, the altimetry models), which reach the fields by
name through `GRDSpace.field_operator`.

1. **The solver.** Horizontal displacement as a coefficient output of the
   elastic sea level solvers (linear and non-linear), with a test against
   the Love-number definition; the existing grid-returning methods kept.
2. **The classes.** `SurfaceDisplacement`, `GRDResponse` and their spaces,
   with the conversions to pygeoinf Sobolev spaces and grids, and the ducc0
   point and grid synthesis of grad_1 V with its adjoint.
3. **The fingerprint operator** returns a `GRDResponse`, horizontal part on
   request, its adjoint as above; `check_operator` on both parts.
4. **The consumers** reach members by name: sea surface height, GRACE, tide
   gauges, altimetry, the models' observables. The index-based response tuple
   goes.
5. **The viscoelastic side:** snapshots hold a `GRDResponse`; the horizontal
   memory field and the adjoints. The design is set out above.
6. **GNSS observation models**, with the frame of the data: frame (Helmert)
   parameters as nuisances, or frame-invariant data.

WP3 of the models submodule follows packages 1 to 4, so that the new scripts
are built on the named response.
