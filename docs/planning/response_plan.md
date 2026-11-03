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
   east and north components on a grid or at points; a projection onto a
   line of sight is for later, and the frame of GNSS data belongs to the
   observation model.
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
angular velocity dual P*(l^t s_2). This is the form the transposed engine
takes (`horizontal_dual` of `adjoint_generalised_solve`). Its reciprocal
statement is a forward solve with a tangential traction, answered through the
numbers (h^v, l^v, k^v); the fingerprint operator's adjoint takes that form
(`tangential_load`), as it takes a displacement dual through the traction
channel zeta^u. The viscoelastic adjoints carry the memory field's
horizontal part as well (package 5).

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

The fingerprint operator's adjoint takes the same channel, so its dot-product
check tests it; the two forms agree to the solver's tolerance in every
rotation mode.

### Sobolev kernels

The ice kernels are L2 objects. The node duals satisfy the gradient identity
on the date grid but lump the deltas at the observation times into their
values, so they do not converge pointwise as the grid is refined; the
distributional view separates the deltas, and the quelled view removes them by
changing the variable to the thickness rate. The proper treatment is the
gradient in a space-time Sobolev space: the Riesz representer of the
derivative.

The ice history I(t, x) is a vector of X = H^s(T) (x) H^r(S^2), with the inner
product (A_t^s (x) A_x^r u, v): A_t = 1 - L_t^2 d^2/dt^2 on the run's interval
of time, pygeoinf's interval Sobolev space, and A_x the sphere's Sobolev
operator. Both factors are diagonal in their eigenbases, so X is too: its
components c_kj are on products of the time eigenfunctions phi_k and the
spherical harmonics, with the metric mu_k nu_j.

The product metric makes the two orders independent: for s > 1/2 the history
at a time is a field of H^r(S^2), and evaluation at a time is bounded. An
isotropic space-time norm, (1 + kappa_k + lambda_l)^sigma on the same basis,
would lose half a derivative in space at each time instead. Histories are
continuous in time, since the loads change continuously over the times
considered, so a smooth time basis serves. The time space's mesh is padded at
both ends, so that no end condition marks the history inside its interval,
the present included.

The space is `FieldHistorySpace`, in core beside `GRDSpace`: pygeoinf's
`TensorProduct` of an interval space and a sphere space, whose vectors are
K x N arrays of components. A generic tensor product is possible only when
both factors expose their coordinates, as these do, and the product metric
is the natural one here; pygeoinf 2.2.0 provides it, with the contraction
that evaluates a history at a time, the lift of a map on the field space to
one acting at each time, and Gaussian measures with separable covariances. A time-local forward map, such as an elastic
fingerprint at each epoch of a GRACE series, commutes with the time basis:
K solves give the whole history whatever the number of epochs.

The forward map reads I at the date nodes, so its formal adjoint takes the
node duals. With d_n the thickness duals of the discrete adjoint, exact for
the discrete map with the deltas lumped in, the L2 dual has components
sum_n phi_k(t_n) d_n,j, and the gradient in X divides them by the metric. For
s > 1/2 evaluation at a time is bounded on H^s(T), so each delta becomes the
Green's function of A_t^s, continuous in time, and the representer converges
as the date grid is refined. Quelling is a one-sided form of the same idea:
the kernel of the rate, a cumulative integral, is the Green's function of
d/dt with a terminal condition.

In pygeoinf's terms the ice history operator maps X to the responses at the
observation dates. It is built by `LinearOperator.from_formal_adjoint` from
the L2 operator whose adjoint is the discrete adjoint, so the Sobolev kernel
of a measurement g is `operator.adjoint(g)`. This is the operator wrapping of
the viscoelastic adjoints, done for the ice history, and a time-dependent
model space for a surface field.

pygeoinf's interval Sobolev spaces (`pygeoinf.interval.Sobolev`) give the
time factor, and their `basis_matrix` the eigenfunctions at given times.

The ice history operator, in `pyslfp.viscoelastic`:

- Domain: a `FieldHistorySpace` of ice thickness, its time coordinate the
  model's non-dimensional time from the run's first date
  (`EarthModel.times_from_dates`), over an interval holding the run's dates.
- Codomain: a pygeoinf `DirectSum` of one `GRDSpace` per observation date,
  the dates on the run's grid.
- Forward: the history read at the run's dates, through an ice model that
  interpolates linearly between them (the `BaseIceModel` contract), and
  `run_ice_history` under frozen shorelines, where the map is linear.
- Adjoint: the duals at the observation dates as `GRDResponse` measurements
  to `adjoint_ice_history`, the exact transpose, whose thickness duals d_n
  give the history's components (Phi^T D) / (mu (x) nu), with D the rows of
  the d_n's components. The continuous adjoint is not an exact transpose, so
  it stays a diagnostic beside the operator.
- Migrating shorelines make the map nonlinear; they wait for the operator
  wrapping of the viscoelastic drivers as a whole.

Tests: the gradient identity in X, <g_X, delta I>_X = delta J for delta I in
X; the convergence of the representer as the date grid is refined at a fixed
X, where the node duals do not converge; and the agreement of the discrete and
continuous adjoints' representers as the run resolves its rheology.

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

### Where it lives

- The solver: `solve_response` with every generalised channel, the ready
  `horizontal_response`, the tangential traction `tangential_load`,
  `solve_nonlinear_response`, and `horizontal_dual` in
  `adjoint_generalised_solve` (`pyslfp.core.sea_level`); the fingerprint
  adjoint takes the tangential traction (`pyslfp.elastic.operator`).
- The snapshots, the horizontal memory field, the `GRDResponse` measurements
  and V in both adjoints (`pyslfp.viscoelastic.evolution`).
- `FieldHistorySpace` (`pyslfp.core.history`) and `IceHistoryOperator`
  (`pyslfp.viscoelastic.ice_history`).
- The layout that came with it: `linear_operators` dissolved into `core`,
  `elastic`, `observations` and `models`, with the altimetry observation
  models removed for `ModelSetup.geocentric_sea_level`.
- Tests: `tests/test_horizontal_displacement.py`, `tests/test_history.py`,
  `tests/viscoelastic/test_evolution.py` and
  `tests/viscoelastic/test_ice_history.py`.

## Observations and GNSS (package 6)

### The shape of an observation model

An observation model takes its observable as an operator from the model space
into the observable's own space, and adds what the measurement does: the
sampling, the network's metadata and the noise. The physics, direct terms
included, stays with whatever built the observable, so one model serves an
elastic set-up and a viscoelastic run alike.

- Tide gauges: the relative sea level change, a field, for example
  `setup.relative_sea_level_change(fingerprint)`; sampled at the stations.
- Altimetry: the geocentric sea level change, a field,
  `setup.geocentric_sea_level(fingerprint)`; sampled at the points, or
  averaged to a global mean.
- GRACE: the potential change, read as coefficients between two degrees, in
  pyslfp's convention or as Stokes coefficients.
- GNSS: the displacement, a `SurfaceDisplacement`; sampled east, north and up
  at the stations, in the frame of the data (below).

The model holds `observable`, `sampling_operator` (from the observable's space
to the data space), `forward_operator` (their composition), `data_space`, the
stations' `points` and `names`, and an optional `noise`: a Gaussian measure on
the data space, or standard deviations, one for every datum or one per datum,
for independent noise; `with_noise` gives the same model with another.
`from_fingerprint` builds each model on a fingerprint operator's response,
with no direct terms. `ModelSetup.response(fingerprint)`, the response to the set-up's load,
supplies the members with no direct terms (the displacement, the potential);
the sea level fields keep their own methods, which add the direct terms. A
viscoelastic observable at a date is a projection of the ice history
operator's codomain, and a rate the difference of two dates over their
interval.

GRACE reads the potential change's coefficients straight from the response's
components, `GRDSpace.projection("potential_change")` into the components of
the field space, so that no synthesis and analysis lie between the solver and
the data; an observable into the field space itself is still taken, at the
cost of one analysis. `WMBMethod` stays as it is, the spectral surrogate the
preconditioners use.

### GNSS and the frame of the data

The response's degree 1 is in the centre-of-mass frame. A GNSS solution's
frame differs from it by a Helmert transformation: a translation T, a small
rotation R and a scale s, or their rates for velocities. At a station x = b r
the transformation adds

    delta u = T + s x + R x x,

in east, north and up: the translation is a degree-1 spheroidal field with
V = U, the scale a degree-0 vertical field, and the rotation a rigid
toroidal motion, which no response holds.
`HelmertTransformation(points, radius, parameters=("translation", "rotation",
"scale"), directions=("east", "north", "up"))` is the design matrix, a
`MatrixLinearOperator` from EuclideanSpace(k) to the station data in the
order of the displacement's point evaluation, with `parameter_names`; the
translation is a length, the rotation in radians and the scale a ratio, in
Earth-fixed axes. `frame_reduction()` gives Z^T, held as the k Householder
reflectors of the design's QR factorisation, so that no dense Z is formed; a
design whose columns are dependent at the stations, or a network with no more
data than parameters, is refused.

`GnssObservationModel(observable, points, names=, directions=, frame=,
parameters=, noise=)` takes a displacement observable, into a
`SurfaceDisplacementSpace` with point values and with the horizontal part if
east or north is observed, and the frame in one of three ways:

- `frame=None`: the data are in the centre-of-mass frame.
- `frame="nuisance"`: the Helmert parameters join the model space, and the
  forward operator, a `RowLinearOperator`, acts on DirectSum([model space,
  EuclideanSpace(k)]) as F u + A h; `joint_prior(prior, frame)` joins a prior
  on h, a measure or standard deviations, stating how well the frame is
  known.
- `frame="invariant"`: the data are reduced to Z^T d, the data space has
  dimension 3N - k, the forward operator is Z^T F, and the noise Z^T C Z.
  This is the nuisance form with no prior information on h, without a
  singular covariance; station data enter as `frame_reduction(d)`.

The noise is given for the station data in every frame. `from_fingerprint`
takes the displacement of a fingerprint operator's response; a viscoelastic
rate is the difference of two dates of `IceHistoryOperator`'s codomain over
their interval.

### Tests

- Each model's forward operator is its sampling composed with its observable,
  and passes the adjoint check; the old forms are reproduced (tide gauges and
  GRACE from `FingerPrintOperator`, altimetry from a set-up).
- GRACE's coefficients read from the components equal those through the
  field space, and the Stokes conversion is unchanged.
- The Helmert design equals T + s x + R x x computed in three dimensions; a
  uniform translation equals the degree-1 field with V = U sampled at the
  stations, and a scale the degree-0 vertical field; Z^T A = 0 and
  Z^T Z = I, and dependent designs are refused.
- GNSS in each frame passes the adjoint check; with `frame="invariant"`,
  adding any Helmert motion to the data leaves Z^T d unchanged and the noise
  is Z^T C Z; the nuisance frame's posterior mean tends to the invariant
  frame's as one over the variance of the frame prior.
- A viscoelastic GNSS observable, the displacement rate between two dates of
  `IceHistoryOperator`, gives a model in the nuisance frame that passes the
  adjoint check.

## Work packages

Packages 1 to 6 are built: the solver's coefficient engine with V
(`SeaLevelEquation._solve_generalised`, and `solve_response` returning a
`GRDResponse`), `pyslfp.core.response`, the fingerprint operator onto a
`GRDSpace` with the horizontal part on request, its consumers (sea surface
height, GRACE, tide gauges, altimetry), which reach the fields by name through
`GRDSpace.field_operator`, the viscoelastic side, and the observation models
with GNSS, all as set out above.

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
   memory field and the adjoints; the Sobolev kernels. Built, as set out
   above.
6. **GNSS observation models**, with the frame of the data: frame (Helmert)
   parameters as nuisances, or frame-invariant data. The observation models
   are reshaped together here, tide gauges, altimetry and GRACE with GNSS:
   each takes its observable as an operator from the model space, a set-up's
   or a viscoelastic operator's, and adds the sampling, the stations' data and
   the noise, never physics.

WP3 of the models submodule follows packages 1 to 4, so that the new scripts
are built on the named response.
