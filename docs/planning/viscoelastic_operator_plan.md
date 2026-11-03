# The viscoelastic evolution as an operator

## Decisions

1. **One general operator.** The basic object maps the ice history, the
   initial sea level and the viscosity to the responses at the observation
   dates. Every other case comes from it by holding arguments fixed. The
   initial sea level is a necessary parameter: an ice history that carries no
   sea level leaves it to be chosen.
2. **Function spaces, no built-in discretisation.** Each argument is an
   element of a Hilbert space of functions: the ice history of a
   `FieldHistorySpace`, the initial sea level of a sphere space, the
   viscosity of a space of radial functions. The run's dates and the radial
   mesh are how the operator evaluates them, not its parameters. Simpler
   parameterisations, such as one viscosity per layer, come by composition.
3. **Log viscosity, every branch.** The viscosity parameter is
   chi_k = log(eta_k / eta_0) for each relaxing branch k of the rheology, with
   eta_0 the unit system's viscosity scale (density times length squared over
   time), so that in the library's units chi_k = log eta_k. Each chi_k is
   piecewise continuous on the layered radial mesh, with jumps allowed at the
   material interfaces.
4. **Shorelines are an option.** Migrating shorelines are the general case.
   Frozen shorelines reduce its form: the map becomes linear in the ice
   history and independent of the initial sea level for observables of change,
   and remains nonlinear in the viscosity.
5. **Derivatives with respect to every argument, by either method.** The
   continuous tangent-linear problem is derived and implemented alongside the
   discrete one, as the adjoints are. The operator takes `method="discrete"`
   or `method="continuous"`, which selects the tangent-linear problem and the
   adjoint together.
6. **Second derivatives.** The operator supplies its second derivative,
   F''(x)[v, .], with the second-order tangent-linear problem forwards and the
   second-order adjoint problem (Yu et al. 2025) backwards, by either method.
   pygeoinf then builds exact Hessian-vector products for any objective on
   the operator, and its Laplace approximations and Newton methods use them.
7. **No lateral viscosity.** Kernels with respect to laterally varying
   viscosity need the generalised spherical harmonics of the strain tensor
   and belong to the three-dimensional codes; the library's viscosity is a
   function of radius.

## The map

Domain: DirectSum([X_I, X_S, X_chi]).

- X_I: a `FieldHistorySpace` of ice thickness over an interval holding the
  run's dates, in the model's non-dimensional time from the first date, as
  for `IceHistoryOperator`.
- X_S: a sphere space (Lebesgue or Sobolev) for the initial sea level S_0.
- X_chi: the log-viscosity space below, one summand per relaxing branch.

Codomain: a `DirectSum` of one `GRDSpace` per observation date, the dates on
the run's grid, as for `IceHistoryOperator`. The responses are changes from
the initial state; an absolute sea level, S_0 + xi(t), is the affine
composition with the S_0 summand.

Value at (I, S_0, chi):

- the rheology: the reference `Rheology`'s branches, with mu_k fixed and
  itau_k = mu_k / eta_k = mu_k exp(-chi_k) / eta_0 at the mesh's nodes, and
  a `ViscoelasticOperator` built from it;
- the initial state: the thickness I(t_0) and the sea level S_0, with the
  ocean function by flotation under migrating shorelines, or the prescribed
  C_0 under frozen ones;
- `run_ice_history` through an ice model that reads I at the run's dates
  (the `BaseIceModel` contract), recording what the derivative needs (below).

Under frozen shorelines C_0 is an input of the operator, not a function of the
arguments, so the shorelines cannot flip as S_0 changes. The value is then
linear in I, independent of S_0, and nonlinear in chi.

Under migrating shorelines the derivative holds the forward trajectory's
ocean functions C(t) fixed, which is the derivative the adjoint already
computes. In the continuum this is exact: the variation of the ocean function
is a line distribution on the shoreline, and it multiplies rho_w S - rho_i I,
which vanishes there (Yu et al. 2025, eqs 51-53 and 62-63), so the singular
terms cancel. On the grid the map is piecewise affine in I and S_0, the
pieces separated by cells flipping between land and ocean, and the C-held
derivative is its derivative on each piece.

## The log-viscosity space

For branch k, the layers of the mesh in which the branch relaxes (mu_k > 0
with a finite viscosity in the reference rheology) each carry an interval
space on their radial range, `pygeoinf.interval` Sobolev or Lebesgue, padded
at both ends like the time factor of `FieldHistorySpace`, so that no end
condition marks the function inside its layer. A branch's space is the direct
sum over its layers, which makes chi_k continuous within a layer, for a
Sobolev order above one half, and free to jump at an interface. X_chi is the
direct sum over branches. Elastic layers (infinite viscosity, the lithosphere)
and fluid ones are outside it.

The space needs two maps to the mesh:

- evaluation at the layer's GLL radii, through the interval space's
  `basis_matrix`, giving chi_k as a nodal (nspec, ngll) array. A node shared
  by two elements of one layer takes one value, and one at an interface takes
  each layer's own;
- its transpose, taking a nodal derivative (dJ / d chi_k at each node, the
  shared nodes' copies summed) to the components of the derivative, which
  the metric then turns into the Riesz representer, as for the ice history.

The kernels already computed are dJ / d itau_k at the nodes, and
d itau_k = -itau_k d chi_k, so the nodal derivative is -itau_k dJ / d itau_k,
which is eta_k dJ / d eta_k.

Layer constants, a reference profile plus a perturbation, or one profile
shared by all branches come by composition with an inclusion or an affine map.

## The tangent-linear problem

### Continuous

Per degree, after the radial discretisation and continuous in time, the
forward problem is

    A_U(w(t), w^) = F(w^; zeta(t)) + b_mem(m(t))(w^),
    dm_k/dt = itau_k (d(w) - m_k),        m_k(t_0) = 0,
    zeta(t) = rho_i (1 - C(t)) I(t) - rho_i (1 - C_0) I(t_0)
              + rho_w (C(t) S(t) - C_0 S_0),    S(t) = S_0 + xi(t),

closed at each time by the sea level equation and the rotational feedback.
The viscosity enters through itau_k alone: the memory force carries mu_k,
which is fixed. Perturbing (I, S_0, chi) about a trajectory, with
d itau_k = -itau_k d chi_k and C(t) held,

    A_U(dw, w^) = F(w^; d zeta) + b_mem(dm)(w^),
    d(dm_k)/dt = itau_k (d(dw) + q_k - dm_k),      dm_k(t_0) = 0,
    q_k = -(d - m_k) d chi_k,
    d zeta = rho_i (1 - C(t)) dI(t) - rho_i (1 - C_0) dI(t_0)
             + rho_w C(t) d xi + rho_w (C(t) - C_0) dS_0.

So the tangent-linear problem is the forward problem with three changes: the
forward trajectory's shorelines are prescribed, the direct load is the
perturbation's, and each branch's drive is shifted by q_k, a known function of
time from the forward trajectory. The forward machinery steps it, as it steps
the continuous adjoint, with q_k entering the exponential trapezoid like the
strain, linear in time over each step.

The derivation in the theory note also shows the duality: pairing the
tangent-linear solution with the measurement duals and integrating by parts in
time recovers the continuous adjoint's kernels for the ice history, the
initial sea level and itau (Yu et al. 2025, eqs 61, 65, 81). This confirms
that the two continuous problems are each other's adjoint before either is
discretised.

### Discrete

The discrete tangent-linear problem is the forward mode of the time-stepped
map, the counterpart of the discrete adjoint's reverse sweep. In the ice
history and the initial sea level it coincides with the continuous one, since
a step is linear in its load. In chi it differentiates the scheme weights e,
alpha and beta in h = dt itau, and the effective modulus through phi_1, with
`scheme_weight_derivatives` already giving the derivatives. The two methods
therefore differ only in the viscosity in the forward direction, and in every
argument in the adjoint.

### What a linearisation records

A linearisation at (I, S_0, chi) is one forward run that records the ocean
functions and the internal states, at every node or at a checkpoint stride.
The tangent-linear runs need d and m at every node only for the
relaxation-time drive, and run forwards, so they refill the gaps from the
checkpoints by forward steps, exactly, without the backward amplification
that the adjoints' reconstruction spends digits on. Both adjoints already
reconstruct between checkpoints.

## The operator

A pygeoinf `Operator` with `linearize`, so that one run yields the value and
the derivative together. The derivative is a `LinearOperator` on the domain:

- forward: one tangent-linear run, discrete or continuous;
- adjoint: one adjoint sweep, which returns all three kernels together, the
  ice kernel carried onto X_I as in `IceHistoryOperator`, the initial sea
  level kernel onto X_S, and the log-viscosity kernels onto X_chi through the
  transpose of nodal evaluation.

With either method the ice kernel's representer is built from the per-node
duals, which for the continuous adjoint are the density under the date
quadrature plus the lumps: integrating the density against the time basis
by the same quadrature is the same sum.

The discrete pair are exact transposes, and the derivative is exact for the
computed map. The continuous pair are each a discretisation of the continuum
problem, consistent with each other and with the derivative of the computed
map to the order of the time stepping, so `check_operator` holds to a
tolerance that falls as the steps are refined. Their disagreement is the
resolution diagnostic that the two adjoints already provide.

Fixing arguments: `operator.fixing(initial_sea_level=..., log_viscosity=...)`,
or any subset, returns the operator on the remaining arguments, built by
composition with the affine inclusion, so its derivatives follow by the chain
rule. Under frozen shorelines with S_0 and chi fixed and the discrete method
the result is a genuine `LinearOperator` on X_I, `IceHistoryOperator`, whose
adjoint is exact, so that pygeoinf's linear machinery applies; with the
continuous method, whose adjoint is a transpose only to the order of the
scheme, the general form is kept.

## Second derivatives

pygeoinf builds the exact Hessian of an objective from the forward operator's
`second_derivative(x, v)`, which is F''(x)[v, .] as a `LinearOperator`. Its
MAP objectives, Laplace approximations, `NewtonCG` and `TrustRegionNewton` use
it when the operator supplies it, and fall back to Gauss-Newton otherwise. For
J = Phi(F(x)),

    H v = F'* Phi'' F' v + (F''(x)[v, .])* Phi'(F(x)),

and the second term is what a second-order adjoint computes: given a
direction v and the dual g = Phi'(F(x)), the adjoint of u -> F''(x)[v, u]. The
first term needs only the first-order problems. So the operator's second
derivative has:

- forward: the second-order tangent-linear problem, the forward problem once
  more, with the trajectory's shorelines held, no direct load, and each
  branch's drive shifted by

      q2_k = -(d_u - m_u) dchi_v,k - (d_v - m_v) dchi_u,k
             + (d - m_k) dchi_u,k dchi_v,k,

  from d^2 itau = itau dchi dchi, with (d_u, m_u) and (d_v, m_v) the
  tangent-linear states of u and v;
- adjoint: the second-order adjoint problem, stepped backwards in the same
  sweep as the first-order adjoint of g, with sources from the tangent-linear
  state of v, which is reconstructed from checkpoints like the forward state.

What is built is the second derivative of the C-held map. With the
trajectory's shorelines held, the map is affine in the ice history and the
initial sea level, so F''(x)[v, u] vanishes unless v or u moves the
viscosity: the second-order terms are the viscosity block and its coupling
to the ice history and the initial sea level, the trade-off between ice and
viscosity that a joint inversion meets. Under frozen shorelines this is the
whole second derivative. Under migrating shorelines it is not: the
cancellation that makes the C-held first derivative exact fails at second
order, because the coefficient rho_w S - rho_i I of the ocean function's
variation is itself perturbed, and its perturbation does not vanish on the
shoreline. Yu et al.'s second-order adjoint load (eq. 112) and Hessian
kernels for I and S_0 (eqs 116-117) therefore carry line distributions on
the shorelines at every time, and one on the initial shoreline: a load
perturbation moves the shoreline, and the moved shoreline changes the ocean
load. These terms are omitted. Evaluating them needs the shoreline located on
the grid, with the normal derivatives of S and I and line integrals along it,
which Yu et al. leave to future work. The theory note derives the C-held
second-order problems from Yu et al.'s equations and records the omitted
terms; the docs state the omission beside the operator. An inversion for the
ice history and the initial sea level alone under frozen shorelines gains
nothing over Gauss-Newton, and the frozen operator with the viscosity fixed
is linear and has no second derivative.

Both methods apply here too: the discrete second-order adjoint is the reverse
mode of the discrete tangent-linear step, and needs the second derivatives of
the scheme weights in h; the continuous one discretises the second-order
adjoint equations with the shoreline terms dropped, so that, as at first
order, the two methods differ by the time discretisation alone. A
Hessian-vector product costs four runs in all (Yu et al. 2025, Section 4):
the forward run and the first-order adjoint, which the linearisation and the
gradient already hold, a tangent-linear run for v, and the second-order
adjoint sweep. pygeoinf's Hessian uses only the adjoint of F''(x)[v, .]; its
forward action, which costs two tangent-linear runs and a second-order one,
serves the checks. Yu et al.'s approximate kernels (eqs 118-120), which drop
the second-order variables, are the Gauss-Newton Hessian that pygeoinf builds
when no second derivative is supplied.

## Costs

- The value at a new chi rebuilds the `ViscoelasticOperator`: the radial
  assembly and a factorisation per degree for each step size. A derivative at
  a fixed point reuses the linearisation's factorisations.
- A tangent-linear run costs about a forward run of linear steps, one sea
  level solve per date with the recorded ocean function, plus the forward
  recomputation between checkpoints when chi varies.
- An adjoint sweep costs about a forward run, as now.
- A Hessian-vector product costs two runs beyond the linearisation and the
  gradient (above).
- Under frozen shorelines with chi fixed nothing needs recording, and the map
  is applied directly.

## Tests

- The Taylor test, ||F(x + e v) - F(x) - e F'(x) v|| = O(e^2), for each
  argument alone and together, under frozen shorelines and about
  migrating-shoreline trajectories with perturbations that move no shoreline,
  for both methods (the continuous one to the order of the stepping).
- `check_operator` on the derivative: to rounding for the discrete pair, and
  for the continuous pair to a tolerance that falls at the scheme's order
  under refinement.
- The two tangent-linear problems agree in the ice history and the initial sea
  level, and converge to each other in chi at the scheme's order.
- Under frozen shorelines the derivative in S_0 vanishes, and the fixed form
  is a `LinearOperator` equal to the present `IceHistoryOperator`.
- The log-viscosity space: evaluation continuous within a layer and jumping at
  interfaces; the nodal transpose exact; the viscosity derivative equal to
  -itau dJ / d itau from the existing kernels; the representer converging as
  radial modes are added, as the ice kernel does as time modes are.
- A rheology with two branches (a Burgers mantle), so that every branch is
  exercised from the start.
- Second derivatives: the Taylor test on the gradient of a misfit,
  ||grad J(x + e u) - grad J(x) - e H u|| = O(e^2), for directions in each
  block; the symmetry of the Hessian, <w, H u> = <H w, u>, to rounding for
  the discrete method and to the scheme's order for the continuous one;
  `check_operator` on F''(x)[v, .] and the symmetry F''(x)[v, u] =
  F''(x)[u, v]; and the vanishing of F'' in the block of the ice history and
  the initial sea level. The Taylor tests about a migrating-shoreline
  trajectory use perturbations that move no shoreline, since the omitted
  terms are exactly those of shoreline motion.

## Steps

1. **The theory.** The continuous tangent-linear problem, its
   discretisation, the discrete tangent-linear problem and the duality with
   the adjoints, in the theory note's section on sensitivity kernels, with
   the summary in theory.rst.
2. **The log-viscosity space**, every branch: the per-layer interval spaces,
   nodal evaluation and its transpose, and the rheology from chi.
3. **The tangent-linear runs**, discrete and continuous, in the computational
   layer beside the adjoints in `SeaLevelEvolution`
   (`tangent_linear_ice_history`).
4. **The operator**: value, `linearize`, `method`, `shorelines` and
   `fixing`; `IceHistoryOperator` as its frozen, fixed case.
5. **Second derivatives**: the second-order tangent-linear and adjoint
   problems, derived in the theory note against Yu et al. (2025) and then
   implemented by both methods, with the second derivatives of the scheme
   weights; `second_derivative` on the operator.
6. **The docs**: numerics.rst (the costs, and the agreement of the two
   methods under refinement), release_4.md and the roadmap.

Kernels with respect to laterally varying viscosity are out of scope: they
need the generalised spherical harmonics of the strain tensor, which belong
to the three-dimensional codes.

## Smoothed flotation: the complete Hessian under migrating shorelines

The second derivative above holds the trajectory's ocean functions, and
`tools/shoreline_hessian.py` shows what that omits: between flips of
flotation cells nothing, across flips the whole Hessian in the initial sea
level and the ice history, a quarter to a third of a misfit's Gauss-Newton
term for 100 m thickness changes at a ten per cent residual. On the grid the
omitted terms are the jumps of the gradient at each flip, and the continuum
line distributions are their limit; neither is a quantity a smooth Hessian
can represent. The grid's own remedy is to make the ocean function
differentiable: a step smoothed over a flotation margin of a chosen height.

### The rule

With S the sea level, I the ice thickness and the land mask as now, the
flotation argument is the height

    x = (S - (rho_i / rho_w) I) / w,

positive over the oceans, with w the smoothing height, and the ocean
function is C = sigma(x) off the land mask, where sigma is the C^2
smoothstep sigma(x) = 6 x^5 - 15 x^4 + 10 x^3 on [0, 1] shifted to [-1, 1],
zero below and one above: compact support, so that a cell more than w from
flotation is land or ocean exactly as now, and continuous second derivative,
so that the Hessian is continuous. w is a length, in the model's units, an
option of `EarthState` (`flotation_width`, None for the sharp rule, the
default), which the nonlinear solver's and the migrating run's states
inherit as they inherit the land mask. Frozen shorelines are untouched; a
migrating run with w > 0 differs from the sharp one by the cells within w of
flotation being partly ocean, and tends to it as w -> 0.

### What the derivatives gain

Every load term is of the form rho_w C S or rho_i (1 - C) I, so each
variation of C enters multiplied by the flotation margin
Phi = rho_w S - rho_i I = rho_w w x. With delta C = sigma'(x) delta x and
delta x = (delta xi + delta S_0 - (rho_i / rho_w) delta I) / w,

    Phi delta C = rho_w Lambda (delta xi + delta S_0 - (rho_i / rho_w) delta I),
    Lambda = x sigma'(x),

a dimensionless field supported on the strip, which vanishes identically for
the sharp rule: this is the cancellation Yu et al. use, now a bounded term
rather than a vanishing one. Collecting terms, the tangent-linear load is

    delta zeta = rho_w C^ delta xi
               + rho_i (1 - C^_n) delta I_n - rho_i (1 - C^_0) delta I_0
               + rho_w (C^_n - C^_0) delta S_0,          C^ = C + Lambda,

the present form with the ocean function replaced by the effective one
C^ = C + Lambda, strip included, in the feedback and in every grounding
factor. The generalised solver and its adjoint already take the ocean
function as a weight field, in the load, the area and the mass shift, so the
first-order tangent-linear run and both adjoints change only in which field
they are handed and in the grounding factors of the kernels, C^_n for C_n.
The gradient of `final_sea_level_misfit` follows.

The second derivative gains one new term, from the second variation of C
and the product of the margin's and the ocean function's first variations:

    delta_uv zeta  ∋  rho_w K(x) (delta_u y)(delta_v y),
    K = (2 sigma' + x sigma'') / w = d(C + Lambda) / dS,
    y = xi + S_0 - (rho_i / rho_w) I,

a direct load in the second-order tangent-linear problem at every node,
built from the two tangent-linear runs' sea levels and the directions, and
in the second-order adjoint sweep the corresponding sources: a sea level
dual rho_w K delta_v y lambda_n at every node, and the same field times
-rho_i and rho_w into the thickness and initial sea level blocks of the
result, beside the derivative of the grounding factors. K is supported on
the strip and continuous, since sigma is C^2; `EarthState.flotation_curvature`
holds it.

### Tests

- The sharp limit: a migrating run with w -> 0 tends to the sharp run, and
  with no cell within w of flotation at any date equals it exactly.
- Mass conservation of a run with fractional ocean cells.
- The Taylor test and the adjoint check along directions that cross
  flotation, which the C-held derivative fails and the smoothed one passes;
  central differences of the gradient across the strip against the Hessian,
  symmetric to rounding, with `tools/shoreline_hessian.py` rerun with w > 0
  showing the measured term recovered.
- The frozen case unchanged: every existing test passes as it stands.

### Steps

1. The rule in `EarthState`, the solver's and the run's states inheriting
   it, and the fields a state exposes, `effective_ocean_function` and
   `flotation_curvature`; snapshots record them.
2. C^ in the tangent-linear run, both first-order adjoints and the misfit
   gradient.
3. The K terms in the second-order runs and sweeps, both methods.
4. The docs: theory.rst, numerics.rst (the tool rerun), release_4.md.

## Open questions

Settled: the class is `SeaLevelEvolutionOperator`, after the driver it
wraps, as `FingerPrintOperator` wraps the fingerprint; the default method is
`"discrete"`, since pygeoinf's factorisations of an assembled normal operator
assume an exact adjoint, which only the discrete pair gives, with
`"continuous"` the choice for a resolved run; and the frozen ocean function
is the evolution's background state's, the state also supplying the `land`
mask, with the initial sea level and the thickness at the first date taken
from the arguments.
