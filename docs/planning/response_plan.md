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
   memory field and the adjoints.
6. **GNSS observation models**, with the frame of the data: frame (Helmert)
   parameters as nuisances, or frame-invariant data.

WP3 of the models submodule follows packages 1 to 4, so that the new scripts
are built on the named response.
