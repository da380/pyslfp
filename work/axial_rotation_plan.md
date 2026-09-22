# The axial component of the rotational feedback

Written 22 September 2026. Theory from Al-Attar, Syvret, Crawford,
Mitrovica and Lloyd (2024), GJI 236, 362-378, `docs/ggad434.pdf`,
section 3; equation numbers below are theirs. This supersedes the
paragraph "The axial rotation component" in `love_numbers_plan.md`,
which was wrong about what degree 0 contributes (see section 3).

## 1. Summary

The solver keeps the two components of the rotation perturbation
orthogonal to the reference axis, through the degree-2, order-1
coefficients, and drops the axial component. Adding the axial component
is a small, exact extension at degree 2, order 0: two new constants, a
three-vector fixed point in `_rotation`, and the angular velocity
returned as a length-3 vector. The adjoint keeps its present form.

The one genuine question is the degree-0 part of the axial centrifugal
potential. It is proportional to r^2, so it is not a gauge; it forces a
uniform radial deformation, and the trace of the inertia perturbation
that feeds back on omega_3 depends on an interior moment of the
degree-0 displacement that no surface Love number supplies. The
recommendation is to project the centrifugal potential onto its
harmonic, degree-2 part throughout: this is the standard treatment in
the literature, it is exactly self-adjoint, it needs nothing from the
table beyond what is there, and what it neglects is a correction of
order 0.1 per cent to omega_3, which itself changes sea level by about
one part in a thousand of the polar-wander term. The exact treatment is
described at the end as a later extension of the Love number solver.

## 2. Theory

### 2.1 Rotational dynamics (paper, section 3.1)

Reference rotation Omega = Omega e_3, inertia tensor C with principal
moments C_1 = C_2 = A (equatorial) and C_3 = C (polar). A load and the
deformation it causes perturb the inertia tensor by c, and the rotation
vector by omega. The linearised steady Euler equation and conservation
of angular momentum give (eqs 38, 40, 41, 42)

    C~ . omega = j,      j = c . Omega,      C~ = diag(C - A, C - A, -C),

so that

    omega_1 = Omega c_13 / (C - A),
    omega_2 = Omega c_23 / (C - A),
    omega_3 = -Omega c_33 / C.

The generalised loading problem (eq. 55) adds a load k on omega, so
C~ . omega = j - k.

The centrifugal potential perturbation is psi = -(Omega x x).(omega x x)
(eq. 44), and eq. (49) defines j as a functional of the displacement u
and the total surface load sigma:

    omega' . j = -int rho u . grad psi' dV - int sigma psi' dS        (49)

for every constant omega', with psi' the centrifugal potential of omega'.
This is the identity that makes the weak form (50) symmetric, and hence
the reciprocity theorems hold; the numerics have to respect it.

### 2.2 The centrifugal potential of each component

With Omega along e_3, (Omega x x).(omega x x) = (Omega.omega)|x|^2 -
(Omega.x)(omega.x). For real orthonormal harmonics without the
Condon-Shortley phase (the model's conventions), at radius r:

    omega_1:  psi = Omega omega_1 z x = sqrt(4pi/15) Omega r^2 omega_1 Y_21c
    omega_2:  psi = Omega omega_2 z y = sqrt(4pi/15) Omega r^2 omega_2 Y_21s
    omega_3:  psi = -Omega omega_3 (x^2 + y^2) = -Omega omega_3 r^2 sin^2(theta)
            = -(2/3) sqrt(4pi) Omega r^2 omega_3 Y_00
              + (2/3) sqrt(4pi/5) Omega r^2 omega_3 Y_20

The first two are what the code has: at the surface r = a, psi_21 =
r_rot omega_perp with r_rot = `rotation_factor` = sqrt(4pi/15) Omega a^2.
The axial component has a degree-2 part, which is harmonic (r^2 Y_20),
and a degree-0 part, which is not (r^2 Y_00; its Laplacian is 6 x
constant). Define

    r_3 = (2/3) sqrt(4pi/5) Omega a^2 = sqrt(16pi/45) Omega a^2,   psi_20 = r_3 omega_3
    r_0 = -(2/3) sqrt(4pi) Omega a^2 = -sqrt(5) r_3,               psi_00 = r_0 omega_3

so r_3 / r_rot = 2/sqrt(3). Checked numerically against the parameters.

### 2.3 The inertia perturbation: what the surface potential gives

For a harmonic potential r^l Y_lm, the first-order change in the mass
integral int rho r^l Y_lm dV is int rho u . grad(r^l Y_lm) dV (mass is
transported with the displacement) plus the surface load term, and the
multipole expansion of the exterior gravitational potential gives, for
the total gravitational potential perturbation phi (excluding the
centrifugal potential) at the surface,

    int rho u . grad(r^l Y_lm) dV + int sigma a^l Y_lm dS
        = -(2l + 1) a^(l+1) phi_lm / (4 pi G).                       (*)

This is MacCullagh's formula in coefficient form. Applying (49) with
the degree-2 parts of psi':

    j_perp = sqrt(5/12pi) (Omega a^3 / G) phi_21    ->  omega_perp = i_rot phi_21,
    j_3^(2) = sqrt(5/9pi) (Omega a^3 / G) phi_20    ->  omega_3 = -i_3 phi_20,

with i_rot = `inertia_factor` = sqrt(5/12pi) Omega a^3 / (G (C - A)) as in
the code, and the new

    i_3 = sqrt(5/9pi) Omega a^3 / (G C).

Consistency checks, all verified numerically with the model parameters:
i_rot / r_rot = 5a / (4 pi G (C - A)) and i_3 / r_3 = 5a / (4 pi G C),
which is the statement that the map phi_2 -> omega is -(5a / 4 pi G)
C~^(-1) R^T with R the map omega -> psi_2; this transpose structure is
what the adjoint relies on. For a rigid earth, j_3^(2) equals Omega
times the load's own c_33 = int sigma a^2 sin^2(theta) dS exactly, and
with the elastic response omega_3 / Omega = -(1 + k_2') c_33 / C, the
textbook length-of-day formula; the prototype below reproduces it to
0.14 per cent, the difference being the tidal feedback of 2.4 below.

Sign check: mass moved from the poles to the equator has sigma_20 < 0,
hence phi_20 > 0 (k_2 < 0 in the attractive convention), hence
omega_3 < 0: the earth slows. Correct.

### 2.4 Where degree 0 enters

The degree-0 part of psi'_3 is -(2/3) Omega r^2. Applying (49) to it,

    j_3^(0) = (2/3) Omega [ 2 int rho u . x dV + a^2 int sigma dS ].

The surface term vanishes: the total load always conserves mass (the
ocean term is fixed by that condition, whatever the direct load). The
volume term is the change in the trace of the inertia tensor, tr(c)/4,
and only the degree-0 (uniform radial) part of u contributes to it:

    int rho u . x dV = sqrt(4pi) int_0^a rho(r) U_00(r) r^3 dr.

This is a radial moment of the degree-0 solution over the interior. The
degree-0 load numbers h_0 and k_0 do not give it: k_0 = -4 pi G a to
1e-8, the shell theorem, because a spherically symmetric
redistribution of mass leaves the exterior potential unchanged. Nor do
the degree-0 tidal numbers, which are zero in the table because they
are defined for the harmonic potential r^0 Y_00, a constant, not for
r^2 Y_00.

Three things could produce a degree-0 displacement: a degree-0 total
load (always zero, above); the degree-0 parts of the generalised loads
zeta^u and zeta^phi of adjoint problems, which are not zero in general
(a point measurement of vertical displacement has one); and the
degree-0 part of psi itself, which forces a uniform expansion or
contraction as the spin rate changes. The last needs a new degree-0
solve for the non-harmonic potential r^2 Y_00, whose forcing is a body
force -rho grad psi = -2 rho c r in the volume, not a boundary term.

### 2.5 The harmonic approximation (recommended)

Replace psi by its degree-2 projection Pi_2 psi everywhere: as the
forcing in the equations of motion and in the definition (49) of j. The
weak form (50) with these replacements is still symmetric, because the
two psi terms are int rho u' . grad psi_2(omega) dV and int rho u .
grad psi'_2(omega') dV with psi_2 linear in omega; so every reciprocity
theorem, and the adjoint of `FingerPrintOperator`, holds exactly for the
approximate dynamics, just as the paper notes for dropping omega_3
altogether. What is neglected is the trace of the inertia perturbation
due to deformation and the deformation due to the degree-0 centrifugal
potential. This is what the length-of-day literature does.

The sea level is unaffected by the choice: psi_00 Y_00 is a constant on
the surface and (53), Delta SL = -(u + phi + psi) / g + Phi_g / g, absorbs
it into Phi_g, which mass conservation fixes. The solver's step
`slc += mean_slc - ocean_average(slc)` does exactly that.

Sizes, for the parameters of the default model:

    i_rot r_rot k^t_2 = 0.314   ->  transverse denominator 1 - 0.314 = 0.686
    i_3 r_3 k^t_2     = 0.00137 ->  axial denominator 1 + 0.00137

The axial feedback is 300 times weaker because C replaces C - A. The
degree-0 feedback neglected here is of the same order as the axial
tidal feedback times a Love-number-sized factor, so of order 1e-4 to
1e-3 of omega_3. And omega_3 itself moves sea level at (2, 0) by about
1.2e-3 of what omega_perp moves it at (2, 1) (prototype, random test
load at degree 64: 7.7e-4 against 0.65, in units where the sea level
maximum is 4.6). The point of the extension is therefore omega_3 as an
output, the length-of-day change and its sensitivity kernel, rather
than its effect on sea level.

### 2.6 The fixed point in the code's notation

With phi^s the static part of the potential (load and generalised
loads), at degree 2:

    omega_perp = [ i_rot (phi^s_21 + k^t_2 r_rot omega_perp) - k_perp / (C - A) ]
    omega_3    = [ -i_3 (phi^s_20 + k^t_2 r_3 omega_3) + k_3 / C ]

solved in closed form,

    omega_perp = [ i_rot phi^s_21 - k_perp / (C - A) ] / (1 - i_rot r_rot k^t_2)
    omega_3    = [ -i_3 phi^s_20 + k_3 / C ]             / (1 + i_3 r_3 k^t_2)

then psi_21 = r_rot omega_perp, psi_20 = r_3 omega_3, and at both orders

    u_2m   = h_2 sigma_2m + (generalised) + h^t_2 psi_2m
    phi_2m = k_2 sigma_2m + (generalised) + k^t_2 psi_2m
    Delta SL_2m = -(u_2m + (phi_2m + psi_2m) / g).

The sign of the k_3 term follows from C~_33 = -C in C~ . omega = j - k.
The k load is the adjoint's `angular_momentum_change`, now a 3-vector.

### 2.7 The adjoint

Nothing changes in form. The L2 adjoint of zeta -> (Delta SL, u, phi,
omega) against weights (w_SL, w_u, w_phi, w_omega) is the generalised
solve with zeta^dag = w_SL, zeta^u = -w_u, zeta^phi = -g w_phi and

    k^dag = -g (w_omega - P^* w_phi),

where P is the map omega -> psi used in the forward problem, and P^*
must be its transpose. In the final form (section 7) P carries the
psi_00 row as well as the degree-2 rows: the term that pairs with it in
the forward problem is the inertia of a degree-0 potential load as a
uniform shell, which needs no Love numbers, so the pair goes in whether
or not the table has the axial numbers (the sea surface height operator
then subtracts the whole psi / g). Including one of the pair without
the other breaks the identity at 3e-3, which was checked.

The existing adjoint identity test of `FingerPrintOperator` (closing to
5e-13 with a computed table) is the check that all of this is right,
once the response space carries a 3-vector.

## 3. Correction to the earlier note

`love_numbers_plan.md` said the axial component would use h and k at
(0, 0) and that the degree-0 tidal numbers being zero settles the
degree-0 question. Neither holds: the degree-0 load numbers describe
the surface response to a surface load, which is zero here; the
degree-0 tidal numbers describe the response to a constant, not to
r^2 Y_00; and the quantity that does enter is an interior moment. What
was right is that the table needs nothing new for the degree-2 part.

## 4. Code changes

Each package ends with the fast suite and ruff green; David commits.

**WP1: parameters.** `EarthModelParameters` gains
`axial_rotation_factor` (r_3) and `axial_inertia_factor` (i_3) beside
`rotation_factor` and `inertia_factor`, computed in `__post_init__` from
the non-dimensional rotation frequency, sea-floor radius, G and the
polar moment. Docstring lists them with the formulae.

**WP2: the solver.** `SeaLevelEquation._rotation` takes the degree-2
block `load_lm[:, 2, :2]` (cos and sin rows, orders 0 and 1) and the
matching static blocks, returns `omega` of length 3 and (2, 2) blocks
for displacement, potential and centrifugal potential; the sin row at
order 0 is zero throughout. Cache `k_t[2]`, `h_t[2]`, the two
denominators, `1 / (C - A)` and `1 / C`. The three call sites
(`sea_level_from_load`, the final synthesis in
`solve_generalised_equation`, and the loop and final synthesis in
`solve_nonlinear_equation`) write the block `[:, 2, :2]` instead of
`[:, 2, 1]`. `np.zeros(2)` becomes `np.zeros(3)` for the zero-load
return and the non-linear solver's initial value. The
`angular_momentum_change` argument is a length-3 array. Module and
method docstrings: "degree-2, orders 0 and 1", omega as
[omega_x, omega_y, omega_z].

No new flag: with `rotational_feedbacks=True` all three components are
computed. (A `rotational_feedbacks="transverse"` mode for comparison
with codes that drop omega_3 would be cheap to add later if wanted.)

**WP3: operators.** In `linear_operators/physics.py`:
`EuclideanSpace(3)` in `lebesgue_response_space` and
`sobolev_response_space`; `centrifugal_potential_operator` maps a
3-vector to `coeffs[:, 2, 1] = r_rot w[:2]` and
`coeffs[0, 2, 0] = r_3 w[2]`, with the adjoint reading both back
(`r b^2` and `r_3 b^2` factors, as now); `response_measure_for_testing`
unchanged in form. `utils.check_response_space` requires dimension 3
with the message updated. `altimetry.sea_surface_height_operator`
needs no change beyond what the operator gives it. `grace.py` does not
touch omega.

**WP4: tests.** `tests/test_physics.py`: extend the reference Picard
iteration to the 3-vector (lagged omega_3 through psi_20 in the same
way as omega_perp through psi_21), so the accelerated solver is checked
against it with rotation on; add a length-of-day check, omega_3 / Omega
against -(1 + k_2') c_33 / C with c_33 from the quadrature of the
converged load times a^2 sin^2(theta) and the factor
1 / (1 + i_3 r_3 k^t_2) applied, to 1e-8; the zero-load case returns
`zeros(3)`. `tests/linear_operators`: the adjoint identity of
`FingerPrintOperator` and of `centrifugal_potential_operator` with the
3-dimensional space (the existing tests, with dimensions updated), and
the `check_response_space` test for the dimension. A regression number
worth recording in this file: the change in sea level over the oceans
from switching the axial term on, expected about 1e-3 of the transverse
rotational term.

**WP5: docs and tutorials.** `tutorials/scripts/01`, `02`, `04`: the
unpacked `angular_velocity` is now three components; in 04 the pole
plot loops over x and y as now, and a third panel or a printed number
gives the length-of-day change, Delta LOD = -LOD omega_3 / Omega (in ms
per 1000 Gt), which is the natural thing to show for the new
component. README and the Sphinx index: one sentence where the
rotational feedback is described. Release note: the fourth output is a
3-vector and the response space's fourth subspace is EuclideanSpace(3),
which breaks code that indexes it or builds measures on it; the
`angular_momentum_change` argument likewise. A minor version.

**WP6: the exact degree-0 treatment.** Done, see section 7.

## 5. Status

WP1 to WP5 done 22 September 2026, uncommitted; David agreed the
harmonic treatment, the names and no transverse-only option, with WP6
to follow at the end. Fast suite 191 tests and ruff green; the adjoint
identity of `FingerPrintOperator` with rotation closes to 3e-10 at
degree 32, the same as before the change. Tutorial 04 now prints the
length-of-day change and plots its kernel beside the two pole-shift
kernels: 5.1 ms for a tenth of West Antarctica, against a hand estimate
of about 4 ms; the kernel is negative at both poles and positive at the
equator, as it should be.

Regression numbers, default model at degree 64, a tenth of West
Antarctica: switching the axial term on changes sea level over the
oceans by 1.2e-4 of its maximum and vertical displacement by 5.6e-5;
omega_z / Omega = -5.8e-8 against |omega_perp| / Omega = 9.6e-6, so the
axial component is 0.6 per cent of the transverse one for this load.
Recorded here rather than in a test.

Release note for the next minor version: the angular velocity returned
by both solvers is now a 3-vector [omega_x, omega_y, omega_z]; the
fourth subspace of the response space is EuclideanSpace(3); the
`angular_momentum_change` argument and `centrifugal_potential_operator`
take 3-vectors; `check_response_space` requires dimension 3. Code that
indexed the 2-vector, or built measures on the 2-dimensional space,
needs updating. `LoveNumbers` gains the five axial numbers and
`has_axial`; `FORCINGS` gains `"centrifugal"`; `DegreeSystem.tide`
takes `power`; the table file gains an `axial` line, old files still
read. The shipped table needs re-uploading for the default model to
use the exact axial treatment.

## 7. The exact degree-0 treatment (WP6), done 22 September 2026

David asked for it after all, at the end. What went in:

*Solver.* `DegreeSystem.tide(power=None)`: the same assembly for a
potential (r/a)^power, the power defaulting to l; at l = 0 with power 2
it is the body force of the spherical mean of the centrifugal potential
of a spin change, and at l = 0 the centre's missing U dof is masked.
`solve_degree(..., forcing="centrifugal")` and `inertia_moment(material,
U)` = sqrt(4pi) sum w jac rho U r^3. `love_numbers` solves degree 0 for
the extra column and records the five axial numbers on the table:
`h_c`, `k_c` (surface displacement and potential per unit (r/a)^2
potential at degree 0) and `m_u`, `m_phi`, `m_c` (the inertia moments
of the degree-0 responses to the two load channels and to that
potential). They are scalars on `LoveNumbers`, NaN by default,
`has_axial` says whether they are known, `converted` scales them
(h_c like h_t, m_u and m_phi as length^4, m_c as mass time^2), and the
file carries them on an `axial` header line. Symmetry of the bilinear
form gives m_u = sqrt(4pi) g a^4 h_c / 2 and m_phi = sqrt(4pi) a^4 k_c
/ 2; `axial_reciprocity_residual` checks them, 2e-15 on PREM. k_c and
m_phi vanish by the shell theorem (1e-13 and 1e-12 relative), so the
new information is two numbers, h_c and m_c. For PREM in SI: h_c =
-9.99e-3 s^2/m, m_u = -2.86e26 m^4, m_c = -6.32e28 kg s^2; the signs
say a spin-up expands the body and a downward traction compresses it.

*Sea level equation.* `EarthModelParameters.uniform_rotation_factor`
r_0 = -sqrt(16pi/9) Omega a^2 = -sqrt(5) r_3. In `_rotation`, with
`load_0`, `displacement_load_0` and `potential_load_0` the degree-0
forcing coefficients,

    j_3 = C i_3 phi_20
        + (4/3) Omega [ (m_u + m_phi) sigma_00 + m_u zeta^u_00
                        + m_phi zeta^phi_00 + m_c psi_00 ]
        + (2/3) Omega a^4 sqrt(4pi) (sigma_00 + zeta^phi_00),

so the axial denominator gains (4/3) Omega m_c r_0 / C = 5.3e-4 on
PREM, and the displacement and potential at degree 0 gain h_c psi_00
and k_c psi_00 (the sea level too, harmlessly). The shell term and the
psi_00 row of `centrifugal_potential_operator` are transposes of each
other and need no Love numbers, so they are always on; with a table
lacking the axial numbers the moment and h_c, k_c terms are zero, which
is again self-adjoint. Checked: with `from_model` PREM at degree 32
the adjoint identity closes to 1e-12; zeroing any one of h_c, m_u, the
shell term or the psi_00 row alone breaks it (2e-5, 2e-5, 3e-3, 2e-5),
so every term is exercised and matched.

*Table and data.* `~/.pyslfp_data/pyslfp_love_numbers/PREM_4096_axial.dat`
is the regenerated table, PREM without ocean to degree 4096 with the
axial line, 32 s; its twelve columns agree with the shipped table to
4e-16. To make the default model exact, David uploads it as
`PREM_4096.dat` in a new version of the Zenodo record and users refresh
with `LoveNumbers.default(refresh=True)`; until then the default runs
the harmonic treatment and a model from `from_planet_model` the exact
one. The cached copy was not overwritten.

*Tests.* `tests/love_numbers/test_solver.py::test_axial_numbers`
(reciprocity, shell theorem, signs, single-degree solves agree, the
degree-2 centrifugal forcing is the tide, a constant is a gauge, units
and truncation); `tests/love_numbers/test_earth_model.py::
test_axial_feedback_uses_the_tables_degree_zero_numbers` (degree-0
displacement responds to spin with the numbers and not without, the
adjoint closes both ways, the degree-0 feedback on omega_z is below
1e-3 and stabilising); the length-of-day test includes the m_c term of
the feedback. Fast suite 193, slow 25, ruff green.

*Effect.* The degree-0 feedback changes omega_z by 5e-4 of itself. For
the outputs the only new thing is the uniform radial displacement under
a spin change, h_c r_0 omega_z, which a GPS station sees in principle:
for a tenth of West Antarctica the Earth contracts by 0.08 mm, 1e-5 of
the largest displacement, and its sensitivity kernel is what the
adjoint terms carry.

## 6. Decisions for David (taken 22 September 2026)

1. The harmonic (degree-2) treatment first, then the exact degree-0
   treatment as WP6, done so that WP6 adds terms without reshaping
   anything: agreed, and done the same day.
2. Names `axial_rotation_factor` and `axial_inertia_factor`: agreed.
3. No `"transverse"` option for `rotational_feedbacks`: agreed.
4. The regression number lives in this file, not in a test.
