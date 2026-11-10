Theory and methods
==================

This page summarises the physics ``pyslfp`` solves and the methods it uses,
at the level needed to use the library with confidence and to read its
source. The full derivations, with every sign and factor as the code uses
it, are in the `theory note
<https://github.com/da380/pyslfp/blob/main/docs/theory/pyslfp_theory.pdf>`_
(``docs/theory`` in the repository). Discretisation, convergence and cost are
on the :doc:`numerics` page, and the non-dimensionalisation on :doc:`units`.


Overview
--------

A change in the distribution of ice, water or sediment on the Earth's surface
deforms the solid Earth, perturbs its gravitational field and alters its
rotation. Each of these changes the height of the sea surface relative to the
solid surface, and the water so redistributed is itself a load. The resulting
pattern of relative sea level change is the solution of the sea level
equation. ``pyslfp`` solves it on a spherically symmetric, self-gravitating,
rotating Earth whose response is characterised, degree by degree, by a table
of Love numbers, and it does so for two physical approximations of the solid
Earth: an elastic one, for present-day and near-future change, and a
viscoelastic one, for glacial isostatic adjustment (GIA) through an ice
history.

The library is one sea level engine with two drivers.

* The radial Earth model is held by the separate package ``planetmodel``:
  layered models, their gravity, unit systems and radial meshes.
* ``pyslfp.love_numbers`` solves the linearised, quasi-static,
  self-gravitating loading problem of such a model at each spherical harmonic
  degree and returns a table of generalised Love numbers.
* ``pyslfp.core`` holds the engine, ``SeaLevelEquation``, which solves the
  linear, generalised and nonlinear forms of the sea level equation in the
  spherical harmonic domain for whatever Love numbers it is given. Nothing in
  it is elastic but the table it is handed.
* ``pyslfp.elastic`` is the fingerprint driver: a fingerprint is a single
  equilibrium solve with the elastic table.
* ``pyslfp.viscoelastic`` is the time-stepping driver: each step of a
  viscoelastic evolution is an equilibrium solve of the same engine, with
  per-step effective Love numbers and a memory field from the internal
  variables. It also computes sensitivity kernels by adjoint methods.
* Each driver exposes its physics as linear operators between Hilbert spaces,
  with their adjoints, for inverse problems built on ``pygeoinf``:
  ``FingerPrintOperator`` in ``pyslfp.elastic`` and ``IceHistoryOperator`` in
  ``pyslfp.viscoelastic``. ``pyslfp.observations`` samples the response as
  tide gauges, altimetry, GRACE and GNSS do.

The two drivers are named for the physical approximation they make, not for an
application: GIA is a use of the viscoelastic driver. They are not symmetric,
one being a map from a load to fields and the other a time integrator, and no
class does both.


Conventions
-----------

Fields on the surface are expanded in real spherical harmonics :math:`Y_{\ell
m}` that are orthonormal on the unit sphere, without the Condon-Shortley phase
(``pyshtools`` normalisation ``'ortho'``, ``csphase=1``). The surface is the
sphere of radius :math:`b`, which is the radius :math:`a` of the body for which
the Love numbers were computed, and surface integrals are evaluated with the
exact quadrature weights of the grid in use.

The gravitational potential is the physical one,
:math:`\Phi = -G\int\rho/|\mathbf{x}-\mathbf{x}'|\,dV'`, negative near added
mass, and the centrifugal potential of rotation with angular velocity
:math:`\boldsymbol{\Omega}` is :math:`-\tfrac12|\boldsymbol{\Omega}\times\mathbf{x}|^2`
with the same sign. Radial displacement is positive outward. Throughout,
:math:`\xi` is the relative sea level change and :math:`\zeta` the surface
load, a mass per unit area. The background state is an ice thickness
:math:`I` and a sea level :math:`S`, the height of the sea surface above the
solid surface (the ocean depth where positive, minus the topography on land),
and the ocean function :math:`C` is one over the oceans and zero elsewhere.


The sea level equation
----------------------

The linear equation
^^^^^^^^^^^^^^^^^^^

With the shorelines held fixed, a direct load :math:`\zeta_d` (for example
:math:`\rho_i \Delta I` for grounded ice) draws an ocean load
:math:`\rho_w C \xi`, so the total load is :math:`\zeta = \zeta_d + \rho_w C\xi`.
The sea surface stays an equipotential of the gravitational and centrifugal
potentials, so

.. math::

   \xi = -u - \frac{\phi + \psi}{g} + c,

where :math:`u` and :math:`\phi` are the displacement and potential
perturbation of the solid surface, :math:`\psi` the centrifugal potential
perturbation, and the constant :math:`c` is fixed by conservation of mass,

.. math::

   \rho_w \int C\xi\,dS = -\int \zeta_d\,dS .

On a spherically symmetric Earth, :math:`u_{\ell m} = h_\ell\zeta_{\ell m}` and
:math:`\phi_{\ell m} = k_\ell\zeta_{\ell m}` apart from the rotational terms
below, so the equation is a fixed point for the load,

.. math::

   \zeta = \zeta_d + \rho_w C\,\mathcal{P}\big[\mathcal{S}\zeta\big], \qquad
   (\mathcal{S}\zeta)_{\ell m} = -\Big(h_\ell + \frac{k_\ell}{g}\Big)\zeta_{\ell m}
   + \text{(rotation)},

with :math:`\mathcal{P}` adding the mass-conserving constant. Multiplication by
:math:`C` has norm one, and the spherical harmonic multiplier is of size
:math:`3\rho_w(1 + k'_\ell - h'_\ell)/[(2\ell+1)\bar\rho]` with
:math:`\rho_w/\bar\rho \approx 0.18`, so the plain iteration contracts. The
library accelerates it with Anderson mixing; see :doc:`numerics`.

The generalised equation and the adjoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The generalised equation adds three forcings to the direct load: a
displacement load :math:`\zeta^u` that acts as a surface traction alone, a
potential load :math:`\zeta^\phi` that acts as a gravitating mass alone, and
an angular momentum forcing :math:`\mathbf{L}` in the rotational balance. They
exist for the adjoint. The fingerprint operator, mapping a direct load to
:math:`(\xi, u, \phi, \boldsymbol{\omega})`, has an adjoint that is itself a
generalised sea level equation (Al-Attar et al. 2024): applied to
:math:`(\xi^\dagger, u^\dagger, \phi^\dagger, \boldsymbol{\omega}^\dagger)` it is
the relative sea level of the generalised equation with

.. math::

   \zeta_d = \xi^\dagger, \qquad \zeta^u = -u^\dagger, \qquad
   \zeta^\phi = -g\,\phi^\dagger, \qquad
   \mathbf{L} = -g\big(\boldsymbol{\omega}^\dagger - \Psi^*\phi^\dagger\big),

where :math:`\Psi` maps an angular velocity to its centrifugal potential. The
identity is exact only when the solver uses the reciprocal numbers of a single
table and the radius, surface gravity and gravitational constant of the body
for which that table was computed, which is why ``EarthModel`` adopts the body
from its table. ``SeaLevelEquation.adjoint_generalised_solve`` is, separately,
the exact transpose of the generalised solve, returning the duals of all of
its inputs, which the viscoelastic adjoints need.

The nonlinear equation with migrating shorelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large changes the ocean function itself changes, set by flotation,

.. math::

   C = H\big(\rho_w S - \rho_i I\big),

so that ice thinner than the water depth that would float it belongs to the
ocean and only grounded ice loads the Earth as ice; optionally the Caspian is
excluded. Given an initial state :math:`(I_0, S_0, C_0)` and an ice thickness
change, the solver iterates on :math:`(S, C)` with the load

.. math::

   \zeta = \rho_i\big[(1 - C)I - (1 - C_0)I_0\big] + \rho_w\big(CS - C_0S_0\big)
   + \rho_s\,\Delta H_s,

the last term being an optional sediment load, and a eustatic shift that
conserves the total of water, grounded ice and sediment over the current ocean
(Kendall et al. 2005). The result is a new state together with :math:`\xi`,
:math:`u`, :math:`\phi` and :math:`\boldsymbol{\omega}`.


Love numbers
------------

Definitions
^^^^^^^^^^^

The numbers the library holds are not the usual dimensionless ones. The solver
works with the physical gravitational potential and with dimensional numbers:
a displacement or a potential per unit of the forcing that produced it. A
surface load :math:`\zeta_{\ell m}Y_{\ell m}` acts in two ways: it presses on the
surface with the traction :math:`-g\zeta_{\ell m}Y_{\ell m}`, and it attracts
the body as a surface mass in Poisson's equation. The generalised Love numbers
of Al-Attar et al. (2024) are the surface response to each acting alone. With
the displacement written as
:math:`\mathbf{u} = U Y_{\ell m}\hat{\mathbf{r}} + V\nabla_1 Y_{\ell m}` and the
potential perturbation as :math:`\phi Y_{\ell m}`, both at :math:`r = a`,

.. math::

   U = h_\ell^u\zeta^u_{\ell m} + h_\ell^\phi\zeta^\phi_{\ell m}, \qquad
   V = l_\ell^u\zeta^u_{\ell m} + l_\ell^\phi\zeta^\phi_{\ell m}, \qquad
   \phi = k_\ell^u\zeta^u_{\ell m} + k_\ell^\phi\zeta^\phi_{\ell m},

where :math:`\zeta^u` presses but does not attract and :math:`\zeta^\phi`
attracts but does not press. A true load does both, so its numbers are the sums
:math:`h_\ell = h_\ell^u + h_\ell^\phi`, :math:`l_\ell = l_\ell^u + l_\ell^\phi`
and :math:`k_\ell = k_\ell^u + k_\ell^\phi`, the properties ``h``, ``l`` and
``k``. In SI, :math:`h` and :math:`l` are in m³ kg⁻¹ and :math:`k` in
m⁴ kg⁻¹ s⁻². A third channel, the tangential traction
:math:`-g\zeta^v_{\ell m}\nabla_1 Y_{\ell m}`, has the numbers :math:`h^v_\ell`,
:math:`l^v_\ell` and :math:`k^v_\ell`, which the adjoint problem for a
functional of horizontal displacement needs. The tidal numbers
:math:`h^t_\ell`, :math:`l^t_\ell` and :math:`k^t_\ell` are the response to the
unit external potential :math:`\psi = (r/a)^\ell Y_{\ell m}`, so :math:`h^t`
and :math:`l^t` are in s² m⁻¹ and :math:`k^t` is dimensionless.

The split into channels is not needed to solve the sea level equation forward,
for which :math:`h_\ell` and :math:`k_\ell` suffice. It is needed because the
adjoint is itself a sea level equation in which an observation of displacement
acts through the traction channel alone, an observation of potential through
the source channel alone, and an observation of horizontal displacement
through the tangential traction.

Conventional numbers and reciprocity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conventional dimensionless load numbers :math:`h'_\ell`, :math:`l'_\ell`
and :math:`k'_\ell` of Farrell (1972) refer the response to the direct
potential of the load, :math:`4\pi G a\zeta_{\ell m}/(2\ell+1)` in the geodetic
sign convention, where the potential is positive near mass:

.. math::

   h'_\ell = \frac{(2\ell+1)\,g}{4\pi G a}\,h_\ell, \qquad
   l'_\ell = \frac{(2\ell+1)\,g}{4\pi G a}\,l_\ell, \qquad
   k'_\ell = -\frac{2\ell+1}{4\pi G a}\,k_\ell - 1,

and the geodetic tidal numbers differ from the library's only by the sign of
the potential and a factor of gravity,
:math:`k^T_\ell = k^t_\ell`, :math:`h^T_\ell = -g\,h^t_\ell` and
:math:`l^T_\ell = -g\,l^t_\ell`. ``LoveNumbers.conventional()`` and
``LoveNumbers.tidal()`` return these. The loading problem is self-adjoint,
which gives the reciprocity relations

.. math::

   g\,h^\phi_\ell = k^u_\ell, \qquad h^v_\ell = \ell(\ell+1)\,l^u_\ell, \qquad
   k^v_\ell = g\,\ell(\ell+1)\,l^\phi_\ell,

the first being eq. (64) of Al-Attar et al. (2024);
``LoveNumbers.reciprocity_residual()`` checks all three.

Degree 1 is posed in the frame of the centre of mass, where the surface
potential perturbation vanishes and :math:`k'_1 = -1`. At degree 0 the tidal
numbers are zero, a uniform external potential being a gauge, while the load
numbers are not: mass conservation fixes :math:`k_0 = -4\pi G a`. Degree 0 also
carries five axial numbers, ``h_c``, ``k_c``, ``m_u``, ``m_phi`` and ``m_c``:
the surface response to the spherical mean of the centrifugal potential of a
change in spin rate, which goes as :math:`r^2` rather than being a constant,
and the inertia moments :math:`\sqrt{4\pi}\int\rho\,U\,r^3\,dr` of the degree-0
responses to the two load channels and to that potential. The axial component
of the rotational feedback needs them, because the trace of the inertia
perturbation is not seen by the degree-2 potential. Symmetry gives
:math:`m_u = \sqrt{4\pi}\,g a^4 h_c/2`, which
``LoveNumbers.axial_reciprocity_residual()`` checks, and :math:`k_c` and
:math:`m_\phi` vanish by the shell theorem.

The radial problem
^^^^^^^^^^^^^^^^^^

The numbers are surface values of the solution of the quasi-static,
self-gravitating loading problem of a spherically symmetric, transversely
isotropic, hydrostatically pre-stressed body, reduced degree by degree. The
weak form is eq. (D76) of Al-Attar & Tromp (2014): solid regions carry the
radial scalars :math:`(U, V, \phi)`, fluid regions above degree zero the
potential alone, and the gravitational field outside the domain is eliminated
by Dirichlet-to-Neumann terms. It is discretised by Gauss-Lobatto-Legendre
spectral elements on a ``planetmodel`` radial mesh, and each degree is solved
on the part of the mesh its solution reaches; see :doc:`numerics`. A model
frozen at a frequency with ``planetmodel.frozen`` has complex moduli and gives
complex Love numbers, which can be computed and plotted; the sea level solver
takes real tables only, and time-dependent viscoelasticity is handled by the
time-domain solver below.


Rotational feedback
-------------------

To first order the change :math:`\boldsymbol{\omega}` in angular velocity
produces a centrifugal potential whose coefficients on the surface are

.. math::

   \psi^c_{21} = R_\perp\,\omega_x, \qquad \psi^s_{21} = R_\perp\,\omega_y, \qquad
   \psi_{20} = R_z\,\omega_z, \qquad \psi_{00} = R_0\,\omega_z,

with :math:`R_\perp`, :math:`R_z` and :math:`R_0` multiples of
:math:`\Omega b^2` (the ``rotation_factor``, ``axial_rotation_factor`` and
``uniform_rotation_factor`` of ``EarthModelParameters``). The transverse
components (polar wander) couple to degree 2, order 1, through the tidal
numbers; the axial component (length of day) to degree 2, order 0, and,
through the axial numbers, to degree 0. On loading time scales the linearised
Liouville equation reduces to its quasi-static balance, and with the inertia
perturbation read from the degree-2 potential by MacCullagh's formula each
component of :math:`\boldsymbol{\omega}` is a scalar fixed point, solved in
closed form inside every iteration of the sea level equation. The returned
displacement, potential and angular velocity are therefore consistent with the
same load.

The equilibrium moments of inertia :math:`A` and :math:`C` in that balance are
specified quantities: they cannot be computed from the spherically symmetric
model in which the loading problem is solved, whose inertia tensor is
isotropic (Yu et al. 2025). The transverse fixed point has the denominator
:math:`1 - k^t_2/k_{\mathrm{sus}}`, with :math:`k_{\mathrm{sus}} = 3G(C-A)/(\Omega^2
b^5)` the oblateness the moments sustain, and it is stable only if this stays
positive for every tidal response the model can produce. The largest is the
hydrostatic fluid limit :math:`k_{\mathrm{hyd}}`, the degree-2 tidal number of
the model with its rigidity set to zero (``fluid_tidal_number``). For the
Earth the observed moments satisfy :math:`k_{\mathrm{hyd}} < k_{\mathrm{sus}}`,
the excess being the non-hydrostatic part of the figure, and they are the
default. For any other body the moments may be built from the model itself,
:math:`C - A = (1+\delta)\,k_{\mathrm{hyd}}\Omega^2 b^5/(3G)` with :math:`\delta`
a non-hydrostatic fraction (Mitrovica et al. 2005;
``EarthModelParameters.with_hydrostatic_moments``). A configuration on the
wrong side of this limit is refused by the time-domain solvers. The axial
feedback is unconditionally stable, so every solver takes
``rotational_feedbacks`` as ``True`` for the full feedback, ``"axial"`` for the
axial component alone, or ``False``.


Viscoelastic loading
--------------------

Rheology and internal variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The viscoelastic side treats a generalised Maxwell (Prony) body with branches
:math:`k` of shear modulus :math:`\mu_k` and relaxation time :math:`\tau_k`, and
a long-term shear modulus :math:`\mu_\infty`, in the internal-variable
formulation of Al-Attar & Tromp (2014):

.. math::

   \mathbf{T} = \kappa\,(\operatorname{tr}\boldsymbol{\varepsilon})\,\mathbf{1}
   + 2\mu_\infty\mathbf{d} + \sum_k 2\mu_k(\mathbf{d} - \mathbf{m}_k), \qquad
   \dot{\mathbf{m}}_k = \frac{\mathbf{d} - \mathbf{m}_k}{\tau_k},

with :math:`\mathbf{d}` the deviatoric strain. The material's shear modulus is
the unrelaxed one, :math:`\mu_U = \mu_\infty + \sum_k\mu_k`, and the classical
Maxwell body is one branch with :math:`\mu_\infty = 0` and
:math:`\tau = \eta/\mu`. ``Rheology`` takes the viscosity as a constant, a
function of radius or a value per layer, or the relaxation time
:math:`\tau(r)` directly, in which case the viscosity :math:`\eta = \mu\tau`
inherits its depth dependence from the shear modulus. Bulk relaxation is not
modelled, fluid regions carry no branches, and a relaxing branch requires an
isotropic material.

For spheroidal forcing the internal variable of each branch reduces, at each
:math:`(\ell, m)`, to three radial scalars (Al-Attar & Tromp 2014, eqs
D68-D73): :math:`M`, the tangential component, :math:`R`, the radial-radial
component, and :math:`S`, the radial-tangential component. They relax towards
the deviatoric strain scalars

.. math::

   e_M = V, \qquad e_R = r\partial_r U - U + \tfrac12\ell(\ell+1)V, \qquad
   e_S = r\partial_r V - V + U,

branch by branch, and enter the elastic problem as a memory force with the
weights :math:`\tfrac43`, :math:`\ell(\ell+1)` and
:math:`\ell(\ell+1)(\ell(\ell+1) - 2)` for :math:`R`, :math:`S` and :math:`M`.
The state of the system is the internal variables alone: the displacement,
potential and sea level are slaved to them through the degree solves.

Time stepping
^^^^^^^^^^^^^

With :math:`h_k = \Delta t/\tau_k`, each scheme eliminates the new internal
variable in favour of the new strain, which leaves an elastic solve with an
effective shear modulus and a memory force built from known quantities.

* Backward Euler, first order:
  :math:`\mathbf{m}^{n+1} = (\mathbf{m}^n + h\,\mathbf{d}^{n+1})/(1+h)`, so
  :math:`\mu_{\mathrm{eff}} = \mu_\infty + \sum_k\mu_k/(1+h_k)`.
* The exponential trapezoid, second order and the default: with
  :math:`e = \exp(-h)`, :math:`\varphi_1 = (1-e)/h`,
  :math:`\alpha = \varphi_1 - e` and :math:`\beta = 1 - \varphi_1`,
  :math:`\mathbf{m}^{n+1} = e\,\mathbf{m}^n + \alpha\,\mathbf{d}^n +
  \beta\,\mathbf{d}^{n+1}`, which is exact for a strain linear in time over the
  step, so :math:`\mu_{\mathrm{eff}} = \mu_\infty + \sum_k\varphi_{1,k}\mu_k`.
* Exponential Euler, explicit:
  :math:`\mathbf{m}^{n+1} = e\,\mathbf{m}^n + (1-e)\,\mathbf{d}^n`, exact for a
  strain constant over the step.

An elastic region is the limit :math:`\tau\to\infty`, stored as
:math:`\tau^{-1} = 0`. The two implicit schemes drive the coupled evolution.

Coupling to the sea level equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the effective moduli fixed, a step's instantaneous response to a surface
load is elastic, characterised by per-step effective Love numbers computed by
the radial solver on the effective material. The internal variables contribute
a memory field, the surface displacement and potential of the memory force
under zero load, together with its degree-0 inertia moment. The step's sea
level equation is then the generalised equation with the effective table and
the memory field as a ready response, closed by the same iteration for the
ocean and rotational feedbacks. The memory field's horizontal displacement
joins the step's horizontal displacement in the same way, but feeds nothing
back, since the sea level equation sees only the vertical displacement and the
potential. Each step's response is a ``GRDResponse`` of the run's response
space, the same object the elastic solvers return. The converged load and centrifugal potential
finish the step, updating the internal variables. A run starts from the
instantaneous elastic response to the load at its first time, the right limit,
so that a load switched on at the start is stepped at full order. In the limit
of infinite relaxation times everywhere the evolution reproduces the elastic
fingerprint.

``SeaLevelEvolution.run_ice_history`` drives the evolution with any ice model,
dates in kiloyears before present. Under frozen shorelines each step solves the
linear equation, the direct load being the grounded ice change over the
background state's ocean function. Under migrating shorelines each step solves
the nonlinear equation, with grounding and mass conservation re-evaluated as
the shorelines move, and every snapshot records its ocean function; the
background state must then be the state at the first date.


Sensitivity kernels
-------------------

Elastic kernels
^^^^^^^^^^^^^^^

The adjoint of the fingerprint operator is a generalised sea level equation,
as above, so a sensitivity kernel costs one solve.
``FingerPrintOperator`` wraps the forward and adjoint solves as a ``pygeoinf``
linear operator, where it composes with the observation models of
``pyslfp.observations`` (below).

The viscoelastic adjoint problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an evolution, an objective functional is a sum over observation dates
:math:`t_i` of measurements

.. math::

   J = \sum_i\Big[\int g^{(i)}_\xi\,\xi(t_i)\,dS + \int g^{(i)}_u\,U(t_i)\,dS
   + \int g^{(i)}_\phi\,\phi(t_i)\,dS + \mathbf{g}^{(i)}_\omega\cdot
   \boldsymbol{\omega}(t_i) + \sum_{\ell m} s^{(i)}_{\ell m}V_{\ell m}(t_i)\Big],

with :math:`U` and :math:`V` the vertical displacement and the potential of the
horizontal displacement :math:`\nabla_1 V`. A ``Measurement`` carries the
duals of the scalar fields and of the angular velocity change. A
``GRDResponse`` given as a measurement stands for its inner product with the
response in its own space, which may be a Sobolev space, and carries all the
duals, the horizontal one included: with :math:`\lambda` the space's metric
and :math:`c` the components, the duals are the metric-weighted components
:math:`\lambda g`. The adjoint theory of glacial isostatic
adjustment was set out by Crawford et al. (2018), and extended to rotational
feedbacks by Yu et al. (2025), whose formulation the library follows. The
adjoint problem has the same form as the forward problem with more general
forces: a sea level dual forces like an ice load, displacement and
potential duals enter as the generalised channels, a horizontal dual as a
tangential traction :math:`\zeta^v_{\ell m} = -s_{\ell m}/(b^2\ell(\ell+1))`,
and an angular velocity dual as an angular momentum forcing. The adjoint ocean function is not determined
dynamically but is the time-reversed ocean function of the forward run, so the
adjoint problem is linear whatever the shorelines did, and under migrating
shorelines only the history :math:`C(t)` of the forward run is needed.

The library computes the adjoint in two ways, deliberately.

* ``SeaLevelEvolution.adjoint_ice_history`` is the exact transpose of the
  time-stepped forward map: one reversed sweep, each step a symmetric banded
  solve per degree chained with the engine's ``adjoint_generalised_solve``.
  Its gradients are exact for the discrete model, to the solvers' tolerance.
  Under frozen shorelines the ice history enters linearly, so no forward run is
  needed.
* ``SeaLevelEvolution.continuous_adjoint_ice_history`` discretises the
  continuous adjoint equations instead. Between observations the adjoint state
  is stepped by the same machinery as a forward run without load, in reversed
  time; at each observation time an instantaneous (unrelaxed) fingerprint
  problem supplies the singular part of the kernel and the adjoint internal
  variables jump by :math:`[\mathbf{m}^\dagger] = \tilde{\mathbf{d}}^\dagger/\tau`
  (Yu et al. 2025, eq. 80). It never sees the stepper's internals.

A discrete adjoint is exact for the discretised functional, which in an
under-resolved run is a poor stand-in for the continuum one. The two methods
differ by the commute error of discretisation and transposition, which
vanishes as the run resolves its rheology, so their disagreement is a
resolution diagnostic for the forward calculation itself: when it is large the
time step should be refined, rather than either gradient preferred.

The views of an ice kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^

The kernel with respect to the ice thickness is singular in time (Yu et al.
2025, eq. 81): a load applied at an observation time produces an instantaneous
elastic response, and every load depends on the initial thickness. It is

.. math::

   K_I(\mathbf{x}, t) = \kappa(\mathbf{x}, t)
   + \sum_i \kappa_i(\mathbf{x})\,\delta(t - t_i)
   + \kappa_0(\mathbf{x})\,\delta(t - t_0),

with a delta at each observation time, whose amplitude is the instantaneous
fingerprint adjoint of that measurement, a delta at the initial time from the
thickness baseline, and a continuous density. ``IceKernels`` carries three
views of it: the discrete per-node duals, which satisfy the gradient identity
exactly for perturbations on the date grid but do not converge pointwise under
refinement; this distributional decomposition, whose parts do converge; and
the quelled view of Backus (1970), the kernel with respect to the rate of
thickness change, which is the cumulative time integral of :math:`K_I`, bounded
with finite jumps at the observation times only, and pairs with thickness
histories continuous in time. ``IceKernels.at_dates`` maps the kernels onto an
ice model's own dates by the transpose of its linear interpolation.

The Sobolev kernel is the derivative's Riesz representer in a space of ice
histories, ``FieldHistorySpace``: the product of a Sobolev space
:math:`H^s(T)` on the run's interval of time, in the model's non-dimensional
time from the run's first date, and one :math:`H^r(S^2)` on the sphere, with the inner product :math:`(A_t^s\otimes A_x^r\,u, v)`, where
:math:`A_t = 1 - L_t^2\,\partial_t^2`. Both factors are diagonal in their
eigenbases, so the history's components :math:`c_{kj}` lie on the products
:math:`\phi_k(t)Y_j(\mathbf{x})` with the metric :math:`\mu_k\nu_j`. The forward
map reads a history at the date nodes, so its derivative has the components
:math:`\sum_n \phi_k(t_n)\,d_{n,j}`, with :math:`d_n` the discrete per-node
duals, and the representer divides them by the metric. For :math:`s > 1/2`
evaluation at a time is bounded on :math:`H^s(T)`, so each delta of
:math:`K_I` becomes the Green's function of :math:`A_t^s`, continuous in time:
the representer converges as time modes are added, where an L2 representer
grows without bound, and as the date grid is refined. The quelled view is a
one-sided form of the same idea, the Green's function of :math:`\partial_t`
with a terminal condition. The time space's mesh is padded at both ends, so
that no end condition distorts the kernel near the present.
``IceHistoryOperator`` is the frozen-shoreline map from a history to the
responses at the observation dates as a ``pygeoinf`` operator, and its adjoint
applied to the duals of a measurement is this kernel.

Initial sea level and viscosity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The initial sea level enters each step's load as
:math:`\rho_w(C_n - C_0)\,\delta S_0`, so its kernel assembles from the per-node
load duals at no extra cost: :math:`\rho_w\sum_n (C_n - C_0)\,\lambda_n`, the
discrete analogue of eq. (65) of Yu et al. (2025), which vanishes identically
under frozen shorelines. This is the kernel for objectives on the sea level
change; one on the absolute sea level adds the sum of its own sea level duals.

Given the recorded forward trajectory, the adjoints also return the
sensitivity to the inverse relaxation time :math:`\tau_k^{-1}(r)` of each branch,
the one-dimensional reduction of Yu et al. (2025, eq. 61): the discrete variant
differentiates the scheme weights in :math:`h = \Delta t\,\tau^{-1}`, the
continuous one integrates the adjoint internal variables against the forward
:math:`\mathbf{d} - \mathbf{m}`, contracted with the weights of the degree-basis
scalars. ``IceKernels.viscosity`` maps them to viscosity through
:math:`\partial\tau^{-1}/\partial\eta = -\tau^{-2}/\mu`.

The kernels' convention
^^^^^^^^^^^^^^^^^^^^^^^

The kernels with respect to a parameter in the Earth, the inverse relaxation
times and the viscosity, and their derivatives from the second-order sweeps,
are densities per unit volume, held as their samples :math:`K(r)` at the nodes
of the material's radial mesh, one array of shape ``(nspec, ngll)`` per
branch. For a spherically symmetric perturbation :math:`\delta(r)` of the
parameter

.. math::

   \delta J = \int K\,\delta\,dV = 4\pi\int K(r)\,\delta(r)\,r^2\,dr
   \approx \sum_p W_p\,K(r_p)\,\delta(r_p),
   \qquad W_p = 4\pi r_p^2\,w_p,

with :math:`w_p` the radial quadrature weight of node :math:`p`, its
Gauss-Lobatto-Legendre weight times the element's Jacobian
(``Material.volume_weights``; the sum is ``Material.volume_pairing``). Neither
the :math:`r^2` nor the :math:`4\pi` is in the kernel: both belong to the
volume measure. The sweeps accumulate the gradient with respect to the nodal
values, :math:`W_p K(r_p)`, and divide the weights out on return
(``Material.kernel_samples``), so that for the discrete adjoint the sum is the
exact derivative of the discrete model under a perturbation given by its
nodal values. A node at the centre has no volume: its gradient entry is zero,
and its sample is the value there of the element's polynomial through its
other nodes. The kernel is the :math:`L^2` representative of the derivative
under the volume measure, the measure of the radial spaces of ``pygeoinf``
and so of ``LogViscositySpace``, which forms the derivative's components
with the same weights; and it converges as a function of radius under
refinement of the radial mesh, where the gradient entries, which carry each
mesh's quadrature weights, do not.

The values are in the material's units: the objective's unit per unit of the
parameter per unit of volume. Converted to SI through the material's
``scales`` they are the objective's unit per parameter unit per cubic metre,
for the inverse relaxation time the objective's unit per
:math:`\mathrm{s}^{-1}` per :math:`\mathrm{m}^3`. The kernels with respect to
the ice history and the initial sea level are fields on the surface grid and
pair by :math:`\int\cdot\,dS`.

The tangent-linear problem and second derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a given history of ocean functions :math:`C(t)` and given relaxation
times, the forward problem is one linear system over the run's interval,
:math:`\mathcal{M}(\Lambda)\,y = \mathcal{B}\,\zeta_d`, in which the inverse
relaxation times :math:`\Lambda` enter the internal-variable equations alone
and the ice history and the initial sea level enter the direct load alone,
:math:`\zeta_d = \rho_i(1-C)I - \rho_i(1-C_0)I_0 + \rho_w(C-C_0)S_0`. With the
viscosity parameterised as :math:`\chi_k = \log(\eta_k/\eta_0)` per branch, the
tangent-linear problem, the derivative's forward action, is the forward
problem once more: the shorelines prescribed from the trajectory, the direct
load replaced by the perturbation's, and the drive of each branch shifted by
:math:`\mathbf{q}_k = -(\mathbf{d}-\mathbf{m}_k)\,\delta\chi_k`, a known function
of time. Holding the ocean functions is exact: their variation is a line
distribution on the shoreline multiplying :math:`\rho_w S - \rho_i I`, which
vanishes there (Yu et al. 2025). The adjoint of the tangent-linear operator is
the continuous adjoint problem, so the two are each other's adjoint before
either is discretised. The discrete tangent-linear problem is the forward mode
of the time-stepped map; in the ice history and the initial sea level it
coincides with the continuous one, and in the viscosity the two differ by the
order of the scheme, as the adjoints do.

Differentiating once more gives the second derivative of the map with the
ocean functions held. It vanishes unless a direction perturbs the viscosity,
since the map is then affine in the ice history and the initial sea level,
and its adjoint is the second-order adjoint problem of Yu et al. (2025): the
adjoint problem forced not by a measurement but by the first-order adjoint and
the viscosity perturbation. A Hessian-vector product of any objective on the
map costs four runs: the forward run, the first-order adjoint, a tangent-linear
run and the second-order adjoint sweep. Under migrating shorelines the
cancellation that makes the first derivative exact fails at second order, and
Yu et al.'s second-order load and Hessian kernels carry line distributions on
the shorelines, which the library omits: what it computes is the second
derivative of the map with the ocean functions held, exact under frozen
shorelines. Their approximate kernels, which drop the second-order variables,
are the Gauss-Newton Hessian.

The remedy on the grid is to make the ocean function differentiable. With
``flotation_width`` set on an ``EarthState``, the flotation condition is
smoothed over that height: with :math:`x = (S - \rho_i I/\rho_w)/w` the
ocean function is :math:`\sigma(x)`, the quintic smoothstep over
:math:`[-1, 1]`, so that a point more than :math:`w` from floating its ice is
land or ocean exactly and the rest are partly ocean. Every load carries the
variation of :math:`C` multiplied by the flotation margin
:math:`\rho_w S - \rho_i I = \rho_w w x`, and the product is
:math:`\rho_w\Lambda\,\delta y` with :math:`\Lambda = x\sigma'(x)` and
:math:`y = \xi + S_0 - \rho_i I/\rho_w`: the derivatives see the effective
ocean function :math:`\hat C = C + \Lambda` in the feedback and in every
grounding factor, which is the cancellation of the sharp rule turned into a
bounded term on the strip, and the second derivative gains the load
:math:`\rho_w K\,\delta_u y\,\delta_v y` with
:math:`K = d\hat C/dS = (2\sigma' + x\sigma'')/w`, continuous since
:math:`\sigma` is twice continuously differentiable. The generalised solver
takes the ocean function as a weight, so nothing changes in it; the
tangent-linear run, both adjoints and both second-order sweeps take
:math:`\hat C` and :math:`K` from the trajectory, and the Hessian is then
complete, at the price of shorelines smoothed over :math:`w` in the forward
run, which tends to the sharp one as :math:`w \to 0`.

``SeaLevelEvolutionOperator`` is the evolution as a ``pygeoinf`` operator on
the ice history, the initial sea level and the log viscosity of every branch,
each an element of a Hilbert space of functions: a ``FieldHistorySpace``, a
sphere space, and a ``LogViscositySpace``, one Sobolev space of radial
profiles per layer in which the branch relaxes, continuous within a layer and
free to jump at the material interfaces. It supplies its derivative and its
second derivative by either method, so that ``pygeoinf``'s Newton methods,
Laplace approximations and exact Hessians apply to any objective built on
it, and ``fixing`` holds arguments fixed by composition, giving the linear
``IceHistoryOperator`` under frozen shorelines with the other two fixed.

Decay times
^^^^^^^^^^^

A decay time is not a field value but the parameter of a curve
:math:`a\,e^{-st} + b` fitted to a sea level series, so its sensitivity comes
from differentiating the fit's stationarity conditions, which turns it into
time-dependent weights on the series and hence into a multi-date measurement.
Following Al-Attar & Tromp (2014, Appendix C) the model times join the fit as
parameters penalised towards the nominal ones, which accounts for the timing
uncertainty of sea level index points (``fit_decay_time``,
``decay_time_measurement``). The datum belongs near the centre of a former ice
sheet and after the melting has stopped, where rebound curves are close to
exponential.

Differentiating the stationarity conditions once more gives the second
derivative of the decay time with respect to the series. ``DecayTimeFunctional``
is the decay time as a functional on the series' values, with those weights as
its gradient and that second derivative as its Hessian. Composed with an
operator :math:`S` that samples a response at the fit dates and with the
evolution operator :math:`F`, it is a nonlinear objective
:math:`\tau \circ S \circ F` whose Hessian follows by the chain rule: the
decay time's own Hessian carried through the first derivatives, plus the
second derivative of :math:`F` weighted by the fit's weights.


The initial sea level
---------------------

An ice model that carries no sea level, unlike the ICE-NG products, leaves the
initial sea level :math:`S_0` to be found, and the standard device is to choose
it so that the run reproduces the observed present-day sea level, or, for a
test body, any final sea level :math:`S_f` at the last date :math:`t_f`. The
run from :math:`S_0` reaches the final sea level

.. math::

   \mathcal{H}(S_0) = S_0 + \xi(t_f;\, S_0),

and the calibration solves :math:`\mathcal{H}(S_0) = S_f` or, more generally,
minimises the misfit

.. math::

   J = \frac12\int w\,\big(\mathcal{H}(S_0) - S_f\big)^2\,dS

with non-negative weights :math:`w`. Its linearisation is
:math:`\mathcal{H}' = I + A`, where :math:`A = \partial\xi(t_f)/\partial S_0` comes
only from the dependence of the loads on :math:`S_0`.

Under frozen shorelines the problem is linear. The shorelines are those of the
starting state, held fixed, so :math:`\xi` does not depend on :math:`S_0` at
all, :math:`A = 0`, and one run gives the answer exactly:
:math:`S_0 = S_f - \xi(t_f)` wherever the weights are positive. The calibrated
state keeps those shorelines as a prescribed ocean function, so that a frozen
run from it reproduces :math:`S_f`; were the ocean function instead re-derived
from each new :math:`S_0` by flotation, the cells near the flotation limit
could flip back and forth between iterations without settling.

Under migrating shorelines the initial shorelines follow from :math:`S_0` by
flotation, and each later load contains :math:`\rho_w(C_n - C_0)S_0`, so
:math:`A` is confined to where the shorelines move, and small, and the
calibration iterates. ``SeaLevelEvolution.calibrate_initial_sea_level`` offers
two methods, both accelerated by Anderson mixing, and ``run_ice_history(...,
final_sea_level=...)`` calibrates before it runs.

* The fixed point, the classical update (Kendall et al. 2005), with
  :math:`r = \mathcal{H}(S_0) - S_f` the residual:

  .. math::

     S_0 \leftarrow S_0 - r = S_f - \xi(t_f;\, S_0).

  It is Newton's method for :math:`\mathcal{H}(S_0) = S_f` with the Jacobian
  :math:`I + A` replaced by the identity, so it contracts at the rate set by
  :math:`A`; with :math:`A = 0` it is the frozen-shoreline solution above.
* Steepest descent with the adjoint gradient,
  :math:`S_0 \leftarrow S_0 - \nabla J`, where

  .. math::

     \nabla J = (I + A^{\mathsf T})\,w\,r

  is the weighted residual plus the initial sea level kernel of a sea level
  measurement :math:`w\,r` at :math:`t_f` (Yu et al. 2025, eq. 87). With
  :math:`A = 0` and unit weights it is the classical update exactly, the
  observation of Yu et al. (2025, Section 3.9) that the classical scheme
  approximates steepest descent. Under migrating shorelines it costs an adjoint
  sweep per iteration, and unit steps contract at about the rate set by
  :math:`A + A^{\mathsf T}`, more slowly than the classical update, which
  inverts the Jacobian approximately where the gradient applies its
  transpose. Its use is for objectives the classical update does not serve,
  such as a comparison with the present that is weighted or confined to where
  the present-day topography is trusted.

The connection with Kalman filtering makes the place of each method clear.
Regard :math:`S_0` as a state with prior mean :math:`S_b` and covariance
:math:`P`, and the final sea level as an observation
:math:`y = \mathcal{H}(S_0) + \varepsilon` with noise covariance :math:`R`. The
Kalman update, iterated about the current estimate (the iterated extended
Kalman filter, equivalent to Gauss-Newton on the posterior), is

.. math::

   S_0 \leftarrow S_b + K\big[y - \mathcal{H}(S_0) - \mathcal{H}'(S_b - S_0)\big],
   \qquad K = P\,\mathcal{H}'^{\mathsf T}\big(\mathcal{H}'P\,\mathcal{H}'^{\mathsf T} + R\big)^{-1}.

For a perfect and complete observation of the final state, :math:`R \to 0`,
the gain becomes :math:`K = \mathcal{H}'^{-1}`, the prior drops out, and the
iteration is Newton's method for :math:`\mathcal{H}(S_0) = y`; with
:math:`\mathcal{H}'` replaced by the identity, :math:`K = I` and it is the
classical update. The classical scheme is therefore an iterated Kalman update
of unit gain, exact in a single step under frozen shorelines. The adjoint supplies the action of
:math:`\mathcal{H}'^{\mathsf T}` that the gain needs beyond that case: in the
language of data assimilation the calibration is a strong-constraint
variational problem for the initial state, which for linear Gaussian problems
coincides with the Kalman smoother, and the adjoint run is the smoother's
backward sweep. An imperfect or incomplete present-day field makes the gain a
genuine blend of prior and data, :math:`K = P(P + R)^{-1}` when
:math:`\mathcal{H}' = I`; the weights of the misfit are the simplest instance,
the inverse of a diagonal noise covariance without a prior. The full Bayesian
estimate is a natural application of ``pygeoinf``.


Observations
------------

Observables and sampling
^^^^^^^^^^^^^^^^^^^^^^^^

An observation model in ``pyslfp.observations`` keeps the physics apart from
the measurement. Its observable :math:`O` is a linear operator from the model
space into the space of the quantity observed (a field, a surface
displacement, the coefficients of the potential change), its sampling
:math:`S` takes that quantity to the data, and the data are

.. math::

   d = S\,O\,u + e,

with the noise :math:`e` a Gaussian measure on the data space, or independent
noise of given standard deviations. Whatever builds the observable holds the
physics, the direct terms of a model set-up included, so one observation model
serves a fingerprint operator, a set-up and ``IceHistoryOperator`` alike; a
rate is the difference of the responses at two observation dates over their
interval.

* Tide gauges observe the relative sea level change :math:`\xi` at the
  stations.
* Satellite altimetry observes the geocentric sea level change
  :math:`\xi + U`, the height of the sea surface measured from the centre of
  mass. By default the height of the centrifugal potential perturbation,
  :math:`\psi/g`, is removed from it (``remove_rotational_contribution``).
* GRACE observes the coefficients of the potential change between two
  degrees, read directly from the response's components, so that no synthesis
  and analysis lie between the solver and the data. Level-2 products give the
  Stokes coefficients of the potential, positive near mass, at their reference
  radius :math:`a` and gravitational parameter :math:`GM`:
  :math:`\Delta C_{\ell m} = -(a/GM)(b/a)^{\ell+1}\phi_{\ell m}/\sqrt{4\pi}`,
  and :math:`\Delta S_{\ell m}` likewise.
* GNSS observes the surface displacement at the stations in east, north and
  up, its horizontal part :math:`\nabla_1 V` included, in the frame of the
  data (below).

A set-up describes one epoch. ``HistorySetup`` is the same vocabulary
through time: the tensor product of a time space and the set-up's space, on
which every operator of the set-up acts at each time by a lift, at one
application per time mode, and a history is sampled by a contraction with a
functional on the time factor: its value at a time, bounded for a time
order above one half; its mean over a window; or its change over a span per
unit time, which is what a velocity or a trend observed over years is, and
needs no more of the time space than evaluation does. The response to a
fingerprint operator at every epoch is such a lift, the elastic physics
through time, and the ice history of a set-up is the argument of the
viscoelastic operators.

Evaluation at a point is bounded only on fields of Sobolev order above one on
the sphere, so the observables of tide gauges, altimetry at points and GNSS
must take values in such a space. The displacement's norm weights the
coefficients of :math:`V` by :math:`\ell(\ell+1)`, so :math:`\nabla_1 V` has
the regularity of :math:`U` and one order serves both. A finite set of
coefficients is bounded on any of the spaces.

The frame of GNSS data
^^^^^^^^^^^^^^^^^^^^^^

A response's displacement is in the frame of the centre of mass. A GNSS
solution realises its own frame, which differs from it by a Helmert
transformation: a translation :math:`\mathbf{T}`, a small rotation
:math:`\mathbf{R}` and a change of scale :math:`s`, or their rates for
velocities. At a station :math:`\mathbf{x} = b\,\hat{\mathbf{r}}` it adds

.. math::

   \delta\mathbf{u} = \mathbf{T} + s\,\mathbf{x} + \mathbf{R}\times\mathbf{x},

whose east, north and up components are

.. math::

   \delta u_e = \mathbf{T}\cdot\hat{\mathbf{e}} + b\,\mathbf{R}\cdot\hat{\mathbf{n}},
   \qquad
   \delta u_n = \mathbf{T}\cdot\hat{\mathbf{n}} - b\,\mathbf{R}\cdot\hat{\mathbf{e}},
   \qquad
   \delta u_r = \mathbf{T}\cdot\hat{\mathbf{r}} + b\,s.

As a field on the sphere the translation is
:math:`(\mathbf{T}\cdot\hat{\mathbf{r}})\,\hat{\mathbf{r}} +
\nabla_1(\mathbf{T}\cdot\hat{\mathbf{r}})`, of degree 1 with :math:`V = U`: a
change of origin changes the degree-1 displacement by equal amounts in
:math:`U` and :math:`V`, and only their difference is independent of it. The
scale is a degree-0 vertical field. The rotation is a rigid toroidal motion,
which no response holds, the response of a spherically symmetric Earth to a
surface load being spheroidal. ``HelmertTransformation`` is the design matrix
:math:`A` from the :math:`k` chosen parameters to the station data, with the
translation a length, the rotation in radians and the scale a ratio, in
Earth-fixed axes.

``GnssObservationModel`` takes the frame in one of three ways.

* Without a frame, the data are taken to be in the centre-of-mass frame.
* As nuisance parameters, :math:`h` joins the model space,
  :math:`d = F u + A h + e` on the direct sum, and a prior on :math:`h`,
  independent of that on :math:`u`, states how well the frame is known.
* Frame-invariant, the data are reduced to :math:`Z^{\mathsf T} d`, with
  :math:`Z` an orthonormal basis of the orthogonal complement of the range of
  :math:`A`. Since :math:`Z^{\mathsf T}A = 0`, the reduced data
  :math:`Z^{\mathsf T}F u + Z^{\mathsf T}e` do not depend on the frame, and
  their noise has the covariance :math:`Z^{\mathsf T}CZ`, of full rank
  :math:`3N - k` for :math:`N` stations.

The invariant form is the nuisance form with no prior information on the
frame. Integrating the nuisance likelihood over :math:`h` under a flat prior
leaves the quadratic form of

.. math::

   C^{-1} - C^{-1}A\,(A^{\mathsf T}C^{-1}A)^{-1}A^{\mathsf T}C^{-1}
   = Z\,(Z^{\mathsf T}CZ)^{-1}Z^{\mathsf T}

in :math:`d - F u`, which is the likelihood of :math:`Z^{\mathsf T}d`. A flat
prior is the singular limit of the nuisance form; the invariant form reaches
it with a covariance of full rank and :math:`k` fewer data. The nuisance form
is the one to use when the frame is partly known, or its parameters are
wanted.


References
----------

* Al-Attar, D. & Tromp, J., 2014. Sensitivity kernels for viscoelastic loading
  based on adjoint methods. *Geophys. J. Int.*, 196, 34-77.
* Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. & Lloyd, A.J., 2024.
  Reciprocity and sensitivity kernels for sea level fingerprints. *Geophys. J.
  Int.*, 236, 362-378.
* Backus, G.E., 1970. Inference from inadequate and inaccurate data. *Proc.
  Natl. Acad. Sci.*, 65, 1-7.
* Bell, B.M. & Cathey, F.W., 1993. The iterated Kalman filter update as a
  Gauss-Newton method. *IEEE Trans. Automat. Contr.*, 38, 294-297.
* Crawford, O., Al-Attar, D., Tromp, J., Mitrovica, J.X., Austermann, J. &
  Lau, H.C.P., 2018. Quantifying the sensitivity of post-glacial sea level
  change to laterally varying viscosity. *Geophys. J. Int.*, 214, 1324-1363.
* Farrell, W.E., 1972. Deformation of the Earth by surface loads. *Rev.
  Geophys.*, 10, 761-797.
* Farrell, W.E. & Clark, J.A., 1976. On postglacial sea level. *Geophys. J. R.
  astr. Soc.*, 46, 647-667.
* Kendall, R.A., Mitrovica, J.X. & Milne, G.A., 2005. On post-glacial sea
  level - II. Numerical formulation and comparative results on spherically
  symmetric models. *Geophys. J. Int.*, 161, 679-706.
* Milne, G.A. & Mitrovica, J.X., 1998. Postglacial sea-level change on a
  rotating Earth. *Geophys. J. Int.*, 133, 1-19.
* Mitrovica, J.X., Wahr, J., Matsuyama, I. & Paulson, A., 2005. The rotational
  stability of an ice-age earth. *Geophys. J. Int.*, 161, 491-506.
* Yu, Z., Al-Attar, D., Syvret, F. & Lloyd, A.J., 2025. Application of first-
  and second-order adjoint methods to glacial isostatic adjustment
  incorporating rotational feedbacks. *Geophys. J. Int.*, 240, 329-348.
