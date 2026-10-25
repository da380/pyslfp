Numerics and performance
========================

This page describes how the equations of :doc:`theory` are discretised and
solved, what controls their accuracy, what they cost, and how to spend the
cores of a machine on them. The timings quoted were measured on an AMD Ryzen AI
7 PRO 350 laptop (8 physical cores, 16 hardware threads), one core unless
stated, and are indicative only.


Spherical harmonics and grids
-----------------------------

Surface fields are ``pyshtools`` grids and coefficient sets, truncated at the
degree ``lmax`` of the ``EarthModel``. Driscoll-Healy grids (``"DH"``, and the
twice-oversampled ``"DH2"``) and Gauss-Legendre grids (``"GLQ"``) are
supported. Surface integrals are evaluated in grid space with the exact
quadrature weights of the grid (Driscoll-Healy or Gauss-Legendre), which agree
with the degree-0 coefficient for band-limited fields and are also exact for
products such as :math:`C\xi`, which are not band-limited to ``lmax``.

The transforms dominate the cost of a sea level solve, at well over ninety per
cent at degree 256, the Python around them being negligible. ``pyshtools``
uses the multi-threaded ``ducc0`` backend, a declared dependency, which is
several times faster than its default at degree 256 and more so above.


Solving the sea level equation
------------------------------

The linear and generalised solvers iterate on the coefficients of the load.
Each evaluation of the fixed-point map multiplies the load coefficients by the
combined factor :math:`-(h_\ell + k_\ell/g)`, replaces the degree-2 and degree-0
coefficients by the closed-form rotational solution, synthesises the sea level
change with one inverse transform, applies the mass-conservation shift with
the grid quadrature, multiplies by the ocean function and returns to
coefficients with one forward transform. The displacement and potential are
synthesised once, from the converged load. The iteration starts from the
eustatic load.

The fixed point is accelerated by Anderson mixing (``AndersonMixer``): the
update combines the last :math:`p` iterates so as to minimise the linearised
residual, the small least-squares problem being solved through its Gram
matrix, updated by one row and column per iteration, with a minimum-norm solve
guarding against rank deficiency. The default memory is
``SeaLevelEquation.anderson_memory = 5``; zero recovers the plain Picard
iteration. The iteration stops when

.. math::

   \frac{\max|C\xi^{(k)} - C\xi^{(k-1)}|}{\max|C\xi^{(k)}|} \le \texttt{rtol},

with ``rtol = 1e-9`` by default, and warns if ``max_iterations`` is reached
first. Because the rotational feedback is solved exactly inside every
iteration, the returned angular velocity, displacement and potential are all
consistent with the converged load. The adjoint solve,
``adjoint_generalised_solve``, is the exact reverse mode of the same code path,
a transposed fixed point run through the same mixer.

The nonlinear solver with migrating shorelines is a plain Picard iteration on
the sea level and ocean function, stopped when the change of :math:`\xi`
between iterations, measured over the whole grid, falls below ``rtol`` relative
to its size, with at most 50 iterations by default.

Measured times for one linear solve, the fingerprint of a ten per cent West
Antarctic melt on PREM with the ICE-7G background:

=======  ================  ==================
lmax     one thread        eight threads
=======  ================  ==================
128      26 ms             17 ms
256      220 ms            120 ms
512      560 ms            290 ms
=======  ================  ==================


Love numbers
------------

The radial problem at each degree is discretised by spectral elements with
``ngll`` Gauss-Lobatto-Legendre nodes (five by default, so quartic elements),
whose edges include every discontinuity of the model; all integrals use the
same nodes as quadrature. Solid nodes carry :math:`(U, V, \phi)`, fluid nodes
:math:`\phi` alone, and the degrees of freedom are numbered node by node, so the
matrix is banded. It is scaled symmetrically by its diagonal, which equalises
the very different units of the displacement and potential unknowns, and
factorised by banded Cholesky, with a fallback to banded LU (the
gravitational and stratification terms do not guarantee positive
definiteness; complex systems always use LU). One factorisation per degree
serves every forcing.

At high degree the solution decays with depth roughly as
:math:`(r/a)^{\ell+1}`, so each degree is solved only above the depth where it
falls below the fraction ``eps`` (by default :math:`10^{-8}`) of its surface
value, closed below by a Dirichlet-to-Neumann term. The default mesh for a
table to degree :math:`L` is graded to match: element widths of
:math:`0.1a/(L+1)` near the surface, growing with depth as the highest degree
that reaches there falls, so the whole mesh has a few hundred elements rather
than the :math:`10(L+1)` a uniform mesh would need. ``graded_mesh`` builds it.
Computing the table for PREM takes about 2 s to degree 256 and 8 s to degree
1024.


Viscoelastic time stepping
--------------------------

The state and its cost
^^^^^^^^^^^^^^^^^^^^^^

The internal variables are stored per branch and per degree over that degree's
part of the radial mesh, three scalars per node and per spherical harmonic
coefficient, ordered so that each degree's solve is one banded back-substitution
with the coefficients as right-hand-side columns. For a one-branch Maxwell
mantle on the default graded PREM mesh this is about 0.1 GB at degree 64,
0.5 GB at degree 128 and 2.1 GB at degree 256.

Each step size needs its own effective system per degree, factorised once and
kept for the four most recent ``(scheme, dt)`` pairs, and its own table of
effective Love numbers, cached for the run. A new step size therefore costs an
assembly, a factorisation and a Love-number table; a repeated one costs
neither. The load and centrifugal right-hand sides are rank one, so each degree
keeps one unit solution per step size and adds the step's responses by outer
products, and the memory force is scattered by a precomputed sparse matrix. A
coupled step then costs one banded solve per degree, for the memory field, and
one sea level solve. Measured with ICE-7G on a two-layer Maxwell PREM, a step
takes about 0.5 s at degree 64 and 2.1 s at degree 128, and the first step of a
new size three to four times that; the ICE-7G run of tutorial 7, 52 steps at
degree 128, takes under two minutes.

Accuracy
^^^^^^^^

The exponential trapezoid is second order and backward Euler first order, and
both are exact for relaxation under prescribed strain of the corresponding form
at any step size, so the error comes from the coupling, not the relaxation
itself. The step should resolve the shortest relaxation time that matters to
the result; ``Rheology.min_relaxation_time`` gives the shortest in the model.
Under migrating shorelines the per-step fields are consistent to the tolerance
of the nonlinear iteration rather than eliminated exactly, and convergence
order is best judged on the frozen path, where the discretisation is smooth.

Adaptive steps
^^^^^^^^^^^^^^

``SeaLevelEvolution.run_adaptive`` (and ``ViscoelasticOperator.integrate`` for
the radial problem alone) chooses the step from the difference between the
trapezoid's update of the internal variables and that of the embedded
exponential Euler companion, which costs nodal arithmetic only. The error is
the root mean square of that difference over :math:`\mathtt{atol} +
\mathtt{rtol}\max|\mathbf{m}|`, and the next step is the current one times
:math:`0.9\,\mathrm{err}^{-1/2}`, clamped to :math:`[0.2, 4]`. The absolute
floor follows the state's own magnitude by default, since the internal
variables carry the units of the degree-basis scalars. Because a new step size
costs factorisations and a Love-number table, proposed steps snap down onto
the ladder :math:`\Delta t_0\,2^n` by default, with a promotion of one rung after
eight accepted steps that would have allowed a larger one; rounding to the
nearest rung instead makes the controller oscillate. The estimate is that of
the first-order companion, so it is conservative for the second-order
solution: on the test bodies of the suite, a step tolerance of
:math:`10^{-1}` gives a global error near :math:`3\times10^{-4}`, and
:math:`10^{-2}` near :math:`7\times10^{-6}`.


Adjoints
--------

A reversed sweep of either adjoint costs about as much as the forward run over
the same dates: one symmetric banded solve per degree per step, and one sea
level solve. Measured over ten steps, the discrete adjoint took 3.9 s at degree
64 and 18 s at degree 128, against 5.2 s and 21 s for the forward steps. Under
frozen shorelines the discrete adjoint needs no forward run at all. Under
migrating shorelines it needs the forward run's ocean functions, one grid per
date, which the snapshots record.

The viscosity kernels need the forward trajectory of internal variables, which
at the sizes above cannot be kept at every date at high degree. A run with
``record=N`` keeps the internal state every :math:`N` dates, together with the
surface load and centrifugal coefficients of every date, which are small. The
reverse sweep reconstructs the states in between by stepping backwards: the
exact inverse of an exponential trapezoid step is the same update with the step
negated, and backward Euler inverts explicitly. Running dissipation backwards
amplifies rounding by :math:`e^{h}` per branch and step, so the sum of
:math:`h = \Delta t/\tau` across a gap is the number of digits spent. In
practice a resolved gap reconstructs to about :math:`10^{-12}`, and a gap of
several relaxation times to about :math:`10^{-11}`.

The disagreement between the discrete and continuous adjoints measures the
resolution of the forward run (:doc:`theory`). At half-kiloyear steps on a
two-layer Maxwell PREM it is under one per cent for the ice kernels of tutorial
8 and a few per cent for the viscosity kernel of tutorial 9.

Kernels have the regularity of the measurements that drive them. A measurement
patch with a sharp edge rings through every relaxation mode at a finite
truncation, so patches should be tapered in degree, as the tutorials do.


Calibrating the initial sea level
---------------------------------

Under frozen shorelines ``calibrate_initial_sea_level`` makes one forward run
and solves the linear problem exactly, by either method. Under migrating
shorelines each iteration is one forward run, plus, for the gradient method,
one adjoint sweep of about the same cost. The iteration stops when the largest
update of the initial sea level falls below ``rtol`` times its largest value
(``1e-4`` by default, of order a metre on the Earth), and returns the last
state it ran, with that run, so the two are consistent; it warns if the
iterations run out first.

On the migrating-shoreline test body of the suite, where the melting of a
marine ice cap moves the shorelines, the unaccelerated classical update reduces
the misfit by a factor of about 0.08 per iteration and unit steps down the
gradient by about 0.33, as the theory leads one to expect. With Anderson mixing,
the default, the classical update recovers the known initial sea level to a
relative error below :math:`10^{-12}` in six runs, and the gradient method to a
few times :math:`10^{-12}` in eight. Calibrating the thickness of ICE-7G from
26 ka against the present day at degree 64 (tutorial 10), the classical update
reduces the misfit from about 150 m to about a centimetre in five runs, and the
gradient method to a few centimetres in seven.


Threads and processes
---------------------

Two libraries do the parallel work in a sea level calculation: the transforms
(``ducc0``) and BLAS. Both take their thread count from the
``OMP_NUM_THREADS`` environment variable when Python starts, all cores if it
is unset. For a single interactive calculation the default is what is wanted,
though the transforms scale modestly, giving about a factor of two on eight
cores at degree 256 to 512.

For many independent solves, as in the ``parallel`` options of the
``pygeoinf`` operators, processes scale far better than threads. Set
``OMP_NUM_THREADS=1`` before starting Python, so that each worker runs
single-threaded and the cores are shared between workers rather than
oversubscribed. The main process can still use several cores for its serial
phases by calling ``pyslfp.set_num_threads(n)``; workers started afterwards are
unaffected, since they read the environment when they start.
``pyslfp.physical_core_count`` counts physical cores, since simultaneous
multithreading gives the transforms nothing.

``ViscoelasticOperator(threads=n)`` runs the degree loops of the radial steps
in a thread pool, with results identical to the serial path. The per-degree
work, banded solves with many right-hand sides and array arithmetic, is
limited by memory bandwidth rather than arithmetic, so the gain is small: a
few per cent at degrees 64 to 128, and of order fifteen per cent at degree 256.
Serial is the default.


How the numerics are checked
----------------------------

The test suite holds the numerics to analytical and structural checks rather
than to stored outputs:

* Love numbers against the closed forms of homogeneous spheres, the
  reciprocity relations to rounding, and reference PREM values;
* the viscoelastic weights against prescribed-strain relaxation, exact at any
  step; the elastic and fully relaxed limits; the correspondence principle
  against frequency-domain numbers of a model frozen at a frequency; the
  convergence order of each scheme; and the elastic fingerprint recovered when
  every relaxation time is infinite;
* mass conservation of the coupled evolution, under frozen and migrating
  shorelines;
* dot-product tests of every adjoint, gradient identities for the kernels at
  machine precision, and central finite differences converging at second order
  about migrating-shoreline trajectories;
* covariance gates: the same physical problem run under deliberately different
  non-dimensional schemes, compared after conversion to SI, which catches any
  factor dropped because the default scheme makes it one.
