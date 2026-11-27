# Formulation of the experiments

This note states the formulation that the scripts in this folder implement. It
uses the notation of Heathcote et al. (2026) and the terms of Gregory et al.
(2019). All calculations are non-dimensional, in the units of the Earth
model; the scripts report results in millimetres of water or of sea level.

## Spaces

The sphere has radius $a$. $H^s$ is the Sobolev space of order $s$ on the
sphere with length scale $\lambda_s$, with the norm

$$\|u\|_{H^s}^2 = \sum_{lm} \left(1 + \lambda_s^2 \frac{l(l+1)}{a^2}\right)^{s} |u_{lm}|^2 ,$$

for $u_{lm}$ the coefficients in fully normalised real harmonics. Throughout,
$\lambda_s = 500$ km. Point evaluation is continuous on $H^s$ for $s > 1$.

A covariance $\alpha (1 + \lambda^2 \Delta)^{-q}$ is always an operator on the
space it is stated on. On $H^s$ the coefficients of a sample then have the
variances

$$\mathrm{var}(u_{lm}) = \alpha \left(1 + \lambda^2 \frac{l(l+1)}{a^2}\right)^{-q}
\left(1 + \lambda_s^2 \frac{l(l+1)}{a^2}\right)^{-s} ,$$

so the same $\alpha$ and $q$ on another space give a different random field.
Every prior and every spatial noise field below is built on $H^2$ with
$\lambda_s = 500$ km. The amplitude $\alpha$ is never given directly. It
follows from a stated pointwise standard deviation, the standard deviation of
the field's value at any one point.

## The sea level equation

A direct load $\zeta$ gives the response $(\xi, u, \phi, \omega)$: the
relative sea level change, the displacement of the solid surface, the change
in gravitational potential, and the change in angular velocity. $\psi$ is the
centrifugal potential that $\omega$ implies. The total load is

$$\sigma = \zeta + \rho_w C \xi ,$$

with $C$ the ocean function. The response includes the ocean load that
$\zeta$ induces, so every operator below acts on $\zeta$ and never on
$\sigma$. Mass conservation, $\int \sigma \, dS = 0$, leaves $\phi$ with no
degree 0.

## GRACE experiments (grace_bias, grace_inversion)

**Model.** The model is the direct load $\zeta \in H^2$, a global field. The
response is in $L^2$.

**Data.** GRACE observes the coefficients of $\phi$ for degrees
$2 \le l \le L$, with $L = 100$:

$$d = A \zeta + z, \qquad A = G \circ \Phi ,$$

where $\Phi$ is the sea level equation's map from $\zeta$ to $\phi$ and $G$ takes the
coefficients. Degree 0 is never observed. Degree 1 is not observed either; it
is constrained only through the ocean load that the sea level equation
couples to the observed degrees.

**Prior.** The prior is $\zeta \sim N(0, Q)$ with
$Q = \alpha (1 + \lambda^2 \Delta)^{-q}$ on $H^2$, $q = 3$ and $\lambda = 250$
km. The pointwise standard deviation is 10 mm of equivalent water thickness,
which is a load of $\rho_w \times 10$ mm. Optionally (`--remove-degree-1`),
the prior is conditioned exactly on $\zeta_{1m} = 0$ for $m = -1, 0, 1$. This
conditions degree 1 only. The prior may be translated by a multiple of one of
its own samples (`--prior-shift`).

**Noise.** The noise is a noise load
$z_\sigma \sim N(0, Q_n)$, $Q_n = \alpha_n (1 + \lambda_n^2 \Delta)^{-q_n}$
on $H^2$, with $\lambda_n = \lambda / 2$. The Love numbers take it to the
coefficients: $z_{lm} = k_l (z_\sigma)_{lm}$. The two unknowns $\alpha_n$
and $q_n$ follow in closed form from the ratio of the spectra,

$$s_l = \frac{\alpha}{\alpha_n}
\left(1 + \lambda^2 \frac{l(l+1)}{a^2}\right)^{-q}
\left(1 + \lambda_n^2 \frac{l(l+1)}{a^2}\right)^{q_n} ,$$

from two conditions: $\sqrt{s_2} = 10$, and $\sqrt{s_l} = 1$ at the crossover
degree $l = 64$. The signal and the noise are both loads, on the same space, so the
ratio involves neither the units nor the metric. With these values,
$q_n = 1.44$, and at degree 256 the noise load has a pointwise standard
deviation of 1.28 mm of water; the amplitude signal-to-noise ratio at degree
100 is 0.48.

**Quantities of interest.** These are the averages of the total load over four
regions, $q = B\zeta$, with weights equal to the region masks smoothed by a
heat kernel at 500 km and divided by their integrals. They also include the degree-1
coefficients of $\zeta$. The WMB estimate is $\hat q = C d$, with

$$\hat\sigma = \sum_{l=2}^{L} \sum_m k_l^{-1} d_{lm} Y_{lm} .$$

**Errors and posterior.** The WMB error is
$\hat q - q \sim N\big(0, (CA - B) Q (CA - B)^* + C R C^*\big)$. The
posterior is the standard Gaussian posterior. The data-space normal equations
are solved by conjugate gradients, preconditioned by the WMB preconditioner
$\phi_{lm} \mapsto (q_l k_l^2 + r_l)^{-1} \phi_{lm}$. The solves stop at a
relative residual of 0.01 times the ratio of the noise's pointwise standard
deviation to the signal's.

## Ocean experiments (altimetry_bias, altimetry_inversion, joint_inversion)

**Model.** The model is $m = (h, \eta_d, \delta\rho)$: the ice thickness
change, the sterodynamic sea level change and the vertically averaged ocean
density change. Each is a global field in $H^2$. Only their restrictions to
their supports have physical meaning. The supports are the ice sheets
$\chi_I$ for $h$ and the ocean $C$ for the two ocean fields. The ice sheets
are all the ice south of 60°S together with all the ice in GTN-G region 5
(Greenland).

**Load.** The direct load is

$$\zeta = \rho_i (1 - C) \chi_I h + C (\rho_w \eta_d + \eta_0 \delta\rho) ,$$

with $\eta_0$ the ocean depth. The factor $(1 - C)$ removes floating ice. The
supports enter only here and in the operators built from the load. The
indicators make the load discontinuous, so the loads lie in $H^s$ only for
$s < 1/2$. The calculations take $s = 1/4$. The response then lies in
$H^{s+1} = H^{5/4}$, on which point evaluation is continuous.

**Constraint.** The ocean fields cannot change the ocean's mass:

$$D m = \int C (\rho_w \eta_d + \eta_0 \delta\rho) \, dS = 0 .$$

**Geocentric sea level change.** Altimetry observes

$$\eta = \eta_d + \xi + u + \psi / g ,$$

which is the geocentric sea level change with the rotational contribution removed, as
altimetry processing does. The observations lie on a regular 1° grid over the ice-free ocean,
within 66° of the equator.

**Global and regional means.** For the ocean average
$\langle f \rangle_O = A_O^{-1} \int C f \, dS$:

- the global mean sea level change is $\mathrm{GMSL} = \langle \eta_d + \xi \rangle_O$;
- the steric sea level change is $\eta_s = -(\eta_0 / \rho_w) C \delta\rho$, with
  global mean $s = \langle \eta_s \rangle_O$;
- the barystatic sea level change is
  $b = -\rho_i (\rho_w A_O)^{-1} \int (1 - C) \chi_I h \, dS$.

Mass conservation gives $\mathrm{GMSL} = b + s$. The scripts compute $b$ and $s$
directly and report the difference as a check on the sea level equation's mass
balance. Regional means split the relative sea level change $\eta_d + \xi$
into three parts:

- the steric sea level change, $\eta_s$;
- the dynamic manometric sea level change, $(\eta_d - \eta_s) + \xi[\zeta_o]$;
- the barystatic manometric sea level change, $\xi[\zeta_i]$.

Here $\zeta_i$ and $\zeta_o$ are the ice and ocean parts of $\zeta$. The
scripts average them over the Tasman Sea.

**Prior.** The fields have the joint covariance

$$Q = \begin{pmatrix}
\alpha_i K_i & 0 & 0 \\
0 & \alpha_d K_d & -\beta \sqrt{\alpha_d \alpha_w}\, e^{-\mu^2 \Delta} K_d^{1/2} K_w^{1/2} \\
0 & \cdot & \alpha_w K_w
\end{pmatrix}, \qquad K_k = (1 + \lambda_k^2 \Delta)^{-q},$$

on $H^2 \oplus H^2 \oplus H^2$. The parameters are $q = 3$; $\lambda_i = 500$
km, $\lambda_d = 100$ km and $\lambda_w = 500$ km; $\beta = 0.9$; and
$\mu = 200$ km. The prior is $N(0, Q)$ conditioned exactly on $D m = 0$. The
fields are not masked: $D$ carries the ocean function, so conditioning
commutes with restriction to the supports. Every statistic within the
supports, and every quantity of interest, is therefore that of the prior
restricted first and conditioned afterwards. The amplitudes follow a chain with
one dimensioned input:

1. $\alpha_d$ gives $\eta_d$ a pointwise standard deviation of 4 mm.
2. $\alpha_w$ sets the pointwise standard deviation of
   $(\bar\eta_0 / \rho_w) \delta\rho$ to 0.75 times that of $\eta_d$, where
   $\bar\eta_0$ is the mean ocean depth.
3. $\alpha_i$ makes the standard deviation of $b$ 1.7 times that of $s$, both
   under the conditioned prior. The ice is uncorrelated with the ocean fields
   and absent from $D$, so the chain closes without iteration.

**Altimetry noise.** The altimetry noise is for a change over one year and
is the sum of four parts, with standard deviations derived from the error
budgets of Ablain et al. (2019) and Prandi et al. (2021):

- a local part, independent at each point, with standard deviation 6 mm;
- a large-scale part, a field
  $N(0, \alpha_c (1 + \lambda_c^2 \Delta)^{-q})$ on $H^2$ evaluated at the
  points, with $\lambda_c = 2000$ km and a pointwise standard deviation of
  0.7 mm;
- an error common to every point, with standard deviation 1 mm;
- an error in the origin of the reference frame, three degree-one patterns
  with standard deviation 0.33 mm each.

All but the first barely average down, and so set a floor on the error of
the GMSL estimate. The scripts report the floor of the large-scale part.
The values of the submitted paper were 2 mm, 0.04 mm and no common or frame
error.

**GRACE noise in the joint experiment.** This is a noise load, as in the GRACE
experiments, with $\lambda_n = 250$ km, solved by the same two conditions
($\sqrt{s_2} = 10$, crossover at degree 64). Here the signal is the ice load
$\rho_i \chi_I h$, whose prior has the scale $\lambda_i$ and the pointwise
standard deviation $\rho_i$ times that of $h$. The signal and the noise are
both loads. The Love numbers take the noise load to the coefficients.

**Estimators and posterior.** The simple GMSL estimate is the latitude-weighted
mean of the altimetry data. Its error has the form above, with $B$ the
GMSL. The joint data are the altimetry and GRACE data, with independent
noise. The normal equations are solved by conjugate gradients with a
surrogate Woodbury preconditioner. The surrogate is a cheaper problem:

- truncation at degree 32;
- one iteration of the sea level equation;
- the response to the ice load alone, without that to the ocean loads;
- the prior's product at the same amplitudes, with no correlation, mask or
  constraint;
- the local altimetry noise only.

Its Woodbury inverse $W$ is applied as $0.9 W + 0.1 R_s^{-1}$. The
preconditioner changes only the number of iterations, not the result.

## Revised configuration (run_all.sh)

The sections above state the configuration of the submitted paper. The
options that `run_all.sh` now passes change the following.

**Mass sources in the ocean experiments.** With `--land-water` the model is
$m = (h, w, \eta_d, \delta\rho)$, where $w$ is the change in the water
stored on the land outside the two ice sheets, a thickness of water with
the load $\rho_w \chi_L w$. With `--source-order 0.25`, $h$ and $w$ lie in
$H^{1/4}$, the space of the loads, and have priors of their own: the
coefficients of a sample have the variances

$$\mathrm{var}(u_{lm}) = \alpha_k \left(1 + \lambda_k^2 \frac{l(l+1)}{a^2}\right)^{-t} ,$$

a spectrum stated on $L^2$, with $t = 1.5$, $\lambda_i = 180$ km and
$\lambda_w = 340$ km. This is a measure on $H^{1/4}$ because $t$ exceeds
$1/4$ by more than one. The pointwise standard deviations are stated for
loads resolved to degree 180: 80 mm of equivalent water thickness for the
ice load, so $h$ has that times $\rho_w / \rho_i$, and 65 mm for $w$. Past
degree 180 the spectrum continues by the same law, so the prior does not
depend on the truncation of the model. $\eta_d$ and $\delta\rho$ are as
above.

**The ocean's fields.** With `--ocean-prior steric`, which `run_all.sh`
passes, the ocean's second field is the steric height $\eta_s$ itself, in
place of the density, and the two ocean fields lie in $H^{5/4}$, the space
of the response. Their prior is stated on two independent fields with
spectra of the form above: $u_s$, the steric height of deep water (46 mm,
280 km, $t = 3$; the year-on-year changes of EN4), and $u_m$, a manometric
sea level (8 mm, 330 km, $t = 2.5$; the ocean component of the ESA Earth
System Model). Then

$$\eta_d = u_s + u_m , \qquad \eta_s = w\, u_s , \qquad w = 1 - e^{-\eta_0 / D_0} ,$$

with $\eta_0$ the depth of the water and $D_0 = 500$ m
(`--steric-depth-scale-m`; `common.steric_depth_share`): the steric height
of a column falls away over shallow water as EN4's changes do, the sea
level does not, and the load takes up the difference. The manometric part
is $\eta_d - \eta_s = u_m + (1 - w) u_s$, and the constraint on the
ocean's mass, which is on this, involves both fields. The product with $w$
is truncated to the degrees of the space. With `--steric-depth-scale-m 0`,
the scripts' own default, $w = 1$. The two fields have the correlation
$(1 + \sigma_m^2 / \sigma_s^2)^{-1/2} = 0.985$ at a point, whatever $w$.

In the GRACE experiments with `--ocean-case removed` or `partly` a model of
the ocean is taken as removed from the data, and the prior's $u_m$ has the
case's standard deviation and scale in place of its own
(`common.OCEAN_CASES`). The load over shallow water is left in the same
proportion: with $f$ the ratio of the case's standard deviation to the full
one (0.025 and 0.25), the ocean's load is that of $u_m + f (1 - w) u_s$,
which the scripts obtain by taking $1 - f (1 - w)$ for $w$. The steric
height itself is not seen by GRACE.

**The ice prior, modulated.** With `--ice-modulation margin`, which
`run_all.sh` passes, the ice is $h = \gamma u$: $u$ the invariant field just
described and $\gamma$ a positive function of position, so that the pointwise
standard deviation of $h$ is $\gamma$ times that of $u$ and its correlation is
that of $u$ (`common.ice_modulation`). Over the grounded ice of each ice
sheet

$$\gamma \propto 1 + a p ,$$

with $p$ the nearness of a point to the edge of the grounded ice: the heat
kernel at the ice prior's own scale $\lambda_i$ applied to the indicator of
everything that is not grounded ice sheet, which is nought deep in the
interior and about a half at the edge. $a = 10$ (`--ice-margin-growth`).
For each sheet $\gamma$ is scaled so that the root mean square of the standard
deviation over its grounded ice is the sheet's own: 155 mm of equivalent
water thickness for Greenland and 60 for Antarctica
(`--ice-greenland-std-mm`, `--ice-antarctica-std-mm`). The floating ice
takes the interior's value, and $\gamma = 1$ off the ice sheets, where the load
operator discards the field. The three numbers are from the year-on-year
changes of the land ice of the ESA Earth System Model, 1995 to 2006: the
root mean square over the grounded ice of each sheet, and a least-squares
fit of the profile to the root mean square at each point. The product $\gamma u$
is truncated to the degrees of the space. The ice is independent of the
other fields and absent from the constraint on the ocean's mass, so the
prior is the image of the unmodulated one under $\gamma$ acting on the ice
alone. `--ice-modulation none` gives the invariant field with the single
standard deviation above.

**The load of the GRACE experiments.** With `--load-prior model` the
unknown of the GRACE experiments is still one field, the direct load
$\zeta$, now in $H^{1/4}$, and it is not divided into its causes. Its prior
is the image of the prior $N(0, Q)$ of the fields above under the load
operator $L$ that takes them to their direct load,

$$\zeta = L m \sim N(0, L Q L^*) ,$$

and the true load is the load of a sample of the fields. GRACE's data
depend on the fields through $\zeta$ alone, so the posterior of $\zeta$ is
the image under $L$ of the posterior the fields would have, and every
quantity that depends on the load (the regional averages of the total
load, its coefficients) has the same posterior in both forms. The
experiments therefore share one model. `--load-prior sobolev` keeps the
prior of the submitted paper. The covariance maps, which need the load's
value at a point, use its average over a cap of radius 2 degrees, since
point values are not defined on $H^{1/4}$.

**Regional averages and WMB's estimate of them.** In `grace_bias.py`,
`grace_inversion.py` and `grace_coverage.py` the averages are over the
regions themselves: $B$ has the region masks as weights, unsmoothed
(`--smoothing-scale-km` smooths them, as in the submitted paper). WMB's
estimate is made as it is in practice (`common.wmb_estimator`): the
coefficients are filtered, taken to a load by the load Love numbers as in
$\hat\sigma$ above, averaged over the region, and rescaled,

$$\hat q_j = s_j \, b_j^T \, W_{\mathrm{L}} \, F \, d ,$$

with $F$ the filter, $W_{\mathrm{L}}$ the map from coefficients to a load,
$b_j$ the average over region $j$ and $s_j$ the factor that makes the
estimate right for a load uniform over the region and absent elsewhere
(Velicogna and Wahr, 2006; `--wmb-rescale`). The filter is the
decorrelating filter DDK5 of Kusche et al. (2009) on all the coefficients
(`--wmb-filter ddk`, `--wmb-ddk-level`; `ddk_filter.py`), the default, or
the Gaussian filter of Wahr et al. (1998) of half-width 300 km on the
coefficients to degree 60 (`--wmb-filter gaussian`, `--wmb-radius-km`,
`--wmb-degree`). The filter and the factor are part of the estimator, so
what they cost in leakage counts as its error, $(CA - B)\zeta$ with $C$ the
matrix of the estimator. The WMB map of `grace_inversion.py` and
`grace_coverage.py` is $W_{\mathrm{L}} F d$, the filtered load with no
averaging. DDK5 is the default because it is the better of the two on
these data.

**GRACE data and noise.** The data are the coefficients of degrees 2 to 96,
every degree of the solutions the noise is taken from. Their noise is the formal error covariance of the
ITSG-Grace2018 solutions for a difference of two annual means,

$$R = \frac{1}{144} \sum_{i=1}^{24} N_i^{-1} ,$$

with $N_i$ the normal matrix of the unconstrained solution to degree 96 of
month $i$ of 2007 and 2008, the months taken as independent
(`grace_covariance.py`). For data cut off at a lower degree $R$ is the block of that
matrix: the covariance of those coefficients within the solution to degree
96. A lower cut-off is not used: at degree 60 the best-determined
coefficients still have noise about equal to the prior's signal, and
discarding the degrees above makes the posterior covariance ring. It is a full matrix, with the dependence on order and the correlations
of a single pair of satellites, and it is not rotationally invariant.
`--coefficient-errors` selects it in the GRACE experiments and
`--grace-coefficient-errors` in the joint experiment.

**Error of the GIA correction.** With `--gia-years 1`, the default, the
data of every experiment carry a further error of low rank, that of the
correction for glacial isostatic adjustment over one year. It comes from
the covariance of the present-day rates of the Stokes coefficients,
degrees 2 to 89, that Caron et al. (2018) derive from more than 100,000
forward models weighted by their fit to sea level records and uplift rates
(`gia_statistics.py`, `--gia-model caron`). That covariance has low rank
to rounding, and its leading 50 eigenvectors (`--gia-rank`), which hold
all but a part in 100,000 of its trace, each times the root of its
eigenvalue, are the columns of a factor $G$ with the error's covariance
$G G^T$:

- on GRACE's coefficients $G$ is those vectors, in the data's convention.
  Over one year the error exceeds the formal noise at degrees 2 to 6 and
  is below a quarter of it by degree 15;
- on the altimetry it is their geoid at the points, minus the potential
  change over $g$: 0.05 mm at the median point. The uniform change of the
  sea surface that goes with the uplift of the sea floor is left out, its
  covariance with the coefficients not being published; it is of the
  order of 0.05 mm over a year, against the 1 mm of the altimetry's own
  common error;
- in the joint experiment the two parts share their columns, so the one
  model's error is correlated between the two sets of data.

In the GRACE experiments $G G^T$ is added to the coefficients' covariance,
which is held as a matrix already. With `--gia-model ensemble` the error is
instead the spread, about their mean, of the rates of an ensemble of GIA
models (`gia_ensemble.py`), which `--gia-scale 0.4` brings to the
uncertainty Prandi et al. (2021) use; so scaled it agrees with the
published covariance on GRACE's coefficients to within a few per cent at
degrees 4 to 8 and is a quarter to a third smaller at degrees 2 and 3.

**Altimetry noise.** As stated above.

**Preconditioners.** GRACE's coefficients number a few thousand, so with
the full $R$ the GRACE experiments assemble the normal matrix
$A Q A^* + R$ of the true operator, at the cost of one application of
$A Q A^*$ per coefficient, and factorise it
(`common.exact_preconditioner`). The conjugate-gradient solves then
converge in an iteration or two.

The signal part $S = A Q A^*$ need not be assembled in full. With
$R = L L^T$, the signal in the metric of the noise is
$M = L^{-1} S L^{-T}$, and the normal matrix is $L (M + 1) L^T$: only the
eigenvalues of $M$ that are not small beside one matter, and their number
is that of the directions in which the data resolve the signal above the
noise, a property of the data and not of the model's resolution. With
`--grace-block-rank` $k$ (`common.sketched_signal`) $M$ is sketched from
$k$ applications of $S$, to the probes $L^{-T} g_i$ with $g_i$ white, and
replaced by its randomised Nyström approximation of rank $k$ (Frangella et
al., 2023), $U E U^T$; the block is $L (U E U^T + 1) L^T$, and the exact
one is the case of $k$ the number of data. At degree 256 with data to
degree 96, 1944 of the 9405 eigenvalues of $M$ exceed one and 2966 exceed
0.1; at degree 255 a sketch of rank 2400 gives 8 iterations to a relative
residual of $10^{-6}$ in place of one, and jointly with the altimetry 12 in
place of 9, for a quarter of the applications.

The ocean experiments use blocks assembled in the data space
(`common.block_preconditioner`, `--preconditioner blocks`):

- GRACE's block is the normal matrix above, with the sea level equation,
  exact or sketched;
- the altimetry's block is a sparse approximate inverse of the covariance
  of $\eta_d$ between the points under its prior, with the local noise:
  the points are put in a coarse-to-fine order (each the farthest from all
  before it) and each is regressed on the $k$ nearest of those before it
  (`--altimetry-neighbours`, 40), which gives
  $T^{-1} \approx (1 - B)^T F^{-1} (1 - B)$ with $B$ sparse and strictly
  lower triangular and $F$ diagonal (Vecchia, 1988; Schäfer et al., 2021;
  `vecchia.py`). It is positive definite by construction and has no
  cut-off to choose. The order and the neighbours are found with a tree,
  so that the build is linear in the number of points. At degree 255 with
  34,216 points it takes the altimetry solve from 22 iterations to 8; with
  the points at 0.5 and 0.25 degrees (137,000 and 549,000) the solve takes
  16 and 64;
- with `--altimetry-neighbours 0` the altimetry's block is instead that
  covariance tapered to vanish beyond 1500 km by the function of Gaspari
  and Cohn (1999), plus the local noise: a sparse matrix, factorised once;
- the two blocks are uncoupled;
- the parts of the noise of low rank are corrected for exactly by a
  Woodbury step: the error common to every point and the frame errors, the
  leading 40 modes of the large-scale part, and the error of the GIA
  correction.

The surrogate of the submitted paper remains as
`--preconditioner surrogate`.

The joint forward operator (`OceanPhysics.joint`) is the column of the
altimetry over GRACE with the sea level equation solved once for the two:
both sets of data are read from one response.

With $w = 1$ the prior covariance is applied on the spherical harmonic
coefficients (`common._steric_covariance`): the independent fields'
spectra, their mixing and the rank-one projection for the conservation of
mass are arithmetic there, between one analysis and one synthesis a field.
With the depth share it is applied as the image of the independent fields
under the mixing and the projection, the products with $w$ being made on
the grid; at degree 128 one action then takes about a quarter longer.

**Sensitivity to the prior, field by field** (`common.sensitivity_table`;
`--prior-sensitivity` in the inversion scripts, which produces these tables
and nothing else). The prior of the model is
the law of $T u$, where $u$ collects independent fields $u_f$ with
rotationally invariant covariances $Q_f$: with `--ocean-prior steric` the
ice, the land water, the steric height and the manometric sea level. $T = P
M$ in the ocean experiments, with $M$ the mixing of the independent fields
into the fields of the model (the sterodynamic sea level is the sum of the
two heights) and $P = 1 - Q C^* (C Q C^*)^{-1} C$ the projection that
conditioning on the conservation of the ocean's mass, $C m = 0$, amounts
to, $Q$ being the covariance of $M u$. In the GRACE experiments $T = L P
M$. With the ice modulated $T$ has a further factor, the multiplication by
$\gamma$ on the ice, after $P M$: the independent field is then the invariant
$u$, and the tables state the sensitivity to its prior with $\gamma$ fixed. So
the prior covariance is $T (\oplus_f Q_f) T^*$ exactly.

For one quantity with target kernel $t$, the posterior expectation is an
estimator whose error on a model $m$ with noise $z$ is $(k - t, m) +
(K^* t, z)$, $k - t = (K A - 1)^* t$. Write $\kappa_f$ for the part of
$T^*(k - t)$ in field $f$, and $s^2$ for the posterior variance. Then

$$s^2 = \sum_f (\kappa_f, Q_f \kappa_f) + (K^* t, R K^* t) ,$$

and if the fields are in truth drawn with covariances $Q_f + \Delta Q_f$,
mass being conserved through the same $P$, the variance of the error
differs from $s^2$ by $\sum_f (\kappa_f, \Delta Q_f \kappa_f)$, with

$$\frac{|(\kappa_f, \Delta Q_f \kappa_f)|}{s^2} \le
\frac{\|\Delta Q_f\|}{\|Q_f\|} F_f , \qquad
F_f = \frac{\|Q_f\| \, \|\kappa_f\|^2}{s^2} ,$$

in the operator norm or, more weakly, the Hilbert-Schmidt norm. The tables
give, for each quantity and field: the share $(\kappa_f, Q_f \kappa_f) /
s^2$ of the posterior variance that the field's prior supplies, which is
also the exact change per unit of a relative change of that prior's
variance at every scale; $F_f$ in the two norms; the noise's share; and
the sum of the shares less one, with $s^2$ computed separately as the
prior variance less the data's reduction of it, which checks the solves.
The norms of $Q_f$ are exact, from its spectrum; all norms and adjoints
are those of each field's own space.

Nothing in this is particular to the posterior expectation. Any linear
estimate $C d$ of $B m$ has the error operator $E = C A - B$, kernel error
$E^* e$ and noise part $(C^* e, R C^* e)$, and
`common.prior_sensitivity(error, gain, ...)` makes the same table from $E$
and $C$, the shares then being parts of whatever variance the estimate
states; `common.posterior_sensitivity` is the case $C = B K$, at one solve
of the normal equations per quantity. A second table
(`common.prior_perturbation_sizes`) gives $\|\Delta Q_f\| / \|Q_f\|$ for
a pointwise standard deviation 25 per cent larger, a scale 1.5 times
longer or shorter, and an order higher by a half. `--paper-bound` in
`grace_inversion.py` adds the factor of the bound stated on the load
space, $\|Q\|_{HS} \|k - t\|^2 / s^2$ with $Q$ the load's prior
covariance, its norm estimated with random probes.

**Maps for a load not drawn from the prior.** `grace_coverage.py
--plot-maps` takes one of the true loads of the ESA Earth System Model
(`--map-truth`), adds one draw of the noise to its data and solves for the
posterior as `grace_inversion.py` does: the figure has the true total
load, WMB's filtered load and the posterior mean, the pointwise posterior
standard deviation from `--std-samples` samples, and the two estimates'
errors. The squared misfit of the data to the prior's prediction, printed
with it, measures how far the load is from the prior.

**Truncation** (`truncation_check.py`). A model is drawn from the prior at
the highest of `--degrees`; its fields are truncated to each lower degree
(the coefficients to that degree), the forward problem of the joint
experiment is solved there with that degree's own grids and masks, and the
predicted data are set against those of the highest degree: the altimetry
at the same points, in mm and against the local error of a point; GRACE's
coefficients as the root mean square of the difference whitened by their
error covariance, which is one for a draw of the noise; and global mean sea
level, its two parts and the parts of a regional mean.

## References

Caron, L., Ivins, E. R., Larour, E., Adhikari, S., Nilsson, J. and Blewitt,
G. (2018). GIA model statistics for GRACE hydrology, cryosphere, and ocean
science. *Geophysical Research Letters* 45, 2203–2212.

Frangella, Z., Tropp, J. A. and Udell, M. (2023). Randomized Nyström
preconditioning. *SIAM Journal on Matrix Analysis and Applications* 44,
718–752.

Gaspari, G. and Cohn, S. E. (1999). Construction of correlation functions in
two and three dimensions. *Quarterly Journal of the Royal Meteorological
Society* 125, 723–757.

Gregory, J. M. et al. (2019). Concepts and terminology for sea level: mean,
variability and change, both local and global. *Surveys in Geophysics* 40,
1251–1289.

Heathcote, D. A. et al. (2026). A scalable Bayesian framework for modern
sea-level inference. *Geophysical Journal International*.

Kusche, J., Schmidt, R., Petrovic, S. and Rietbroek, R. (2009).
Decorrelated GRACE time-variable gravity solutions by GFZ, and their
validation using a hydrological model. *Journal of Geodesy* 83, 903–913.

Prandi, P. et al. (2021). Local sea level trends, accelerations and
uncertainties over 1993–2019. *Scientific Data* 8, 1.

Schäfer, F., Katzfuss, M. and Owhadi, H. (2021). Sparse Cholesky
factorization by Kullback-Leibler minimization. *SIAM Journal on Scientific
Computing* 43, A2019–A2046.

Vecchia, A. V. (1988). Estimation and model identification for continuous
spatial processes. *Journal of the Royal Statistical Society, Series B* 50,
297–312.

Velicogna, I. and Wahr, J. (2006). Measurements of time-variable gravity
show mass loss in Antarctica. *Science* 311, 1754–1756.

Wahr, J., Molenaar, M. and Bryan, F. (1998). Time variability of the
Earth's gravity field: hydrological and oceanic effects and their possible
detection using GRACE. *Journal of Geophysical Research* 103, 30205–30229.
