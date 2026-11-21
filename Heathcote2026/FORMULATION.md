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

**GRACE data and noise.** The data are the coefficients of degrees 2 to 60
(at most 96). Their noise is the formal error covariance of the
ITSG-Grace2018 solutions for a difference of two annual means,

$$R = \frac{1}{144} \sum_{i=1}^{24} N_i^{-1} ,$$

with $N_i$ the normal matrix of the unconstrained solution to degree 96 of
month $i$ of 2007 and 2008, the months taken as independent
(`grace_covariance.py`). $R$ for degrees 2 to 60 is the block of that
matrix: the covariance of those coefficients within the solution to degree
96. It is a full matrix, with the dependence on order and the correlations
of a single pair of satellites, and it is not rotationally invariant.
`--coefficient-errors` selects it in the GRACE experiments and
`--grace-coefficient-errors` in the joint experiment.

**Error of the GIA correction.** With `--gia-years 1 --gia-scale 0.4`, the
altimetry and GRACE data of the ocean experiments carry a further error of
low rank: the spread, about their mean, of the present-day rates of an
ensemble of GIA models (`gia_ensemble.py`), over one year and scaled by
0.4. In the joint experiment it is one model's error in both sets of data,
and so is correlated between them.

**Altimetry noise.** As stated above.

**Preconditioners.** GRACE's coefficients number a few thousand, so with
the full $R$ the GRACE experiments assemble the normal matrix
$A Q A^* + R$ of the true operator, at the cost of one application of
$A Q A^*$ per coefficient, and factorise it
(`common.exact_preconditioner`). The conjugate-gradient solves then
converge in an iteration or two.

The ocean experiments use blocks assembled in the data space
(`common.block_preconditioner`, `--preconditioner blocks`):

- GRACE's block is the normal matrix above, with the sea level equation;
- the altimetry's block is the covariance of $\eta_d$ between the points
  under its prior, tapered to vanish beyond 1500 km by the function of
  Gaspari and Cohn (1999), plus the local noise: a sparse matrix,
  factorised once;
- the two blocks are uncoupled;
- the parts of the noise of low rank are corrected for exactly by a
  Woodbury step: the error common to every point and the frame errors, the
  leading 40 modes of the large-scale part, and the error of the GIA
  correction.

The surrogate of the submitted paper remains as
`--preconditioner surrogate`.

## References

Gaspari, G. and Cohn, S. E. (1999). Construction of correlation functions in
two and three dimensions. *Quarterly Journal of the Royal Meteorological
Society* 125, 723–757.

Gregory, J. M. et al. (2019). Concepts and terminology for sea level: mean,
variability and change, both local and global. *Surveys in Geophysics* 40,
1251–1289.

Heathcote, D. A. et al. (2026). A scalable Bayesian framework for modern
sea-level inference. *Geophysical Journal International*.
