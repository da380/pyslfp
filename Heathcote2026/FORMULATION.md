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

**Altimetry noise.** The altimetry noise is the sum of two parts:

- a local part, independent at each point, with standard deviation 0.5 times
  that of $\eta_d$;
- a large-scale part, a field
  $N(0, \alpha_c (1 + \lambda_c^2 \Delta)^{-q})$ on $H^2$ evaluated at the
  points, with $\lambda_c = 2000$ km and a pointwise standard deviation of
  0.01 times that of $\eta_d$.

The large-scale part barely averages down, and so sets a floor on the error
of the GMSL estimate. The scripts report that floor.

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

## References

Gregory, J. M. et al. (2019). Concepts and terminology for sea level: mean,
variability and change, both local and global. *Surveys in Geophysics* 40,
1251–1289.

Heathcote, D. A. et al. (2026). A scalable Bayesian framework for modern
sea-level inference. *Geophysical Journal International*.
