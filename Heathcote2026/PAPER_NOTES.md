# Paper and code: points for the revision

This note checks Heathcote et al. (2026, GJI, arXiv 2608.22336v1) against
[FORMULATION.md](FORMULATION.md). It lists, by section, where the paper's
text or numbers differ from the formulation the scripts implement. It is
a working note for the revision. Delete it once the revision is done.

## Numbers that need recomputing

Four defects in the code behind the paper's runs change its results. Each is
fixed in the scripts.

1. **The GRACE noise model.** pyslfp's push-forward of a noise load onto
   the GRACE coefficients took the covariance's eigenvalues $\sigma_k$ as
   the coefficient variances. On the Sobolev load space the variances are
   $\sigma_k / g_k$, with $g_k = (1 + \lambda_s^2 l(l+1)/a^2)^2$ the
   metric's weight. The data noise was therefore $g_k$ times the stated one
   in variance: 56 times at degree 32, 709 at 64 and 4000 at 100. The
   amplitude signal-to-noise ratio in the data was the calibrated one
   divided by $\sqrt{g_k}$, so noise overtook the signal far below degree
   64. The runs were consistent for the noise they used, since the synthetic
   noise and the likelihood shared it. This touches every GRACE result:
   §3.1, §5.1 and the GRACE part of §5.2.
2. **The GRACE forward operator** (§5.1 only). The paper's forward problem
   (eq. 8 and §5.1.1) is the right one: GRACE observes the response to the
   direct load. The runs applied the forward operator to the total load,
   $\zeta + \rho_w C \xi$, which counts the induced ocean load twice. For
   prior samples at degree 64 this changes the data by 12–17 per cent.
   §3.1 used the right operator.
3. **The units of the joint GRACE noise** (§5.2 only). The GRACE noise must
   be solved against a signal in the noise's units, which is a load: the ice
   load, $\rho_i$ times the ice thickness. The runs used the water
   equivalent thickness, $\rho_i / \rho_w$ times the thickness. That made
   the noise $1/\rho_w$ times too large in the model's units, about 5.5
   times, and the amplitude signal-to-noise ratio at degree 2 about 1.8
   rather than 10, before the factor of item 1.
4. **The degree-1 labels** (§5.1.6). The degree-$l$ coefficients were
   labelled in the order $m = 0, -1, 1, \ldots$, but the vector holds
   $m = 0, 1, \ldots, l, -1, \ldots, -l$. The labels (1,1) and (1,−1) in
   Fig. 8, Fig. 10 and the degree-1 reductions are therefore swapped.

**What needs recomputing:**

- **§3.1 (Figs 1–2):** the nominal and true WMB error distributions, and
  the statements that WMB understates the error by factors of two to three
  and that the true error is about half the signal (item 1).
- **§5.1 (Figs 5–11):** items 1 and 2, with item 4 for the labels. This
  covers:
  - the KL divergences of 1.26–3.84 nats and the variance reductions of
    94–96 per cent;
  - the degree-1 reductions of 70.5, 76.9 and 75.0 per cent and the KL
    divergence of 1.97 nats;
  - the relative kernel errors of 15–21 per cent (Bayesian) and 35–40 per
    cent (WMB), 51–58 per cent at degree 1 and 4–6 per cent at degree 2;
  - the iteration counts.
- **§5.2 (Figs 13–18 and every number involving GRACE):** items 1 and 3. With
  both corrected the gravity data are much stronger. This covers:
  - GMSLR: the GRACE-only reduction of 69.9 per cent and 0.83 nats, and the
    joint 99.4 per cent;
  - the (b, s) plane: 88.4 and 9.2 per cent (GRACE), 94.5 and 83.2 per cent
    (joint), 3.13 nats, the correlation of −0.93, and
    $\sigma_{b+s} \approx 0.06$ mm and $\sigma_{b-s} \approx 0.29$ mm;
  - the Tasman Sea: 88.5 per cent (GRACE), 94.3 and 39.4 per cent (joint),
    the generalised variance, and 3.76 nats.

  The altimetry-only results are unaffected.

Item 2 alone, in quick mode (degree 64, data to degree 32), against a run
with item 1 already corrected:

- The posterior variances of the regional averages rise from 2.2–2.9 to
  about 4.1 mm²: counting the induced load twice made the data look more
  informative than they are.
- The degree-1 variance reductions fall from 56–64 to 21–27 per cent. The
  induced ocean load is what couples degree 1 to the observed degrees, and
  counting it twice doubled that coupling. The degree-one recovery of
  §5.1.6 (reductions of 70.5–76.9 per cent, 1.97 nats) is the claim most
  likely to weaken.
- The WMB kernel ratios fall from 0.35–0.46 to 0.31–0.37.
- The Bayesian ratios change from 0.46–0.60 to 0.47–0.52, above WMB's. With
  degree 1 removed from the prior they fall to 0.29–0.33.

This reverses the paper's ordering at full resolution. Quick mode stops the
data at degree 32, so the comparison may not carry over. It needs the full run
before the text of §5.1.7 is revised.

Item 3 alone, in quick mode, against a run with item 1 already corrected:
the GRACE-only barystatic variance reduction rises from 90.3 to 99.3 per cent,
and the GMSLR reduction from 70.1 to 83.5 per cent.

**Section 3.1.4, the GRACE noise.** The paper says the noise load has a
pointwise standard deviation of 3.76 mm, with an amplitude signal-to-noise
ratio of 0.75 at degree 100. The stated parameters do not give these values:
$q = 3$, $\lambda = 250$ km, $\lambda_n = 125$ km, $\sqrt{s_2} = 10$ and a
crossover at 64. They give $q_n = 1.44$, a pointwise standard deviation of
1.28 mm at degree 256, and a ratio of 0.48 at degree 100. The scripts,
before and after the rewrite, agree with these values. The text's figures
seem to come from another parameter set.

**The solver tolerance and the posterior variances.** The paper stops the
conjugate gradient solves at a relative residual of 0.01 times the
noise-to-signal ratio (§5.1.2). That is enough for the posterior expectation.
A posterior variance, however, is the prior variance less a correction
computed through the solves. Its error relative to itself is then about the
tolerance divided by the ratio of the posterior variance to the prior. With
the GRACE noise sized correctly, the joint inversion reduces some variances
by more than 99 per cent. At the paper's tolerance these come out wrong by
10–20 per cent; at degree 64, for example, the joint barystatic variance
exceeds the GRACE-only one. The scripts therefore push the posterior onto the
quantities of interest at a tighter tolerance. The expectation and the
samples keep the paper's criterion. The text should state both.

**§5.1.3, Fig. 5.** The noise in the unsmoothed WMB estimate is the
inflated noise of item 1. In this noise model the noise is a load, which the
Love numbers take to the coefficients, so WMB's division by $k_l$ returns
that noise load cut at degree 100 and amplifies nothing. At Fig. 5's
settings (degree 256, data to degree 100, one draw) the noise in the
unsmoothed estimate has an rms of 1.27 mm with the corrected spectrum, against
5.97 mm with the inflated one; the true total load's rms is 9.26 mm. With the
corrected noise the unsmoothed estimate is close to the truth, with no
short-wavelength noise to speak of. The figure, and the text attributing the
noise to the division by the Love numbers, need revising. A noise model whose
coefficient errors grow with degree would bring that amplification back.
`grace_inversion.py` therefore drops the smoothed WMB panel. In its place is
the pointwise posterior standard deviation of the total load, from
`--std-samples` randomise-then-optimise samples (100 in `run_all.sh`). §5.1.4,
which says the map is not shown, changes with it.

## Text to revise

**§3.1.4 and §3.2.4, the covariances.** $Q = \alpha(1 + \lambda^2\Delta)^{-q}$
is the covariance on $H^2$, but $H^2$ needs a length scale. The scripts use
500 km. Coefficient variances then carry a further factor
$(1 + \lambda_s^2 l(l+1)/a^2)^{-2}$, and the samples are smoother than $q$
alone suggests. The text should state the scale and say on which space the
covariance acts. The same holds for $Q_n$ and for the correlated altimetry
noise.

**§3.2.4, masking.** The paper pushes the prior forward through the masks $P$
and then conditions on $Dm = 0$. A masked field is discontinuous and so not in
$H^2$. $PQP^*$ is therefore not a covariance on the model space $M$, and the
masked fields are not elements of it. The formulation keeps the fields global
and the prior unmasked, conditioned on $Dm = 0$. The supports enter only
through the loads and the observation operators. Because $D$ carries the ocean
function, the restriction of this prior to the supports is the paper's
prior. All in-support statistics and quantities of interest are therefore
unchanged. Fields are masked only for display.

**§3.2.4, regularity.** The paper says the loads are in $L^\infty$ and $\xi$
is in $W^{1,p}$. The Hilbert-space statement that matches the computations
is this: the loads lie in $H^s$ for $0 \le s < 1/2$, and the response in
$H^{s+1}$. Point evaluation is continuous because $s + 1 > 1$. The scripts
take $s = 1/4$.

**§5.1.4, degree 1.** The text says the degree-one component is removed
from the prior. The option conditions the prior on the degree-1
coefficients alone. The runs behind the paper also removed degree 0 (the net
mass of the direct load). The claim that the posterior covariance functions
then localise tightly should be rechecked with degree 1 alone. Degree 0 of the
potential is never observed, since GRACE starts at degree 2.

**§5.2.2, the surrogate.** The text says the surrogate removes the
gravitational, rotational and deformational (GRD) contribution to the
geocentric sea level change. The surrogate keeps the GRD response to the
ice load, which the GRACE part needs, and drops that to the ocean loads.
Its prior also drops the correlation between $\eta_d$ and $\delta\rho$. The
preconditioner applied is $0.9P + 0.1R_s^{-1}$, not $P$ of eq. (63) alone.

**§5.2.3, the steric part.** The paper takes $s$ as the residual
GMSLR − $b$. The scripts compute $s = \langle\eta_s\rangle_O$ directly and
report GMSLR − $b$ − $s$ as a check on mass balance. The two agree to
rounding, so this is a choice of wording only.

**The ice-sheet mask.** The paper's mask is the grounded ice sheets. The
library defines the ice sheets as all ice south of 60°S plus all ice in
GTN-G region 5. The peripheral glaciers there are included, and the factor
$(1 - C)$ removes floating ice from the load. The support differs slightly
from that of the paper's runs.

## Terminology

The paper follows Gregory et al. (2019). The library and the scripts move to
the same names:

- geocentric sea level change, not sea surface height;
- sterodynamic sea level change, not dynamic topography;
- relative sea level change.

The paper's own terms, dynamic and barystatic manometric sea level change,
are already in use.
