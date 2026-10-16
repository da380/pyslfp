Units and non-dimensionalisation
================================

The library computes in a single internal frame: planetmodel's
``Scales.geophysical`` at the Earth's mean radius. Under these scales
the gravitational constant and the mean radius are one, lengths are
measured in Earth radii, densities against the Earth's mean, and the
time unit is :math:`1/\sqrt{G\bar\rho} \approx 1648` s. Every
``EarthModelParameters`` built from the defaults lives in this frame,
and the viscoelastic coupling requires the radial material to live in
it too.

SI belongs at the rim. Raw physical inputs — the constants fed to
``EarthModelParameters``, deck files, Love number tables, ice sheet
thicknesses — are given in SI and converted once on the way in; results
are converted back through ``parameters.scales.factor`` where SI output
is wanted. A planetmodel model authored in other units (an SI deck,
say) enters through ``model.converted(parameters.scales)`` before its
mesh and material are built; the coupling refuses a material whose
scales disagree with the model's, since a mixed-frame calculation
cannot be trusted. Love number tables convert themselves on being read,
whatever units they were written in.

Two conventions sit on top of the frame. Ice histories are dated in
kiloyears before present, the unit a glacial problem is naturally
posed in, and converted internally. And relaxation is not the frame's
business: Maxwell times of order :math:`10^7` in internal units are
expected and harmless, because time enters the evolution only through
the resolved products :math:`h = \Delta t / \tau`; the frame's time
scale instead keeps the elasto-gravitational operators of order one,
which is where conditioning lives.

One rule follows from working in any fixed frame: never drop a factor
from a formula because the current scheme makes it one. The surface
gravity, the gravitational constant, radii and density ratios are
written out everywhere, and the scheme makes them cheap rather than
the code making them absent. A dropped factor is invisible in the
frame it was dropped in and wrong in every other, and the test suite
holds covariance gates (the same body run under deliberately different
schemes, compared after conversion to SI) to keep it that way.
